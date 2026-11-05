use ptu_plan::agglomeration::{self, AgglomerationOptions};
use ptu_plan::analysis::{NetworkId, analyze_program};
use ptu_plan::interaction::InteractionGraph;
use ptu_plan::learning::features::*;
use ptu_plan::learning::model::*;
use ptu_plan::learning::policy::learned_agglomerate;
use ptu_plan::learning::rollout::*;
use ptu_plan::learning::training::*;
use ptu_plan::master::{MasterOptions, MasterStatus, initial_partitions, solve_sequential};
use ptu_plan::parsing::json::Program;
use ptu_plan::partition::{SpectralOptions, micropartition_network};
use ptu_plan::solver::*;
use ptu_plan::store::*;
use std::process::Command;

fn fixture() -> Program {
    serde_json::from_str(include_str!("fixtures/solver.json")).unwrap()
}
fn observation() -> Observation {
    Observation {
        hilg: None,
        feature_version: FEATURE_VERSION,
        nodes: vec![
            ClusterInput {
                id: 8,
                features: vec![0.4; NODE_FEATURES],
            },
            ClusterInput {
                id: 12,
                features: (0..NODE_FEATURES).map(|i| i as f64 / 10.0).collect(),
            },
        ],
        relations: vec![
            Relation {
                source: 0,
                target: 1,
                channel: 2,
                weight: 1.0,
            },
            Relation {
                source: 1,
                target: 0,
                channel: 3,
                weight: 1.0,
            },
        ],
        global: vec![0.5; GLOBAL_FEATURES],
        context: vec![0.2; CONTEXT_FEATURES],
        actions: vec![
            ActionInput {
                pair: Some((0, 1)),
                features: vec![0.3; PAIR_FEATURES],
            },
            ActionInput {
                pair: None,
                features: vec![0.0; PAIR_FEATURES],
            },
        ],
    }
}
fn labels() -> Supervision {
    Supervision {
        preferences: vec![(1, 0)],
        values: vec![
            ValueTarget {
                action: 0,
                log_effort: 3.5,
                lower_bound: false,
            },
            ValueTarget {
                action: 1,
                log_effort: 1.2,
                lower_bound: false,
            },
        ],
    }
}

fn sync_finish_events(record: &mut RolloutRecord) {
    for outcome in &mut record.outcomes {
        let result = outcome.result.as_ref().unwrap();
        for event in &mut outcome.events {
            if let ptu_plan::instrumentation::MasterEvent::Finished { status, statistics } = event {
                *status = result.status;
                *statistics = result.statistics.clone();
            }
        }
    }
}

#[test]
fn all_shared_encoder_and_head_gradients_match_finite_differences() {
    let mut model = MergeModel::new(
        ModelConfig {
            hidden: 2,
            layers: 1,
        },
        13,
    )
    .unwrap();
    let observation = observation();
    let labels = labels();
    let (loss, gradient) = model.loss_gradient(&observation, &labels, 0.7).unwrap();
    assert_eq!(loss, model.loss(&observation, &labels, 0.7).unwrap());
    assert!(gradient[..NODE_FEATURES * 2].iter().any(|g| g.abs() > 1e-5));
    for (i, &analytic) in gradient.iter().enumerate() {
        let original = model.parameters[i];
        let epsilon = 1e-6;
        model.parameters[i] = original + epsilon;
        let plus = model.loss_gradient(&observation, &labels, 0.7).unwrap().0;
        model.parameters[i] = original - epsilon;
        let minus = model.loss_gradient(&observation, &labels, 0.7).unwrap().0;
        model.parameters[i] = original;
        let numeric = (plus - minus) / (2.0 * epsilon);
        assert!(
            (numeric - analytic).abs() < 2e-5 * (1.0 + numeric.abs()),
            "parameter {i}: {analytic} vs {numeric}"
        );
    }
}

#[test]
fn graph_encoding_is_permutation_invariant_and_preserves_edge_direction() {
    let model = MergeModel::new(
        ModelConfig {
            hidden: 3,
            layers: 2,
        },
        17,
    )
    .unwrap();
    let observation = observation();
    let expected = model.predict(&observation).unwrap();
    let mut permuted = observation.clone();
    permuted.nodes.swap(0, 1);
    for relation in &mut permuted.relations {
        relation.source = 1 - relation.source;
        relation.target = 1 - relation.target;
    }
    let actual = model.predict(&permuted).unwrap();
    for (a, b) in expected
        .logits
        .iter()
        .chain(&expected.log_effort)
        .zip(actual.logits.iter().chain(&actual.log_effort))
    {
        assert!((a - b).abs() < 1e-10);
    }
    let mut reversed = observation;
    for relation in &mut reversed.relations {
        std::mem::swap(&mut relation.source, &mut relation.target);
    }
    let reversed = model.predict(&reversed).unwrap();
    assert!(
        expected
            .logits
            .iter()
            .zip(reversed.logits)
            .any(|(a, b)| (a - b).abs() > 1e-6)
    );
}

#[test]
fn censored_values_only_penalize_underestimation_and_shapes_are_checked() {
    let model = MergeModel::new(
        ModelConfig {
            hidden: 2,
            layers: 1,
        },
        19,
    )
    .unwrap();
    let observation = observation();
    let labels = Supervision {
        preferences: vec![],
        values: vec![ValueTarget {
            action: 0,
            log_effort: 0.0,
            lower_bound: true,
        }],
    };
    let (loss, gradient) = model.loss_gradient(&observation, &labels, 1.0).unwrap();
    assert_eq!(loss, 0.0);
    assert!(gradient.iter().all(|g| *g == 0.0));
    let mut bad = observation.clone();
    bad.relations[0].channel = RELATIONS;
    assert!(model.predict(&bad).is_err());
    let mut bad = model.clone();
    bad.parameters.pop();
    assert!(bad.predict(&observation).is_err());
    let mut bad = model;
    bad.parameters[0] = f64::NAN;
    assert!(bad.validate().is_err());
}

#[test]
fn adam_training_learns_ranking_and_resume_is_deterministic() {
    let options = TrainOptions {
        epochs: 80,
        batch_size: 1,
        learning_rate: 0.01,
        model: ModelConfig {
            hidden: 3,
            layers: 1,
        },
        ..Default::default()
    };
    let examples = vec![Example {
        id: "fixture".into(),
        group: "group".into(),
        observation: observation(),
        labels: labels(),
    }];
    let mut model = MergeModel::new(options.model, 42).unwrap();
    let mut optimizer = Adam::new(model.parameters.len());
    let initial = model
        .loss_gradient(&examples[0].observation, &examples[0].labels, 1.0)
        .unwrap()
        .0;
    let original = model.clone();
    fit(&mut model, &mut optimizer, &examples, &options, 0).unwrap();
    let final_loss = model
        .loss_gradient(&examples[0].observation, &examples[0].labels, 1.0)
        .unwrap()
        .0;
    assert!(final_loss < initial * 0.25, "{initial} -> {final_loss}");
    assert_eq!(model.choose(&examples[0].observation, 0.0).unwrap().0, 1);
    let mut resumed = original;
    let mut state = Adam::new(resumed.parameters.len());
    let half = TrainOptions {
        epochs: 40,
        ..options
    };
    fit(&mut resumed, &mut state, &examples, &half, 0).unwrap();
    let bytes = serde_json::to_vec(&(resumed, state)).unwrap();
    let (mut resumed, mut state): (MergeModel, Adam) = serde_json::from_slice(&bytes).unwrap();
    fit(&mut resumed, &mut state, &examples, &half, 40).unwrap();
    for (a, b) in model.parameters.iter().zip(resumed.parameters) {
        assert!((a - b).abs() < 1e-10);
    }
}

#[test]
fn counterfactuals_stop_labels_and_full_oracle_witnesses_round_trip() {
    let root = tempfile::tempdir().unwrap();
    let store = DatasetStore::open(root.path()).unwrap();
    let options = RolloutOptions {
        trace: ptu_plan::instrumentation::TraceLevel::Full,
        ..Default::default()
    };
    let entries = collect_program(&store, &fixture(), None, &options).unwrap();
    assert_eq!(entries.len(), 2);
    for entry in &entries {
        audit_record(&store, entry).unwrap();
    }
    let record: RolloutRecord = store.load(&entries[0]).unwrap();
    assert_eq!(record.observation.actions.len(), 2);
    assert!(record.observation.actions.last().unwrap().pair.is_none());
    assert!(
        record
            .outcomes
            .iter()
            .all(|o| o.result.as_ref().unwrap().status == MasterStatus::Feasible)
    );
    let labels = record.supervision(2.0).unwrap();
    assert!(!labels.preferences.is_empty());
    let mut missing_witness = record.clone();
    for event in &mut missing_witness.outcomes[0].events {
        if let ptu_plan::instrumentation::MasterEvent::OracleFinished { certificate, .. } = event {
            *certificate = None;
        }
    }
    assert!(missing_witness.validate().is_err());
    let mut missing_finish = record.clone();
    missing_finish.outcomes[0].events.pop();
    assert!(missing_finish.validate().is_err());
    let mut unknown = record.clone();
    for outcome in &mut unknown.outcomes {
        let result = outcome.result.as_mut().unwrap();
        result.status = MasterStatus::BudgetExhausted;
        result.certificate = None;
    }
    sync_finish_events(&mut unknown);
    let labels = unknown.supervision(1.0).unwrap();
    assert!(labels.preferences.is_empty());
    assert!(labels.values.iter().all(|v| v.lower_bound));
    let mut abandoned = unknown.clone();
    for outcome in &mut abandoned.outcomes {
        outcome.result.as_mut().unwrap().statistics.local_unknown = 1;
    }
    sync_finish_events(&mut abandoned);
    assert!(abandoned.supervision(1.0).unwrap().values.is_empty());
    let mut infeasible = unknown;
    for outcome in &mut infeasible.outcomes {
        outcome.result.as_mut().unwrap().status = MasterStatus::SequentiallyInfeasible;
    }
    sync_finish_events(&mut infeasible);
    let labels = infeasible.supervision(1.0).unwrap();
    assert!(labels.values.is_empty());
    assert!(labels.preferences.is_empty());
}

#[test]
fn learned_merges_remain_legal_and_solver_certificates_replay() {
    let program = fixture();
    let analysis = analyze_program(&program).unwrap();
    let graph = InteractionGraph::from_analysis(&analysis);
    let seeds = micropartition_network(
        &graph,
        NetworkId::Initial,
        &SpectralOptions {
            max_cluster_size: 1,
            ..Default::default()
        },
    )
    .unwrap()
    .clusters;
    let model = MergeModel::new(
        ModelConfig {
            hidden: 3,
            layers: 1,
        },
        42,
    )
    .unwrap();
    let (state, trace) = learned_agglomerate(
        &analysis,
        &graph,
        &seeds,
        &AgglomerationOptions::default(),
        &model,
        0.1,
    )
    .unwrap();
    let valid = agglomeration::observe(
        &analysis,
        &graph,
        NetworkId::Initial,
        &state.clusters,
        &AgglomerationOptions::default(),
    )
    .unwrap();
    assert_eq!(valid.clusters.len(), state.clusters.len());
    assert!(!trace.decisions.is_empty());
    let grounded = ground_reachable_program(&program, &GroundingOptions::default()).unwrap();
    let result = solve_sequential(
        &grounded,
        &initial_partitions(&graph, &state.clusters).unwrap(),
        &MasterOptions::default(),
    )
    .unwrap();
    validate_problem_certificate(&grounded, &result.certificate.unwrap()).unwrap();
}

#[test]
fn collect_train_and_planner_clis_work_without_external_tools() {
    let root = tempfile::tempdir().unwrap();
    let store_path = root.path().join("store");
    let checkpoint = root.path().join("model.json");
    let output = Command::new(env!("CARGO_BIN_EXE_collect-rollouts"))
        .args(["tests/fixtures/solver.json", "--store"])
        .arg(&store_path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let store = DatasetStore::open(&store_path).unwrap();
    let group = store
        .snapshot()
        .unwrap()
        .entries
        .into_iter()
        .find_map(|e| e.metadata.group_id)
        .unwrap();
    let seed = (0..100)
        .find(|&seed| group_split(&group, seed) == Split::Train)
        .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_train-merge-model"))
        .arg("--store")
        .arg(&store_path)
        .arg("--output")
        .arg(&checkpoint)
        .args([
            "--epochs",
            "3",
            "--hidden",
            "2",
            "--layers",
            "1",
            "--split-seed",
            &seed.to_string(),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let output = Command::new(env!("CARGO_BIN_EXE_ptu-plan"))
        .args([
            "tests/fixtures/solver.json",
            "--sequential",
            "--agglomerate",
            "--micro-max-size",
            "1",
            "--merge-model",
        ])
        .arg(&checkpoint)
        .args([
            "--analysis-json",
            "--master-trace",
            "--merge-beam",
            "--beam-depth",
            "2",
            "--beam-max-evaluations",
            "20",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert!(value.get("merge_policy").is_some());
    assert!(
        value["merge_policy"]["beam"]["evaluations"]
            .as_u64()
            .unwrap()
            > 0
    );
    assert_eq!(value["sequential"]["status"], "Feasible");
    assert!(
        value["master_events"]
            .as_array()
            .unwrap()
            .iter()
            .any(|event| {
                event
                    .get("OracleStarted")
                    .is_some_and(|start| !start["request"].is_null())
            })
    );
    let output = Command::new(env!("CARGO_BIN_EXE_dataset-store"))
        .arg("--store")
        .arg(store_path)
        .arg("audit")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn domain_splits_and_source_groups_are_enforced_during_training() {
    let directory = tempfile::tempdir().unwrap();
    let store = DatasetStore::open(directory.path()).unwrap();
    let mut found = std::collections::BTreeSet::new();
    for i in 0..200 {
        let mut program = fixture();
        program.domain.name = format!("split-{i}");
        program.problem.as_mut().unwrap().domain_name = program.domain.name.clone();
        let group = ptu_plan::training::sha256(&serde_json::to_vec(&program.domain).unwrap());
        let split = format!("{:?}", group_split(&group, 42));
        if found.insert(split) {
            collect_program(&store, &program, None, &RolloutOptions::default()).unwrap();
        }
        if found.len() == 3 {
            break;
        }
    }
    assert_eq!(found.len(), 3);
    let options = TrainOptions {
        epochs: 2,
        model: ModelConfig {
            hidden: 2,
            layers: 1,
        },
        ..Default::default()
    };
    let (entry, _, report) = train_store(&store, &options, None).unwrap();
    assert_eq!(report.training.records, 2);
    assert_eq!(report.validation.records, 2);
    assert_eq!(report.test.records, 2);
    let checkpoint: Checkpoint = store.load(&entry).unwrap();
    checkpoint.validate().unwrap();
    assert_eq!(checkpoint.training_groups.len(), 1);
    assert_eq!(checkpoint.validation_groups.len(), 1);
    let mut bad = checkpoint.clone();
    bad.validation_groups = bad.training_groups.clone();
    assert!(bad.validate().is_err());
    let mut bad = checkpoint.clone();
    bad.validation_records.push(bad.training_records[0].clone());
    assert!(bad.validate().is_err());
    let mut bad = checkpoint.clone();
    bad.options.model.hidden += 1;
    assert!(bad.validate().is_err());
    let mut bad = checkpoint.clone();
    bad.options.learning_rate = f64::NAN;
    assert!(bad.validate().is_err());
    let source = store
        .snapshot()
        .unwrap()
        .entries
        .into_iter()
        .find(|e| e.metadata.kind == RecordKind::Rollout)
        .unwrap();
    let mut record: RolloutRecord = store.load(&source).unwrap();
    record.group_id = ptu_plan::training::sha256(b"false-group");
    store
        .put(
            Metadata {
                group_id: Some(record.group_id.clone()),
                ..source.metadata
            },
            &record,
        )
        .unwrap();
    assert!(
        train_store(&store, &options, None)
            .unwrap_err()
            .to_string()
            .contains("source group")
    );
}

#[test]
fn rollout_audit_rejects_changed_features_and_partition_provenance() {
    let directory = tempfile::tempdir().unwrap();
    let store = DatasetStore::open(directory.path()).unwrap();
    let entries = collect_program(&store, &fixture(), None, &RolloutOptions::default()).unwrap();
    let mut record: RolloutRecord = store.load(&entries[0]).unwrap();
    record.observation.nodes[0].features[0] += 1.0;
    let bad = store.put(entries[0].metadata.clone(), &record).unwrap();
    assert!(
        audit_record(&store, &bad)
            .unwrap_err()
            .to_string()
            .contains("observation")
    );
    let seed = (0..100)
        .find(|&s| group_split(&record.group_id, s) == Split::Train)
        .unwrap();
    let options = TrainOptions {
        epochs: 1,
        split_seed: seed,
        ..Default::default()
    };
    assert!(
        train_store(&store, &options, None)
            .unwrap_err()
            .to_string()
            .contains("observation")
    );
    let mut record: RolloutRecord = store.load(&entries[0]).unwrap();
    record.outcomes[0].partitions[0].roots.clear();
    let bad = store.put(entries[0].metadata.clone(), &record).unwrap();
    assert!(
        audit_record(&store, &bad)
            .unwrap_err()
            .to_string()
            .contains("partitions")
    );
}

#[test]
fn beam_search_reencodes_deduplicates_and_replans_beyond_its_horizon() {
    use ptu_plan::learning::beam::*;
    let mut program = fixture();
    let network = &mut program
        .problem
        .as_mut()
        .unwrap()
        .init_tn
        .as_mut()
        .unwrap()
        .tn;
    for i in 2..4 {
        let mut task = network.subtasks[i % 2].clone();
        task.id = None;
        network.subtasks.push(task);
    }
    let analysis = analyze_program(&program).unwrap();
    let graph = InteractionGraph::from_analysis(&analysis);
    let seeds = micropartition_network(
        &graph,
        NetworkId::Initial,
        &SpectralOptions {
            max_cluster_size: 1,
            ..Default::default()
        },
    )
    .unwrap()
    .clusters;
    let config = ModelConfig {
        hidden: 2,
        layers: 1,
    };
    let mut model = MergeModel::new(config, 42).unwrap();
    model.parameters.fill(0.0);
    let width = 4 * config.hidden + PAIR_FEATURES + GLOBAL_FEATURES + CONTEXT_FEATURES + 1;
    let policy = model.parameters.len() - 2 * (width + 1);
    model.parameters[policy + width - 1] = -10.0;
    let options = AgglomerationOptions::default();
    let (state, trace) = beam_agglomerate(
        &analysis,
        &graph,
        &seeds,
        &options,
        &model,
        0.1,
        &MasterOptions::default(),
        &BeamOptions {
            depth: 2,
            width: 16,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(state.clusters.len(), 1);
    assert_eq!(trace.decisions.len(), 3);
    assert!(trace.beam.as_ref().unwrap().deduplicated > 0);
    assert!(trace.beam.as_ref().unwrap().prediction_cache_hits > 0);
    assert_eq!(trace.beam.as_ref().unwrap().rounds, 3);
    let (expected, expected_trace) =
        learned_agglomerate(&analysis, &graph, &seeds, &options, &model, 0.1).unwrap();
    let (actual, trace) = beam_agglomerate(
        &analysis,
        &graph,
        &seeds,
        &options,
        &model,
        0.1,
        &MasterOptions::default(),
        &BeamOptions {
            depth: 1,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        actual
            .clusters
            .iter()
            .map(|c| (&c.id, &c.nodes))
            .collect::<Vec<_>>(),
        expected
            .clusters
            .iter()
            .map(|c| (&c.id, &c.nodes))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        trace.decisions.iter().map(|d| d.action).collect::<Vec<_>>(),
        expected_trace
            .decisions
            .iter()
            .map(|d| d.action)
            .collect::<Vec<_>>()
    );
    let (limited, trace) = beam_agglomerate(
        &analysis,
        &graph,
        &seeds,
        &options,
        &model,
        0.1,
        &MasterOptions::default(),
        &BeamOptions {
            max_evaluations: 1,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(trace.beam.as_ref().unwrap().evaluations, 1);
    assert!(trace.beam.as_ref().unwrap().truncated);
    assert_eq!(limited.clusters.len(), 3);
    agglomeration::observe(
        &analysis,
        &graph,
        NetworkId::Initial,
        &limited.clusters,
        &options,
    )
    .unwrap();
}
