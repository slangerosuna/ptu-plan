use ptu_plan::{
    agglomeration::{AgglomerationOptions, observe},
    analysis::{NetworkId, analyze_program},
    interaction::InteractionGraph,
    learning::{
        beam::*, features::*, latent::*, latent_training::*, model::*, rollout::*, training::*,
    },
    master::{MasterOptions, MasterStatus, initial_partitions, solve_sequential},
    parsing::json::Program,
    partition::{SpectralOptions, micropartition_network},
    solver::*,
    store::*,
};
fn fixture() -> Program {
    serde_json::from_str(include_str!("fixtures/solver.json")).unwrap()
}
fn transition() -> (Observation, Observation) {
    let before = Observation {
        hilg: None,
        feature_version: FEATURE_VERSION,
        nodes: [4, 9, 12]
            .into_iter()
            .enumerate()
            .map(|(i, id)| ClusterInput {
                id,
                features: vec![0.2 + i as f64 * 0.1; NODE_FEATURES],
            })
            .collect(),
        relations: vec![Relation {
            source: 0,
            target: 2,
            channel: 2,
            weight: 1.0,
        }],
        global: vec![0.3; GLOBAL_FEATURES],
        context: vec![0.2; CONTEXT_FEATURES],
        actions: vec![
            ActionInput {
                pair: Some((0, 1)),
                features: vec![0.2; PAIR_FEATURES],
            },
            ActionInput {
                pair: None,
                features: vec![0.0; PAIR_FEATURES],
            },
        ],
    };
    let after = Observation {
        nodes: vec![
            before.nodes[2].clone(),
            ClusterInput {
                id: 13,
                features: vec![0.6; NODE_FEATURES],
            },
        ],
        relations: vec![],
        actions: vec![ActionInput {
            pair: None,
            features: vec![0.0; PAIR_FEATURES],
        }],
        ..before.clone()
    };
    (before, after)
}
#[test]
fn exact_encoding_decoder_matches_the_original_heads_and_rejects_bad_latents() {
    let model = MergeModel::new(
        ModelConfig {
            hidden: 3,
            layers: 2,
        },
        13,
    )
    .unwrap();
    let (before, _) = transition();
    let hidden = model.encode(&before).unwrap();
    let expected = model.predict(&before).unwrap();
    let actual = model.predict_encoded(&before, &hidden).unwrap();
    assert_eq!(expected.logits, actual.logits);
    assert_eq!(expected.log_effort, actual.log_effort);
    let mut bad = hidden.clone();
    bad[0][0] = f64::NAN;
    assert!(model.predict_encoded(&before, &bad).is_err());
    assert!(model.predict_encoded(&before, &hidden[..1]).is_err());
}
#[test]
fn all_transition_gradients_match_finite_differences_and_fit_frozen_targets() {
    let encoder = MergeModel::new(
        ModelConfig {
            hidden: 2,
            layers: 1,
        },
        7,
    )
    .unwrap();
    let mut model = LatentModel::new(&encoder, 17).unwrap();
    let (before, after) = transition();
    let hidden = encoder.encode(&before).unwrap();
    let example = TransitionExample {
        inputs: model.inputs(&before, &hidden, 0, &after).unwrap(),
        target: encoder.encode(&after).unwrap(),
    };
    let (loss, gradient) = model.loss_gradient(&example).unwrap();
    assert!((loss - model.loss(&example).unwrap()).abs() < 1e-14);
    for (i, &analytic) in gradient.iter().enumerate() {
        let old = model.parameters[i];
        let epsilon = 1e-6;
        model.parameters[i] = old + epsilon;
        let plus = model.loss(&example).unwrap();
        model.parameters[i] = old - epsilon;
        let minus = model.loss(&example).unwrap();
        model.parameters[i] = old;
        let numeric = (plus - minus) / (2.0 * epsilon);
        assert!(
            (numeric - analytic).abs() < 1e-6,
            "parameter {i}: {analytic} vs {numeric}"
        );
    }
    let original = encoder.parameters.clone();
    let mut optimizer = Adam::new(model.parameters.len());
    let options = TrainOptions {
        learning_rate: 0.01,
        ..Default::default()
    };
    for _ in 0..100 {
        let (_, gradient) = model.loss_gradient(&example).unwrap();
        optimizer
            .update_parameters(&mut model.parameters, &gradient, &options)
            .unwrap();
    }
    assert!(model.loss(&example).unwrap() < loss / 10.0);
    assert_eq!(encoder.parameters, original);
    let predicted = model.transition(&before, &hidden, 0, &after).unwrap();
    let mut permuted = after.clone();
    permuted.nodes.reverse();
    let mut reversed = model.transition(&before, &hidden, 0, &permuted).unwrap();
    reversed.reverse();
    assert_eq!(predicted, reversed);
    let mut bad = after.clone();
    bad.nodes[1].id = before.nodes[0].id;
    assert!(model.transition(&before, &hidden, 0, &bad).is_err());
    assert!(model.transition(&before, &hidden, 1, &after).is_err());
    let mut wrong_encoder = encoder.clone();
    wrong_encoder.parameters[0] += 0.1;
    assert!(model.validate(&wrong_encoder).is_err());
}
fn train_fixture() -> (tempfile::TempDir, DatasetStore, TrainOptions, MergeModel) {
    let root = tempfile::tempdir().unwrap();
    let store = DatasetStore::open(root.path()).unwrap();
    let entries = collect_program(&store, &fixture(), None, &RolloutOptions::default()).unwrap();
    let group = entries[0].metadata.group_id.as_ref().unwrap();
    let split_seed = (0..100)
        .find(|&s| group_split(group, s) == Split::Train)
        .unwrap();
    let model = MergeModel::new(
        ModelConfig {
            hidden: 2,
            layers: 1,
        },
        31,
    )
    .unwrap();
    let options = TrainOptions {
        model: model.config,
        epochs: 40,
        learning_rate: 0.01,
        split_seed,
        ..Default::default()
    };
    (root, store, options, model)
}
#[test]
fn audited_transition_training_resumes_deterministically_and_rejects_changed_scope() {
    let (_root, store, options, encoder) = train_fixture();
    let (entry, _, report) = train_latent_store(&store, &encoder, &options, None).unwrap();
    let checkpoint: LatentCheckpoint = store.load(&entry).unwrap();
    checkpoint.validate().unwrap();
    assert!(report.training.node_mse.unwrap() < report.initial_training.node_mse.unwrap() / 5.0);
    assert_eq!(report.validation.transitions, 0);
    assert_eq!(report.test.transitions, 0);
    let half = TrainOptions {
        epochs: 20,
        ..options.clone()
    };
    let (entry, _, _) = train_latent_store(&store, &encoder, &half, None).unwrap();
    let partial = store.load(&entry).unwrap();
    let (entry, _, _) = train_latent_store(&store, &encoder, &half, Some(partial)).unwrap();
    let resumed: LatentCheckpoint = store.load(&entry).unwrap();
    assert_eq!(checkpoint.model.parameters, resumed.model.parameters);
    let mut bad = checkpoint.clone();
    bad.groups[1] = bad.groups[0].clone();
    assert!(bad.validate().is_err());
    let mut bad = checkpoint.clone();
    bad.records[1] = bad.records[0].clone();
    assert!(bad.validate().is_err());
    let mut changed = encoder.clone();
    changed.parameters[0] += 0.01;
    assert!(train_latent_store(&store, &changed, &half, Some(checkpoint)).is_err());
}
#[test]
fn latent_beam_reencodes_committed_states_and_retains_exact_solver_replay() {
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
    let mut model = MergeModel::new(
        ModelConfig {
            hidden: 2,
            layers: 1,
        },
        42,
    )
    .unwrap();
    model.parameters.fill(0.0);
    let width = 4 * 2 + PAIR_FEATURES + GLOBAL_FEATURES + CONTEXT_FEATURES + 1;
    let policy = model.parameters.len() - 2 * (width + 1);
    model.parameters[policy + width - 1] = -10.0;
    let mut latent = LatentModel::new(&model, 42).unwrap();
    // Deliberately poor imagined embeddings must still produce legal moves.
    latent.parameters.fill(3.0);
    let options = AgglomerationOptions::default();
    let beam = BeamOptions {
        depth: 2,
        ..Default::default()
    };
    let (state, trace) = beam_agglomerate_with_latent(
        &analysis,
        &graph,
        &seeds,
        &options,
        &model,
        0.1,
        &MasterOptions::default(),
        &beam,
        Some(&latent),
    )
    .unwrap();
    assert_eq!(state.clusters.len(), 1);
    let stats = trace.beam.unwrap();
    assert_eq!(stats.exact_encodings, stats.rounds);
    assert!(stats.latent_transitions > 0);
    assert_eq!(stats.prediction_cache_hits, 0);
    assert_eq!(stats.deduplicated, 0);
    assert!(stats.rounds > beam.depth);
    assert!(stats.latent_model_sha256.is_some());
    // Duplicated consumable roots have no solution, so use the base fixture to
    // check a positive certificate through the same guided path.
    let base = fixture();
    let analysis = analyze_program(&base).unwrap();
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
    let (state, _) = beam_agglomerate_with_latent(
        &analysis,
        &graph,
        &seeds,
        &options,
        &model,
        0.1,
        &MasterOptions::default(),
        &beam,
        Some(&latent),
    )
    .unwrap();
    let problem = ground_reachable_program(&base, &GroundingOptions::default()).unwrap();
    let result = solve_sequential(
        &problem,
        &initial_partitions(&graph, &state.clusters).unwrap(),
        &MasterOptions::default(),
    )
    .unwrap();
    assert_eq!(result.status, MasterStatus::Feasible);
    validate_problem_certificate(&problem, result.certificate.as_ref().unwrap()).unwrap();
    let invalid = BeamOptions { depth: 5, ..beam };
    assert!(
        beam_agglomerate_with_latent(
            &analysis,
            &graph,
            &seeds,
            &options,
            &model,
            0.1,
            &MasterOptions::default(),
            &invalid,
            Some(&latent)
        )
        .is_err()
    );
    let capped = BeamOptions {
        max_evaluations: 1,
        ..beam
    };
    let (state, trace) = beam_agglomerate_with_latent(
        &analysis,
        &graph,
        &seeds,
        &options,
        &model,
        0.1,
        &MasterOptions::default(),
        &capped,
        Some(&latent),
    )
    .unwrap();
    assert_eq!(trace.beam.unwrap().evaluations, 1);
    observe(
        &analysis,
        &graph,
        NetworkId::Initial,
        &state.clusters,
        &options,
    )
    .unwrap();
}
#[test]
fn latent_training_and_planner_clis_preserve_checkpoint_identity() {
    let (root, store, options, encoder) = train_fixture();
    let encoder_path = root.path().join("encoder.json");
    let latent_path = root.path().join("latent.json");
    std::fs::write(&encoder_path, serde_json::to_vec(&encoder).unwrap()).unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_train-latent-model"))
        .arg("--store")
        .arg(root.path())
        .arg("--merge-model")
        .arg(&encoder_path)
        .arg("--output")
        .arg(&latent_path)
        .args([
            "--epochs",
            "3",
            "--split-seed",
            &options.split_seed.to_string(),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let checkpoint: LatentCheckpoint =
        serde_json::from_slice(&std::fs::read(&latent_path).unwrap()).unwrap();
    checkpoint.validate().unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_ptu-plan"))
        .args([
            "tests/fixtures/solver.json",
            "--sequential",
            "--agglomerate",
            "--micro-max-size",
            "1",
            "--merge-beam",
            "--beam-depth",
            "2",
            "--analysis-json",
        ])
        .arg("--merge-model")
        .arg(&encoder_path)
        .arg("--latent-model")
        .arg(&latent_path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert!(report.to_string().contains("latent_model_sha256"));
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_dataset-store"))
        .arg("--store")
        .arg(root.path())
        .arg("audit")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        store
            .snapshot()
            .unwrap()
            .entries
            .iter()
            .any(|e| e.metadata.kind == RecordKind::LatentModel)
    );
}
