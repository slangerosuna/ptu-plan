use ptu_plan::instrumentation::TraceLevel;
use ptu_plan::learning::interface::*;
use ptu_plan::learning::rollout::{RolloutOptions, collect_program};
use ptu_plan::master::*;
use ptu_plan::parsing::json::Program;
use ptu_plan::solver::*;
use ptu_plan::store::*;
use std::process::Command;

fn fixture() -> Program {
    serde_json::from_str(include_str!("fixtures/solver.json")).unwrap()
}
fn partitions() -> Vec<RootPartition> {
    (0..2)
        .map(|id| RootPartition {
            id,
            roots: vec![id],
        })
        .collect()
}

#[test]
fn interface_gradients_cover_all_classes_and_probabilities_are_normalized() {
    let input = InterfaceInput {
        feature_version: INTERFACE_FEATURE_VERSION,
        atoms: (0..3)
            .map(|i| AtomInput {
                atom: GroundAtom {
                    predicate: format!("p{i}"),
                    arguments: vec![],
                },
                features: (0..ATOM_FEATURES).map(|j| (i + j) as f64 / 20.0).collect(),
            })
            .collect(),
    };
    let targets = [
        FluentStatus::Unknown,
        FluentStatus::False,
        FluentStatus::True,
    ];
    let weights = [0.25, 1.0, 1.0];
    let mut model = InterfaceModel::new(2, 7).unwrap();
    let (loss, gradient) = model.loss_gradient(&input, &targets, &weights).unwrap();
    assert!((loss - model.loss(&input, &targets, &weights).unwrap()).abs() < 1e-14);
    for (i, &expected) in gradient.iter().enumerate() {
        let original = model.parameters[i];
        let epsilon = 1e-5;
        model.parameters[i] = original + epsilon;
        let a = model.loss_gradient(&input, &targets, &weights).unwrap().0;
        model.parameters[i] = original - epsilon;
        let b = model.loss_gradient(&input, &targets, &weights).unwrap().0;
        model.parameters[i] = original;
        assert!(
            ((a - b) / (2.0 * epsilon) - expected).abs() < 1e-7,
            "parameter {i}"
        );
    }
    for probabilities in model.predict(&input).unwrap() {
        assert!((probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(probabilities.iter().all(|p| *p > 0.0 && *p < 1.0));
    }
    let mut invalid = input.clone();
    invalid.atoms[0].features[0] = f64::NAN;
    assert!(model.predict(&invalid).is_err());
    assert!(
        model
            .loss_gradient(&input, &targets[..2], &weights)
            .is_err()
    );
    assert!(
        model
            .loss_gradient(&input, &targets, &[0.0, 1.0, 1.0])
            .is_err()
    );
}

struct UniformGuide(bool);
impl InterfaceGuide for UniformGuide {
    fn propose(&self, query: &InterfaceQuery<'_>) -> Result<BoundaryState, SolverError> {
        let atoms = query.boundary.iter().cloned().collect();
        Ok(if self.0 {
            BoundaryState {
                must_be_true: atoms,
                ..Default::default()
            }
        } else {
            BoundaryState {
                must_be_false: atoms,
                ..Default::default()
            }
        })
    }
}
#[test]
fn wrong_and_unknown_proposals_preserve_exhaustive_search_and_replay() {
    let model_predictor = InterfaceModel::new(3, 42).unwrap();
    for text in [
        include_str!("fixtures/solver.json"),
        include_str!("fixtures/interleaving.json"),
    ] {
        let program: Program = serde_json::from_str(text).unwrap();
        let model = ground_reachable_program(&program, &GroundingOptions::default()).unwrap();
        let baseline = solve_sequential(&model, &partitions(), &MasterOptions::default()).unwrap();
        for guide in [
            &UniformGuide(false) as &dyn InterfaceGuide,
            &UniformGuide(true),
            &model_predictor,
        ] {
            let mut events = Vec::new();
            let result = solve_sequential_guided(
                &model,
                &partitions(),
                &MasterOptions::default(),
                &MasterGuidance {
                    interface: Some(guide),
                    conflict_core_checks: 4,
                },
                TraceLevel::Full,
                Some(&mut |e| events.push(e)),
            )
            .unwrap();
            assert_eq!(result.status, baseline.status);
            if let Some(cert) = &result.certificate {
                validate_problem_certificate(&model, cert).unwrap();
            }
            assert!(result.statistics.interface_proposals > 0);
            assert!(events.iter().any(|e| matches!(
                e,
                ptu_plan::instrumentation::MasterEvent::InterfaceProposed { .. }
            )));
        }
    }
}

#[test]
fn interface_training_uses_verified_boundaries_and_resumes_deterministically() {
    let root = tempfile::tempdir().unwrap();
    let store = DatasetStore::open(root.path()).unwrap();
    let entries = collect_program(&store, &fixture(), None, &RolloutOptions::default()).unwrap();
    let group = entries[0].metadata.group_id.as_ref().unwrap();
    let split_seed = (0..100)
        .find(|&s| group_split(group, s) == Split::Train)
        .unwrap();
    let config = InterfaceTrainOptions {
        epochs: 100,
        hidden: 3,
        learning_rate: 0.01,
        split_seed,
        ..Default::default()
    };
    let (entry, _, report) = train_interface_store(&store, &config, None).unwrap();
    assert!(report.training.loss < report.initial_training.loss / 2.0);
    assert!(report.training.per_class_atoms[0] > 0);
    assert!(report.training.per_class_atoms[2] > 0);
    assert_eq!(report.validation.interfaces, 0);
    assert_eq!(report.test.interfaces, 0);
    let checkpoint: InterfaceCheckpoint = store.load(&entry).unwrap();
    checkpoint.validate().unwrap();
    let half = InterfaceTrainOptions {
        epochs: 50,
        ..config
    };
    let (entry, _, _) = train_interface_store(&store, &half, None).unwrap();
    let partial: InterfaceCheckpoint = store.load(&entry).unwrap();
    let bytes = serde_json::to_vec(&partial).unwrap();
    let partial = serde_json::from_slice(&bytes).unwrap();
    let (entry, _, _) = train_interface_store(&store, &half, Some(partial)).unwrap();
    let resumed: InterfaceCheckpoint = store.load(&entry).unwrap();
    for (a, b) in checkpoint
        .model
        .parameters
        .iter()
        .zip(&resumed.model.parameters)
    {
        assert!((a - b).abs() < 1e-10);
    }
    let mut bad = checkpoint.clone();
    bad.sources[0].split = Split::Validation;
    assert!(bad.validate().is_err());
}

#[test]
fn interface_training_and_guided_planner_clis_work_end_to_end() {
    let root = tempfile::tempdir().unwrap();
    let store_path = root.path().join("store");
    let store = DatasetStore::open(&store_path).unwrap();
    let entries = collect_program(
        &store,
        &fixture(),
        None,
        &RolloutOptions {
            conflict_core_checks: 4,
            trace: TraceLevel::Full,
            ..Default::default()
        },
    )
    .unwrap();
    let seed = (0..100)
        .find(|&s| group_split(entries[0].metadata.group_id.as_ref().unwrap(), s) == Split::Train)
        .unwrap();
    let checkpoint = root.path().join("interface.json");
    let output = Command::new(env!("CARGO_BIN_EXE_train-interface-model"))
        .arg("--store")
        .arg(&store_path)
        .arg("--output")
        .arg(&checkpoint)
        .args(["--epochs", "3", "--split-seed", &seed.to_string()])
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
            "--micro-max-size",
            "1",
            "--interface-model",
        ])
        .arg(&checkpoint)
        .args([
            "--conflict-core-checks",
            "4",
            "--analysis-json",
            "--master-trace",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let result: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(result["sequential"]["status"], "Feasible");
    assert!(
        result["sequential"]["statistics"]["interface_proposals"]
            .as_u64()
            .unwrap()
            > 0
    );
    let output = Command::new(env!("CARGO_BIN_EXE_dataset-store"))
        .arg("--store")
        .arg(&store_path)
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
fn partial_successes_have_local_evidence_and_tampered_witnesses_are_rejected() {
    let root = tempfile::tempdir().unwrap();
    let store = DatasetStore::open(root.path()).unwrap();
    let options = RolloutOptions {
        trace: TraceLevel::Full,
        master: MasterOptions {
            max_local_calls: 2,
            ..Default::default()
        },
        ..Default::default()
    };
    let entries = collect_program(&store, &fixture(), None, &options).unwrap();
    let record: ptu_plan::learning::rollout::RolloutRecord = store.load(&entries[0]).unwrap();
    let model = ground_reachable_program(&fixture(), &GroundingOptions::default()).unwrap();
    let examples = examples_from_rollout(&model, &record).unwrap();
    assert!(!examples.is_empty());
    assert!(
        examples
            .iter()
            .all(|e| e.scope == InterfaceEvidence::LocalWitness)
    );
    let group = entries[0].metadata.group_id.as_ref().unwrap();
    let split_seed = (0..100)
        .find(|&seed| group_split(group, seed) == Split::Train)
        .unwrap();
    let config = InterfaceTrainOptions {
        epochs: 2,
        split_seed,
        include_local: false,
        ..Default::default()
    };
    assert!(train_interface_store(&store, &config, None).is_err());
    let (_, _, report) = train_interface_store(
        &store,
        &InterfaceTrainOptions {
            include_local: true,
            ..config
        },
        None,
    )
    .unwrap();
    assert!(report.training.local_interfaces > 0);
    assert_eq!(report.training.global_interfaces, 0);
    let mut tampered = record.clone();
    for outcome in &mut tampered.outcomes {
        for event in &mut outcome.events {
            if let ptu_plan::instrumentation::MasterEvent::OracleFinished {
                certificate: Some(c),
                ..
            } = event
            {
                c.final_state.clear();
            }
        }
    }
    assert!(examples_from_rollout(&model, &tampered).is_err());
}

#[test]
fn boundary_only_inference_preserves_probabilities_and_row_gradient_means() {
    let model = ground_reachable_program(&fixture(), &GroundingOptions::default()).unwrap();
    let network = model.initial_networks[0].network.induced(&[0]).unwrap();
    let atom = GroundAtom {
        predicate: "done".into(),
        arguments: vec!["a".into()],
    };
    let boundary = vec![atom];
    let future_reads = boundary.iter().cloned().collect();
    let query = InterfaceQuery {
        problem: &model,
        network: &network,
        incoming: &model.initial_state,
        boundary: &boundary,
        future_reads: &future_reads,
    };
    let full = InterfaceInput::from_query(&query);
    let selected = InterfaceInput::for_boundary(&query);
    assert!(full.atoms.len() > selected.atoms.len());
    assert_eq!(selected.atoms.len(), 1);
    let classifier = InterfaceModel::new(3, 42).unwrap();
    let all = classifier.predict(&full).unwrap();
    let one = classifier.predict(&selected).unwrap();
    let index = full
        .atoms
        .iter()
        .position(|a| a.atom == boundary[0])
        .unwrap();
    assert_eq!(all[index], one[0]);
    assert_eq!(full.atoms[index].features, selected.atoms[0].features);
    let targets: Vec<_> = full
        .atoms
        .iter()
        .map(|a| {
            if a.atom == boundary[0] {
                FluentStatus::True
            } else {
                FluentStatus::Unknown
            }
        })
        .collect();
    let weights = [0.25, 1.0, 1.0];
    let (loss, gradient) = classifier.loss_gradient(&full, &targets, &weights).unwrap();
    let mut expected_loss = 0.0;
    let mut expected_gradient = vec![0.0; gradient.len()];
    for (atom, target) in full.atoms.iter().zip(&targets) {
        let input = InterfaceInput {
            feature_version: INTERFACE_FEATURE_VERSION,
            atoms: vec![atom.clone()],
        };
        let (l, g) = classifier
            .loss_gradient(&input, &[*target], &weights)
            .unwrap();
        expected_loss += l / targets.len() as f64;
        for (sum, g) in expected_gradient.iter_mut().zip(g) {
            *sum += g / targets.len() as f64;
        }
    }
    assert!((loss - expected_loss).abs() < 1e-12);
    for (a, b) in gradient.iter().zip(expected_gradient) {
        assert!((a - b).abs() < 1e-12);
    }
}
