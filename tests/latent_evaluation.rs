use ptu_plan::{
    learning::{
        latent::LatentModel,
        latent_evaluation::*,
        latent_training::LatentCheckpoint,
        model::{MergeModel, ModelConfig},
        rollout::{RolloutOptions, RolloutRecord, collect_program},
        training::{Adam, TrainOptions},
    },
    parsing::json::Program,
    store::*,
};
fn fixture() -> (tempfile::TempDir, DatasetStore, LatentCheckpoint) {
    let directory = tempfile::tempdir().unwrap();
    let store = DatasetStore::open(directory.path()).unwrap();
    let mut program: Program = serde_json::from_str(include_str!("fixtures/solver.json")).unwrap();
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
    let mut options = RolloutOptions {
        max_levels: 1,
        max_candidates: 2,
        ..Default::default()
    };
    options.master.max_local_calls = 8;
    collect_program(&store, &program, None, &options).unwrap();
    let mut encoder = MergeModel::new(
        ModelConfig {
            hidden: 2,
            layers: 1,
        },
        31,
    )
    .unwrap();
    encoder.parameters.fill(0.0);
    let mut model = LatentModel::new(&encoder, 32).unwrap();
    model.parameters.fill(0.0);
    let checkpoint = LatentCheckpoint {
        schema_version: 1,
        optimizer: Adam::new(model.parameters.len()),
        options: TrainOptions {
            model: encoder.config,
            ..Default::default()
        },
        encoder,
        model,
        completed_epochs: 0,
        snapshot_id: store.snapshot().unwrap().fingerprint().unwrap(),
        records: Default::default(),
        groups: Default::default(),
    };
    checkpoint.validate().unwrap();
    (directory, store, checkpoint)
}
#[test]
fn open_loop_evaluation_compares_depths_baselines_and_missing_samples_without_mutation() {
    let (_directory, store, checkpoint) = fixture();
    let snapshot = store.snapshot().unwrap().fingerprint().unwrap();
    let serialized = serde_json::to_vec(&checkpoint).unwrap();
    let options = DriftOptions {
        depth: 4,
        paths_per_record: 1,
        max_paths: 1,
        ..Default::default()
    };
    let report = evaluate_latent_drift(&store, &checkpoint, &options).unwrap();
    assert_eq!(report.total_paths, 1);
    assert_eq!(report.audited_records, 1);
    let split = report.splits.iter().find(|s| s.paths > 0).unwrap();
    assert_eq!(split.records.len(), 1);
    assert_eq!(split.groups.len(), 1);
    for depth in &split.depths[..3] {
        assert_eq!(depth.samples, 1);
        let learned = depth.learned.as_ref().unwrap();
        let identity = depth.identity.as_ref().unwrap();
        assert_eq!(learned.node_mse, 0.0);
        assert_eq!(identity.node_mse, 0.0);
        assert_eq!(learned.value_mae, 0.0);
        assert_eq!(identity.value_mae, 0.0);
        if depth.depth < 3 {
            assert_eq!(learned.policy_decisions, 1);
            assert_eq!(learned.policy_top1_agreement, Some(1.0));
            assert_eq!(identity.policy_top1_agreement, Some(1.0));
        } else {
            assert_eq!(learned.policy_decisions, 0);
            assert_eq!(learned.policy_top1_agreement, None);
            assert_eq!(identity.policy_top1_agreement, None);
        }
    }
    assert_eq!(split.depths[3].samples, 0);
    assert!(split.depths[3].learned.is_none());
    assert!(split.depths[3].identity.is_none());
    for other in report.splits.iter().filter(|s| s.paths == 0) {
        assert!(
            other
                .depths
                .iter()
                .all(|d| d.samples == 0 && d.learned.is_none() && d.identity.is_none())
        );
    }
    assert_eq!(store.snapshot().unwrap().fingerprint().unwrap(), snapshot);
    assert_eq!(serde_json::to_vec(&checkpoint).unwrap(), serialized);
}
#[test]
fn evaluation_rejects_unverified_sources_and_enforces_global_path_limits() {
    let (_directory, store, checkpoint) = fixture();
    let options = DriftOptions {
        max_paths: 1,
        paths_per_record: 8,
        ..Default::default()
    };
    let report = evaluate_latent_drift(&store, &checkpoint, &options).unwrap();
    assert_eq!(report.total_paths, 1);
    assert!(report.truncated);
    let invalid = DriftOptions {
        depth: 5,
        ..options
    };
    assert!(evaluate_latent_drift(&store, &checkpoint, &invalid).is_err());
    let invalid = DriftOptions {
        max_paths: 0,
        ..options
    };
    assert!(evaluate_latent_drift(&store, &checkpoint, &invalid).is_err());
    let entry = store
        .snapshot()
        .unwrap()
        .entries
        .into_iter()
        .find(|e| e.metadata.kind == RecordKind::Rollout)
        .unwrap();
    let mut bad: RolloutRecord = store.load(&entry).unwrap();
    bad.observation.nodes[0].features[0] += 1.0;
    store.put(entry.metadata, &bad).unwrap();
    let error = evaluate_latent_drift(&store, &checkpoint, &DriftOptions::default()).unwrap_err();
    assert!(error.to_string().contains("observation"));
}
#[test]
fn drift_cli_saves_a_reproducible_report_without_publishing_dataset_records() {
    let (directory, store, checkpoint) = fixture();
    let model = directory.path().join("latent.json");
    checkpoint.save(&model).unwrap();
    let snapshot = store.snapshot().unwrap().fingerprint().unwrap();
    let report = directory.path().join("reports/drift.json");
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_evaluate-latent"))
        .arg("--store")
        .arg(directory.path())
        .arg("--latent-model")
        .arg(&model)
        .arg("--output")
        .arg(&report)
        .args(["--depth", "3", "--max-paths", "1"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let saved: DriftReport = serde_json::from_slice(&std::fs::read(report).unwrap()).unwrap();
    assert_eq!(saved.total_paths, 1);
    assert_eq!(saved.snapshot_id, snapshot);
    let repeated = evaluate_latent_drift(
        &store,
        &checkpoint,
        &DriftOptions {
            depth: 3,
            max_paths: 1,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        serde_json::to_value(&saved).unwrap(),
        serde_json::to_value(&repeated).unwrap()
    );
    assert_eq!(store.snapshot().unwrap().fingerprint().unwrap(), snapshot);
    assert_eq!(
        saved.checkpoint_sha256,
        ptu_plan::training::sha256(&serde_json::to_vec(&checkpoint).unwrap())
    );
}
