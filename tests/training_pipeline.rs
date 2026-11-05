use ptu_plan::{
    instrumentation::TraceLevel,
    learning::{
        pipeline::{ExportOptions, export_data},
        rollout::{RolloutOptions, collect_program},
    },
    parsing::json::Program,
    store::{DatasetStore, Split, group_split},
    training::sha256,
};
use std::{collections::BTreeSet, fs};

#[test]
fn portable_export_audits_labels_holds_out_domains_and_preserves_packages() {
    let root = tempfile::tempdir().unwrap();
    let store = DatasetStore::open(&root.path().join("data")).unwrap();
    let options = RolloutOptions {
        trace: TraceLevel::Full,
        ..Default::default()
    };
    let mut seen = BTreeSet::new();
    for i in 0..200 {
        let mut p: Program = serde_json::from_str(include_str!("fixtures/solver.json")).unwrap();
        p.domain.name = format!("pipeline-{i}");
        p.problem.as_mut().unwrap().domain_name = p.domain.name.clone();
        let group = sha256(&serde_json::to_vec(&p.domain).unwrap());
        let split = group_split(&group, 42);
        let index = match split {
            Split::Train => 0,
            Split::Validation => 1,
            Split::Test => 2,
        };
        if !seen.insert(index) {
            continue;
        }
        collect_program(&store, &p, None, &options).unwrap();
        if seen.len() == 3 {
            break;
        }
    }
    assert_eq!(seen.len(), 3);
    let report = export_data(
        root.path(),
        &ExportOptions {
            wl: false,
            hidden: 2,
            layers: 1,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(
        report
            .splits
            .iter()
            .all(|s| s.merge_records > 0 && s.interfaces > 0 && s.transitions > 0)
    );
    let package = root.path().join("prepared").join(&report.package);
    let references: Vec<serde_json::Value> =
        serde_json::from_slice(&fs::read(package.join("references.json")).unwrap()).unwrap();
    for head in [
        "merge_loss_gradient",
        "interface_loss_gradient",
        "latent_loss_gradient",
    ] {
        assert!(references.iter().any(|r| r.get(head).is_some()));
    }
    let again = export_data(
        root.path(),
        &ExportOptions {
            wl: false,
            hidden: 2,
            layers: 1,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(again.package, report.package);
    assert_eq!(again.records_sha256, report.records_sha256);
    // Benchmarking an exported fixture must use held-out sources only, replay
    // every positive result, rotate repeated runs, and preserve resume identity.
    let models = root.path().join("models");
    fs::create_dir(&models).unwrap();
    for (template, name) in [
        ("merge-template.json", "merge-policy.json"),
        ("interface-template.json", "interface-policy.json"),
        ("latent-template.json", "latent-policy.json"),
    ] {
        fs::copy(package.join(template), models.join(name)).unwrap();
    }
    let benchmark = root.path().join("benchmark.json");
    let data = root.path().join("data");
    let args = [
        "--store",
        data.to_str().unwrap(),
        "--models",
        models.to_str().unwrap(),
        "--output",
        benchmark.to_str().unwrap(),
        "--repeats",
        "2",
    ];
    let run = std::process::Command::new(env!("CARGO_BIN_EXE_benchmark-models"))
        .args(args)
        .output()
        .unwrap();
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    let raw = fs::read(&benchmark).unwrap();
    let result: serde_json::Value = serde_json::from_slice(&raw).unwrap();
    assert_eq!(result["complete"], true);
    assert_eq!(result["source_problems"], 1);
    let cases = result["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 1);
    let group = cases[0]["group_id"].as_str().unwrap();
    assert_eq!(group_split(group, 42), Split::Test);
    let runs = cases[0]["runs"].as_array().unwrap();
    assert_eq!(runs.len(), 16);
    assert!(runs.iter().all(|r| r["status"] == "Feasible"
        && r["replay_verified"] == true
        && r["error"].is_null()));
    let resumed = std::process::Command::new(env!("CARGO_BIN_EXE_benchmark-models"))
        .args(args)
        .arg("--resume")
        .output()
        .unwrap();
    assert!(resumed.status.success());
    assert_eq!(fs::read(&benchmark).unwrap(), raw);
    let changed = std::process::Command::new(env!("CARGO_BIN_EXE_benchmark-models"))
        .args(args)
        .args(["--resume", "--value-weight", "0.2"])
        .output()
        .unwrap();
    assert!(!changed.status.success());
    assert_eq!(fs::read(&benchmark).unwrap(), raw);
    // A fixed cohort is checked against held-out source IDs even when a later
    // dataset contains additional problems. The input file is checksum-pinned.
    let cohort = root.path().join("cohort.json");
    fs::write(&cohort, &raw).unwrap();
    let cohort_output = root.path().join("cohort-result.json");
    let fixed = std::process::Command::new(env!("CARGO_BIN_EXE_benchmark-models"))
        .args([
            "--store",
            data.to_str().unwrap(),
            "--models",
            models.to_str().unwrap(),
            "--output",
            cohort_output.to_str().unwrap(),
            "--cohort",
            cohort.to_str().unwrap(),
            "--repeats",
            "1",
        ])
        .output()
        .unwrap();
    assert!(
        fixed.status.success(),
        "{}",
        String::from_utf8_lossy(&fixed.stderr)
    );
    let fixed: serde_json::Value =
        serde_json::from_slice(&fs::read(&cohort_output).unwrap()).unwrap();
    assert_eq!(fixed["cases"][0]["problem_id"], cases[0]["problem_id"]);
    assert!(fixed["cohort_sha256"].is_string());
    // Collector options change provenance IDs. Exact program fingerprints still
    // match the same held-out instance, without relying on names or root counts.
    let mut semantic: serde_json::Value = serde_json::from_slice(&raw).unwrap();
    let old_id = "f".repeat(64);
    semantic["cases"][0]["problem_id"] = serde_json::json!(old_id);
    fs::write(&cohort, serde_json::to_vec(&semantic).unwrap()).unwrap();
    let semantic_output = root.path().join("semantic-result.json");
    let matched = std::process::Command::new(env!("CARGO_BIN_EXE_benchmark-models"))
        .args([
            "--store",
            data.to_str().unwrap(),
            "--models",
            models.to_str().unwrap(),
            "--output",
            semantic_output.to_str().unwrap(),
            "--cohort",
            cohort.to_str().unwrap(),
            "--repeats",
            "1",
        ])
        .output()
        .unwrap();
    assert!(
        matched.status.success(),
        "{}",
        String::from_utf8_lossy(&matched.stderr)
    );
    let matched: serde_json::Value =
        serde_json::from_slice(&fs::read(semantic_output).unwrap()).unwrap();
    assert_eq!(matched["cases"][0]["problem_id"], cases[0]["problem_id"]);
    assert_eq!(matched["cases"][0]["cohort_problem_id"], old_id);
    let mut duplicate: serde_json::Value = serde_json::from_slice(&raw).unwrap();
    let case = duplicate["cases"][0].clone();
    duplicate["cases"].as_array_mut().unwrap().push(case);
    fs::write(&cohort, serde_json::to_vec(&duplicate).unwrap()).unwrap();
    let rejected = std::process::Command::new(env!("CARGO_BIN_EXE_benchmark-models"))
        .args([
            "--store",
            data.to_str().unwrap(),
            "--models",
            models.to_str().unwrap(),
            "--output",
            cohort_output.to_str().unwrap(),
            "--cohort",
            cohort.to_str().unwrap(),
            "--resume",
        ])
        .output()
        .unwrap();
    assert!(!rejected.status.success());
    // A cached immutable package is checked, never silently regenerated over corruption.
    fs::write(package.join("records.jsonl"), "corrupt").unwrap();
    assert!(
        export_data(
            root.path(),
            &ExportOptions {
                wl: false,
                hidden: 2,
                layers: 1,
                ..Default::default()
            }
        )
        .unwrap_err()
        .to_string()
        .contains("checksum")
    );
}

#[test]
fn wl_export_builds_instance_features_and_audit_rejects_changed_subtree_counts() {
    use ptu_plan::learning::{
        model::MergeModel,
        rollout::{RolloutAuditor, RolloutRecord},
    };
    let root = tempfile::tempdir().unwrap();
    let store = DatasetStore::open(&root.path().join("data")).unwrap();
    let p: Program = serde_json::from_str(include_str!("fixtures/solver.json")).unwrap();
    let entries = collect_program(
        &store,
        &p,
        None,
        &RolloutOptions {
            trace: TraceLevel::Full,
            ..Default::default()
        },
    )
    .unwrap();
    let entry = &entries[0];
    let mut record: RolloutRecord = store.load(entry).unwrap();
    assert!(record.observation.hilg.as_ref().unwrap().wl.is_some());
    let model = MergeModel::new_wl(
        ptu_plan::learning::model::ModelConfig {
            hidden: 4,
            layers: 2,
        },
        42,
    )
    .unwrap();
    model.predict(&record.observation).unwrap();
    let report = export_data(
        root.path(),
        &ExportOptions {
            hidden: 4,
            layers: 2,
            ..Default::default()
        },
    )
    .unwrap();
    let package = root.path().join("prepared").join(report.package);
    let checkpoint: ptu_plan::learning::training::Checkpoint =
        serde_json::from_slice(&fs::read(package.join("merge-template.json")).unwrap()).unwrap();
    assert_eq!(checkpoint.model.schema_version, 2);
    let latent: ptu_plan::learning::latent_training::LatentCheckpoint =
        serde_json::from_slice(&fs::read(package.join("latent-template.json")).unwrap()).unwrap();
    latent.validate().unwrap();
    record
        .observation
        .hilg
        .as_mut()
        .unwrap()
        .wl
        .as_mut()
        .unwrap()
        .clusters[0][0] += 1.0;
    let changed = store.put(entry.metadata.clone(), &record).unwrap();
    let snapshot = store.snapshot().unwrap();
    let mut auditor = RolloutAuditor::new(&store, &snapshot);
    assert!(
        auditor
            .audit(&changed)
            .unwrap_err()
            .to_string()
            .contains("symbolic partition state")
    );
}
