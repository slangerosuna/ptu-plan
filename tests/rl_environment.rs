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
use std::{
    collections::BTreeSet,
    fs,
    io::{BufRead, BufReader, Write},
    path::PathBuf,
    process::{Command, Stdio},
};
fn fixture() -> (tempfile::TempDir, PathBuf) {
    let root = tempfile::tempdir().unwrap();
    let store = DatasetStore::open(&root.path().join("data")).unwrap();
    let mut seen = BTreeSet::new();
    for i in 0..200 {
        let mut p: Program = serde_json::from_str(include_str!("fixtures/solver.json")).unwrap();
        p.domain.name = format!("rl-env-{i}");
        p.problem.as_mut().unwrap().domain_name = p.domain.name.clone();
        let split = group_split(&sha256(&serde_json::to_vec(&p.domain).unwrap()), 42);
        let index = match split {
            Split::Train => 0,
            Split::Validation => 1,
            Split::Test => 2,
        };
        if seen.insert(index) {
            collect_program(
                &store,
                &p,
                None,
                &RolloutOptions {
                    trace: TraceLevel::Full,
                    ..Default::default()
                },
            )
            .unwrap();
        }
        if seen.len() == 3 {
            break;
        }
    }
    assert_eq!(seen.len(), 3);
    let report = export_data(
        root.path(),
        &ExportOptions {
            hidden: 4,
            layers: 1,
            ..Default::default()
        },
    )
    .unwrap();
    let parent = root
        .path()
        .join("prepared")
        .join(report.package)
        .join("merge-template.json");
    (root, parent)
}
#[test]
fn planner_rl_environment_samples_legal_train_trajectories_and_rejects_validation_sampling() {
    let (root, parent) = fixture();
    let checkpoint: serde_json::Value =
        serde_json::from_slice(&fs::read(&parent).unwrap()).unwrap();
    for (split, sample, success) in [
        ("Train", true, true),
        ("Validation", true, false),
        ("Test", false, false),
    ] {
        let mut child = Command::new(env!("CARGO_BIN_EXE_train"))
            .args([
                "rl-environment",
                "--store",
                root.path().join("data").to_str().unwrap(),
                "--checkpoint",
                parent.to_str().unwrap(),
                "--train-problems",
                "2",
                "--validation-problems",
                "2",
                "--total-ms",
                "1000",
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let mut stdout = BufReader::new(child.stdout.take().unwrap());
        let mut line = String::new();
        stdout.read_line(&mut line).unwrap();
        let ready: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert_eq!(ready["train"].as_array().unwrap().len(), 1);
        assert_eq!(ready["validation"].as_array().unwrap().len(), 1);
        let request = serde_json::json!({"model":checkpoint["model"],"split":split,"sample":sample,"indices":[0],"seed":123});
        let mut stdin = child.stdin.take().unwrap();
        writeln!(stdin, "{}", request).unwrap();
        drop(stdin);
        line.clear();
        stdout.read_line(&mut line).unwrap();
        if success {
            let episodes: Vec<serde_json::Value> = serde_json::from_str(&line).unwrap();
            assert_eq!(episodes.len(), 1);
            let e = &episodes[0];
            assert_eq!(e["split"], "Train");
            assert_eq!(e["solved"], true);
            assert_eq!(e["replay_verified"], true);
            assert!(e["certificate"].is_object());
            assert!(e["reward"].as_f64().unwrap().is_finite());
            for (decision, probs) in e["decisions"]
                .as_array()
                .unwrap()
                .iter()
                .zip(e["probabilities"].as_array().unwrap())
            {
                let probabilities: Vec<f64> = serde_json::from_value(probs.clone()).unwrap();
                assert!((probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-12);
                assert!(decision["action"].as_u64().unwrap() < (probabilities.len() as u64));
            }
        } else {
            assert!(line.is_empty());
        }
        let output = child.wait_with_output().unwrap();
        assert_eq!(
            output.status.success(),
            success,
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
fn committed_online_problems_require_train_domains_matching_hashes_and_valid_certificates() {
    let (root, parent) = fixture();
    let checkpoint: serde_json::Value =
        serde_json::from_slice(&fs::read(&parent).unwrap()).unwrap();
    let store = DatasetStore::open(&root.path().join("data")).unwrap();
    let snapshot = store.snapshot().unwrap();
    let entry = snapshot
        .entries
        .iter()
        .find(|e| {
            e.metadata.kind == ptu_plan::store::RecordKind::Problem
                && checkpoint["training_groups"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|g| g.as_str() == e.metadata.group_id.as_deref())
        })
        .unwrap();
    let output_dir = root.path().join("online");
    let mut generation = Command::new(env!("CARGO_BIN_EXE_train"));
    let result = generation
        .args([
            "generate-online",
            "--store",
            root.path().join("data").to_str().unwrap(),
            "--checkpoint",
            parent.to_str().unwrap(),
            "--output",
            output_dir.to_str().unwrap(),
            "--template-id",
            &entry.id,
            "--count",
            "2",
            "--max-roots",
            "2",
            "--seed",
            "11",
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let ready = fs::read_dir(output_dir.join("ready"))
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let marker = fs::read(&ready).unwrap();
    let batch: serde_json::Value = serde_json::from_slice(&marker).unwrap();
    let temporary = output_dir
        .join("jobs")
        .join(batch["job"].as_str().unwrap())
        .join("solutions/.interrupted.tmp");
    fs::write(&temporary, b"partial sample").unwrap();
    fs::remove_file(&ready).unwrap();
    let recovered = generation.output().unwrap();
    assert!(
        recovered.status.success(),
        "{}",
        String::from_utf8_lossy(&recovered.stderr)
    );
    assert!(!temporary.exists());
    assert_eq!(fs::read(&ready).unwrap(), marker);
    let reference = &batch["samples"][0];
    assert!(reference["id"].is_string());
    let raw = fs::read_to_string(output_dir.join(reference["file"].as_str().unwrap())).unwrap();
    let valid = serde_json::json!({"id":reference["id"],"json":raw});
    let mut bad_certificate: serde_json::Value = serde_json::from_str(&raw).unwrap();
    bad_certificate["certificate"]["initial_network"] = serde_json::json!(999999);
    let bad_certificate = serde_json::to_string(&bad_certificate).unwrap();
    let mut foreign: serde_json::Value = serde_json::from_str(&raw).unwrap();
    foreign["program"]["domain"]["name"] = serde_json::json!("held-out-forgery");
    let foreign = serde_json::to_string(&foreign).unwrap();
    for (split, input, success) in [
        ("Train", valid.clone(), true),
        ("Validation", valid.clone(), false),
        (
            "Train",
            serde_json::json!({"id":"0".repeat(64),"json":raw}),
            false,
        ),
        (
            "Train",
            serde_json::json!({"id":sha256(bad_certificate.as_bytes()),"json":bad_certificate}),
            false,
        ),
        (
            "Train",
            serde_json::json!({"id":sha256(foreign.as_bytes()),"json":foreign}),
            false,
        ),
    ] {
        let mut child = Command::new(env!("CARGO_BIN_EXE_train"))
            .args([
                "rl-environment",
                "--store",
                root.path().join("data").to_str().unwrap(),
                "--checkpoint",
                parent.to_str().unwrap(),
                "--train-problems",
                "2",
                "--validation-problems",
                "2",
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let mut stdout = BufReader::new(child.stdout.take().unwrap());
        let mut line = String::new();
        stdout.read_line(&mut line).unwrap();
        let request = serde_json::json!({"model":checkpoint["model"],"split":split,"sample":false,
            "indices":[0],"online":[input],"seed":12});
        let mut stdin = child.stdin.take().unwrap();
        writeln!(stdin, "{request}").unwrap();
        drop(stdin);
        line.clear();
        stdout.read_line(&mut line).unwrap();
        if success {
            let episodes: serde_json::Value = serde_json::from_str(&line).unwrap();
            assert_eq!(episodes[0]["online"], true);
            assert_eq!(episodes[0]["problem_id"], reference["id"]);
            assert_eq!(episodes[0]["replay_verified"], true);
        } else {
            assert!(line.is_empty());
        }
        let result = child.wait_with_output().unwrap();
        assert_eq!(
            result.status.success(),
            success,
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
    }
    assert_eq!(fs::read(ready).unwrap(), marker);
}
