use std::collections::BTreeSet;
use std::fs;
use std::process::Command;

use ptu_plan::parsing::json::Program;
use ptu_plan::solver::*;
use ptu_plan::training::{
    DatasetWriter,
    backward::{BackwardOptions, generate, literal_requirements, regress},
    panda::primitive_plan,
    sample_fingerprint,
};

#[test]
fn incremental_generation_retains_partial_samples_and_reachable_certificates_audit() {
    use ptu_plan::training::backward::generate_to;
    let directory = tempfile::tempdir().unwrap();
    let writer = DatasetWriter::new(directory.path()).unwrap();
    let mut p = fixture();
    p.domain.compound_tasks.push(ptu_plan::parsing::json::Task {
        name: "unused".into(),
        parameters: (0..8)
            .map(|i| ptu_plan::parsing::json::Symbol {
                name: format!("?v{i}"),
                symbol_type: Some("item".into()),
            })
            .collect(),
    });
    let grounding = GroundingOptions { max_instances: 150 };
    let report = generate_to(
        &p,
        &grounding,
        &BackwardOptions {
            count: 100,
            seed: 73,
            max_attempts: 10,
            ..BackwardOptions::default()
        },
        |sample| {
            assert_eq!(sample.grounding_mode, GroundingMode::InitialReachable);
            writer.write(sample)?;
            Ok(())
        },
    )
    .unwrap();
    assert!(!report.samples.is_empty());
    assert!(report.samples.len() < 100);
    assert_eq!(report.attempts, 10);
    // No manifest yet: auditing recovers these committed samples.
    assert!(!directory.path().join("manifest.json").exists());
    assert_eq!(writer.audit(&grounding).unwrap(), report.samples.len());
    assert!(directory.path().join("manifest.json").exists());
}

#[test]
fn audit_checks_cached_micropartitions_without_committing_bad_manifest() {
    let directory = tempfile::tempdir().unwrap();
    let writer = DatasetWriter::new(directory.path()).unwrap();
    let sample = samples().remove(0);
    writer.write(&sample).unwrap();
    let path = directory.path().join(format!("{}.json", sample.id));
    let mut value: serde_json::Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    value["micropartitions"]["clusters"] = serde_json::json!([]);
    fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
    let error = writer.audit(&GroundingOptions::default()).unwrap_err();
    assert!(error.to_string().contains("micropartitions"), "{error}");
    assert!(!directory.path().join("manifest.json").exists());
}

#[test]
fn spectral_cache_metrics_allow_round_trip_noise_but_not_changed_weights() {
    let directory = tempfile::tempdir().unwrap();
    let writer = DatasetWriter::new(directory.path()).unwrap();
    let sample = samples().remove(0);
    writer.write(&sample).unwrap();
    let path = directory.path().join(format!("{}.json", sample.id));
    let mut value: serde_json::Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    let weight = value["micropartitions"]["cut_weight"].as_f64().unwrap();
    value["micropartitions"]["cut_weight"] = (weight + 1e-12).into();
    fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
    assert_eq!(writer.audit(&GroundingOptions::default()).unwrap(), 1);
    fs::remove_file(directory.path().join("manifest.json")).unwrap();
    value["micropartitions"]["cut_weight"] = (weight + 1.0).into();
    fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
    assert!(
        writer
            .audit(&GroundingOptions::default())
            .unwrap_err()
            .to_string()
            .contains("micropartitions")
    );
}

fn fixture() -> Program {
    serde_json::from_str(include_str!("fixtures/solver.json")).unwrap()
}
fn atom(name: &str) -> GroundAtom {
    GroundAtom {
        predicate: name.into(),
        arguments: Vec::new(),
    }
}
fn set(names: &[&str]) -> State {
    names.iter().map(|name| atom(name)).collect()
}
fn samples() -> Vec<ptu_plan::training::TrainingSample> {
    generate(
        &fixture(),
        &GroundingOptions::default(),
        &BackwardOptions {
            count: 4,
            seed: 73,
            max_attempts: 1000,
            ..BackwardOptions::default()
        },
    )
    .unwrap()
}

#[test]
fn backward_root_floor_produces_merge_training_problems() {
    let samples = generate(
        &fixture(),
        &GroundingOptions::default(),
        &BackwardOptions {
            count: 6,
            min_roots: 2,
            max_roots: 2,
            max_attempts: 1000,
            ..Default::default()
        },
    )
    .unwrap();
    for sample in samples {
        assert_eq!(
            sample
                .program
                .problem
                .unwrap()
                .init_tn
                .unwrap()
                .tn
                .subtasks
                .len(),
            2
        );
    }
    for min_roots in [0, 4] {
        assert!(
            generate(
                &fixture(),
                &GroundingOptions::default(),
                &BackwardOptions {
                    min_roots,
                    ..Default::default()
                }
            )
            .is_err()
        );
    }
}

#[test]
fn regression_respects_both_polarities_preconditions_and_add_wins() {
    let action = GroundAction {
        task: GroundTask {
            name: "step".into(),
            arguments: Vec::new(),
        },
        precondition: Condition::And(vec![
            Condition::Atom(atom("input")),
            Condition::Not(Box::new(Condition::Atom(atom("locked")))),
        ]),
        add: set(&["done", "both"]),
        delete: set(&["old", "both"]),
    };
    let required = BoundaryState {
        must_be_true: set(&["done", "both", "preserve"]),
        must_be_false: set(&["old", "absent"]),
    };
    let before = regress(&required, &action).unwrap();
    assert_eq!(before.must_be_true, set(&["input", "preserve"]));
    assert_eq!(before.must_be_false, set(&["locked", "absent"]));
    assert!(
        regress(
            &BoundaryState {
                must_be_true: set(&["old"]),
                ..BoundaryState::default()
            },
            &action
        )
        .is_err()
    );
    assert!(
        regress(
            &BoundaryState {
                must_be_false: set(&["done"]),
                ..BoundaryState::default()
            },
            &action
        )
        .is_err()
    );
    assert!(
        regress(
            &BoundaryState {
                must_be_false: set(&["input"]),
                ..BoundaryState::default()
            },
            &action
        )
        .is_err()
    );
    assert!(literal_requirements(&Condition::Or(vec![Condition::True, Condition::False])).is_err());
    assert!(literal_requirements(&Condition::False).is_err());
}

#[test]
fn sampled_solutions_are_unique_reproducible_and_replay_validated() {
    let first = samples();
    let second = samples();
    assert_eq!(
        first.iter().map(|s| &s.id).collect::<Vec<_>>(),
        second.iter().map(|s| &s.id).collect::<Vec<_>>()
    );
    assert_eq!(
        first.iter().map(|s| &s.id).collect::<BTreeSet<_>>().len(),
        4
    );
    for sample in first {
        let model = ground_program(&sample.program, &GroundingOptions::default()).unwrap();
        validate_problem_certificate(&model, &sample.certificate).unwrap();
        assert!(!sample.optimal);
        assert!(!sample.certificate.actions.is_empty());
        assert_eq!(sample.boundaries.len(), sample.action_cost + 1);
        assert!(
            sample
                .boundaries
                .iter()
                .all(|label| !label.exact_remaining_cost)
        );
        assert_eq!(sample.boundaries.last().unwrap().remaining_actions, 0);
        let mut renamed = sample.program.clone();
        renamed.problem.as_mut().unwrap().name = "cosmetic rename".into();
        assert_eq!(
            sample_fingerprint(&renamed, &sample.certificate.actions).unwrap(),
            sample_fingerprint(&sample.program, &sample.certificate.actions).unwrap()
        );
    }
}

#[test]
fn dataset_publication_is_additive_and_audit_detects_corruption() {
    let directory = tempfile::tempdir().unwrap();
    let writer = DatasetWriter::new(directory.path()).unwrap();
    let samples = samples();
    for sample in &samples {
        assert!(writer.write(sample).unwrap());
    }
    let path = directory.path().join(format!("{}.json", samples[0].id));
    let original = fs::read(&path).unwrap();
    assert!(!writer.write(&samples[0]).unwrap());
    assert_eq!(fs::read(&path).unwrap(), original);
    assert_eq!(writer.audit(&GroundingOptions::default()).unwrap(), 4);
    let manifest = writer.manifest().unwrap();
    assert!(
        manifest
            .iter()
            .all(|entry| !entry.optimal && entry.source == "backward")
    );
    let mut value: serde_json::Value = serde_json::from_slice(&original).unwrap();
    value["optimal"] = true.into();
    fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
    assert!(writer.audit(&GroundingOptions::default()).is_err());
    // Rebuilding metadata cannot conceal incorrect semantic labels either.
    writer.manifest().unwrap();
    assert!(writer.audit(&GroundingOptions::default()).is_err());
    fs::write(&path, &original).unwrap();
    writer.manifest().unwrap();
    value = serde_json::from_slice(&original).unwrap();
    value["boundaries"][0]["remaining_actions"] = 999.into();
    fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
    writer.manifest().unwrap();
    assert!(writer.audit(&GroundingOptions::default()).is_err());
}

#[test]
fn converted_panda_plans_preserve_occurrences_and_reject_raw_or_partial_output() {
    let plan = primitive_plan("==>\n3 move a b\n9 noop\nroot 0\n0 travel a b -> go 3 9\n").unwrap();
    assert_eq!(plan.len(), 2);
    assert_eq!(plan[0].name, "move");
    assert_eq!(plan[0].arguments, vec!["a", "b"]);
    for invalid in [
        "3 move a b\nroot 0\n",
        "==>\n3 move[a,b]\nroot 0\n",
        "==>\n3 __compiled\nroot 0\n",
        "==>\n3 noop\n3 noop\nroot 0\n",
        "==>\n3 noop\n",
    ] {
        assert!(primitive_plan(invalid).is_err(), "{invalid}");
    }
    assert!(primitive_plan("==>\nroot\n").unwrap().is_empty());
}

#[test]
fn panda_optimal_labels_require_completed_cost_aware_search_and_verification() {
    use ptu_plan::training::{panda::PandaEvidence, sha256};
    let domain = "(define (domain unit))";
    let problem = "(define (problem unit))";
    let evidence = PandaEvidence {
        objective: "unit_cost_primitive_actions".into(),
        engine_arguments: ["-p","-H","zero","-g","action","-w","1","-S","42","-t","15","input.sas"].map(str::to_string).into(),
        tool_sha256: ["parser","grounder","engine"].into_iter().map(|name| (name.into(), "a".repeat(64))).collect(),
        tool_revisions: Default::default(), domain_sha256: sha256(domain.as_bytes()), problem_sha256: sha256(problem.as_bytes()),
        domain_hddl: domain.into(), problem_hddl: problem.into(), reported_cost: 1,
        search_log: "Heuristic #0 = zero()\n - type: cost optimal\n - weight: 1\n - suboptimal: false\n- Status: Solved\n- Total costs of actions: 1\n".into(),
        converted_plan: "==>\n3 move a b\nroot 3\n".into(), verification_log: "Plan verification result: true\n".into(),
    };
    evidence.validate(1).unwrap();
    let mut bad = evidence.clone();
    bad.engine_arguments[4] = "path".into();
    assert!(bad.validate(1).is_err());
    let mut bad = evidence.clone();
    bad.search_log.push_str("- Status: Timeout\n");
    assert!(bad.validate(1).is_err());
    let mut bad = evidence.clone();
    bad.search_log = bad
        .search_log
        .replace("suboptimal: false", "suboptimal: true");
    assert!(bad.validate(1).is_err());
    let mut bad = evidence.clone();
    bad.verification_log = "Plan verification result: false\n".into();
    assert!(bad.validate(1).is_err());
    let mut bad = evidence.clone();
    bad.domain_hddl
        .push_str("\n(:metric minimize (total-cost))");
    bad.domain_sha256 = sha256(bad.domain_hddl.as_bytes());
    assert!(bad.validate(1).is_err());
    assert!(evidence.validate(2).is_err());
}

#[test]
fn external_action_sequences_require_an_exact_htn_witness() {
    let model = ground_program(&fixture(), &GroundingOptions::default()).unwrap();
    let original = solve_problem(&model, &SearchBudget::default())
        .unwrap()
        .certificate
        .unwrap();
    let recovered = solve_with_plan(&model, &original.actions, &SearchBudget::default())
        .unwrap()
        .certificate
        .unwrap();
    assert_eq!(recovered.actions, original.actions);
    validate_problem_certificate(&model, &recovered).unwrap();
    let wrong = vec![original.actions[0].clone(), original.actions[0].clone()];
    assert_eq!(
        solve_with_plan(&model, &wrong, &SearchBudget::default())
            .unwrap()
            .status,
        ProofStatus::Infeasible
    );
    assert!(
        solve_with_plan(&model, &[], &SearchBudget::default())
            .unwrap()
            .certificate
            .is_none()
    );
}

#[test]
fn ipc_style_implicit_supertypes_and_type_case_variants_ground_correctly() {
    let mut program = fixture();
    let types = program.domain.types.as_mut().unwrap();
    types.retain(|ty| ty.name != "item");
    types
        .iter_mut()
        .find(|ty| ty.name == "special")
        .unwrap()
        .symbol_type = Some("ITEM".into());
    program.problem.as_mut().unwrap().objects[0].symbol_type = Some("Special".into());
    program.domain.actions[0].parameters[0].symbol_type = Some("Item".into());
    let model = ground_program(&program, &GroundingOptions::default()).unwrap();
    assert_eq!(model.actions.len(), 2);
    assert_eq!(
        solve_problem(&model, &SearchBudget::default())
            .unwrap()
            .status,
        ProofStatus::Feasible
    );
}

#[test]
fn json_only_collection_and_audit_need_no_external_parser_or_ipc_checkout() {
    let directory = tempfile::tempdir().unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_generate-training"))
        .args([
            "--template",
            concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/solver.json"),
            "--synthetic-count",
            "4",
            "--real-limit",
            "0",
            "--seed",
            "73",
            "--ipc-dir",
            "/no/ipc",
            "--parser",
            "/no/parser",
            "--output",
        ])
        .arg(directory.path())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let manifest: Vec<serde_json::Value> =
        serde_json::from_slice(&fs::read(directory.path().join("manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest.len(), 4);
    let output = Command::new(env!("CARGO_BIN_EXE_generate-training"))
        .args(["--audit-only", "--parser", "/no/parser", "--output"])
        .arg(directory.path())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn generator_reports_impossible_quotas_and_limits_without_fake_records() {
    for options in [
        BackwardOptions {
            count: 1,
            max_attempts: 0,
            ..BackwardOptions::default()
        },
        BackwardOptions {
            count: 1,
            max_depth: 0,
            max_attempts: 8,
            ..BackwardOptions::default()
        },
        BackwardOptions {
            max_roots: 0,
            ..BackwardOptions::default()
        },
    ] {
        assert!(generate(&fixture(), &GroundingOptions::default(), &options).is_err());
    }
}

#[test]
fn publication_audits_records_and_refuses_to_replace_different_files() {
    let source = tempfile::tempdir().unwrap();
    let destination = tempfile::tempdir().unwrap();
    let writer = DatasetWriter::new(source.path()).unwrap();
    for sample in samples() {
        writer.write(&sample).unwrap();
    }
    writer.audit(&GroundingOptions::default()).unwrap();
    fs::write(source.path().join("README.md"), "dataset").unwrap();
    fs::write(source.path().join("collection-report.json"), "{}").unwrap();
    let publish = || {
        Command::new("python3")
            .arg(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/scripts/publish_training.py"
            ))
            .arg("--source")
            .arg(source.path())
            .arg("--destination")
            .arg(destination.path())
            .arg("--auditor")
            .arg(env!("CARGO_BIN_EXE_generate-training"))
            .output()
            .unwrap()
    };
    for _ in 0..2 {
        let result = publish();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
    }
    let manifest: Vec<serde_json::Value> =
        serde_json::from_slice(&fs::read(destination.path().join("manifest.json")).unwrap())
            .unwrap();
    assert_eq!(manifest.len(), 4);
    fs::write(
        destination.path().join("README.md"),
        "existing user content",
    )
    .unwrap();
    let result = publish();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("refusing to overwrite"));
    assert_eq!(
        fs::read_to_string(destination.path().join("README.md")).unwrap(),
        "existing user content"
    );
}
