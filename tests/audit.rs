use ptu_plan::instrumentation::{MasterEvent, TraceLevel};
use ptu_plan::learning::rollout::*;
use ptu_plan::parsing::json::Program;
use ptu_plan::solver::{BoundaryState, Condition, GroundAtom, ProofStatus};
use ptu_plan::store::DatasetStore;

fn fixture() -> Program {
    serde_json::from_str(include_str!("fixtures/solver.json")).unwrap()
}

#[test]
fn full_trace_requests_must_match_partition_budget_and_boundary_provenance() {
    let directory = tempfile::tempdir().unwrap();
    let store = DatasetStore::open(directory.path()).unwrap();
    let entries = collect_program(
        &store,
        &fixture(),
        None,
        &RolloutOptions {
            trace: TraceLevel::Full,
            ..Default::default()
        },
    )
    .unwrap();
    let original: RolloutRecord = store.load(&entries[0]).unwrap();
    audit_record(&store, &entries[0]).unwrap();
    // A failed request has no positive certificate to expose its wrong network.
    let mut bad = original.clone();
    let mut changed = false;
    for outcome in &mut bad.outcomes {
        let failed = outcome.events.iter().find_map(|event| match event {
            MasterEvent::OracleFinished {
                call,
                status: ProofStatus::Infeasible,
                ..
            } => Some(*call),
            _ => None,
        });
        for event in &mut outcome.events {
            if let MasterEvent::OracleStarted {
                call,
                request: Some(request),
                ..
            } = event
                && Some(*call) == failed
            {
                request.network.tasks.clear();
                request.network.precedence.clear();
                request.goal = Condition::False;
                changed = true;
            }
        }
    }
    assert!(changed);
    let entry = store.put(entries[0].metadata.clone(), &bad).unwrap();
    assert!(
        audit_record(&store, &entry)
            .unwrap_err()
            .to_string()
            .contains("induced network")
    );
    let mut bad = original.clone();
    for outcome in &mut bad.outcomes {
        for event in &mut outcome.events {
            if let MasterEvent::OracleStarted { budget, .. } = event {
                budget.max_expansions = bad.options.master.local_budget.max_expansions + 1;
            }
        }
    }
    let entry = store.put(entries[0].metadata.clone(), &bad).unwrap();
    assert!(
        audit_record(&store, &entry)
            .unwrap_err()
            .to_string()
            .contains("search limits")
    );
    let mut bad = original;
    for outcome in &mut bad.outcomes {
        for event in &mut outcome.events {
            if let MasterEvent::OracleStarted {
                interface_context: Some(context),
                ..
            } = event
            {
                context.future_reads.insert(GroundAtom {
                    predicate: "foreign".into(),
                    arguments: vec![],
                });
            }
        }
    }
    let entry = store.put(entries[0].metadata.clone(), &bad).unwrap();
    assert!(
        audit_record(&store, &entry)
            .unwrap_err()
            .to_string()
            .contains("future reads")
    );
}

#[test]
fn audit_cache_reuses_only_verified_proofs_and_the_same_compilation_scope() {
    let directory = tempfile::tempdir().unwrap();
    let store = DatasetStore::open(directory.path()).unwrap();
    let mut program = fixture();
    program.problem.as_mut().unwrap().init_state.clear();
    let entries = collect_program(
        &store,
        &program,
        None,
        &RolloutOptions {
            trace: TraceLevel::Full,
            conflict_core_checks: 4,
            ..Default::default()
        },
    )
    .unwrap();
    let mut duplicate: RolloutRecord = store.load(&entries[0]).unwrap();
    assert!(
        duplicate
            .outcomes
            .iter()
            .any(|o| !o.result.as_ref().unwrap().conflict_cores.is_empty())
    );
    duplicate.elapsed_ms += 123;
    let copied = store.put(entries[0].metadata.clone(), &duplicate).unwrap();
    let mut capped = duplicate.clone();
    capped.options.max_ground_instances = 1;
    let capped = store.put(entries[0].metadata.clone(), &capped).unwrap();
    // Keep the oracle query and all publication events unchanged apart from the
    // incoming assumptions. A cached exact query must not bless a broader cut.
    let mut forged = duplicate.clone();
    for outcome in &mut forged.outcomes {
        for core in &mut outcome.result.as_mut().unwrap().conflict_cores {
            core.incoming = Some(BoundaryState::default());
        }
        for event in &mut outcome.events {
            if let MasterEvent::ConflictCoreVerified { core } = event {
                core.incoming = Some(BoundaryState::default());
            }
        }
    }
    forged.validate().unwrap();
    let forged = store.put(entries[0].metadata.clone(), &forged).unwrap();
    let snapshot = store.snapshot().unwrap();
    let mut auditor = RolloutAuditor::new(&store, &snapshot);
    for entry in &entries {
        auditor.audit(entry).unwrap();
    }
    auditor.audit(&copied).unwrap();
    assert_eq!(auditor.statistics.prepared_problems, 1);
    assert!(auditor.statistics.core_verifications > 0);
    assert!(auditor.statistics.core_cache_hits > 0);
    assert!(
        auditor
            .audit(&forged)
            .unwrap_err()
            .to_string()
            .contains("dependency")
    );
    assert!(
        auditor
            .audit(&capped)
            .unwrap_err()
            .to_string()
            .contains("grounding budget")
    );
    duplicate.elapsed_ms += 1;
    let outside = store.put(entries[0].metadata.clone(), &duplicate).unwrap();
    assert!(
        auditor
            .audit(&outside)
            .unwrap_err()
            .to_string()
            .contains("snapshot")
    );
}

#[test]
fn full_trace_core_applications_require_matching_worlds_and_valid_cube_jumps() {
    let directory = tempfile::tempdir().unwrap();
    let store = DatasetStore::open(directory.path()).unwrap();
    let mut program = fixture();
    program.problem.as_mut().unwrap().init_state.clear();
    let entries = collect_program(
        &store,
        &program,
        None,
        &RolloutOptions {
            trace: TraceLevel::Full,
            conflict_core_checks: 4,
            ..Default::default()
        },
    )
    .unwrap();
    let original: RolloutRecord = store.load(&entries[0]).unwrap();
    original.validate().unwrap();
    let mut changed = false;
    let mut bad = original.clone();
    for outcome in &mut bad.outcomes {
        let result = outcome.result.as_ref().unwrap();
        for event in &mut outcome.events {
            if let MasterEvent::ConflictCoreApplied {
                core_index,
                initial_state: Some(state),
                ..
            } = event
            {
                let assumptions = result.conflict_cores[*core_index]
                    .incoming
                    .as_ref()
                    .unwrap();
                let atom = assumptions.must_be_false.iter().next().unwrap();
                state.insert(atom.clone());
                changed = true;
            }
        }
    }
    assert!(changed);
    assert!(
        bad.validate()
            .unwrap_err()
            .to_string()
            .contains("unjustified core")
    );
    let mut bad = original;
    for outcome in &mut bad.outcomes {
        for event in &mut outcome.events {
            if let MasterEvent::ConflictCoreApplied { next_ordinal, .. } = event {
                *next_ordinal += 1;
            }
        }
    }
    assert!(
        bad.validate()
            .unwrap_err()
            .to_string()
            .contains("unjustified core")
    );
}

#[test]
fn paired_core_benchmark_reports_budgets_order_and_independent_proofs() {
    let directory = tempfile::tempdir().unwrap();
    let store_path = directory.path().join("store");
    let store = DatasetStore::open(&store_path).unwrap();
    let mut impossible = fixture();
    impossible.problem.as_mut().unwrap().init_state.clear();
    for program in [fixture(), impossible] {
        collect_program(&store, &program, None, &RolloutOptions::default()).unwrap();
    }
    let output = directory.path().join("comparison.json");
    let result = std::process::Command::new(env!("CARGO_BIN_EXE_benchmark-cores"))
        .args(["--store", store_path.to_str().unwrap()])
        .args(["--output", output.to_str().unwrap()])
        .args(["--limit", "2", "--master-time-ms", "1000"])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(output).unwrap()).unwrap();
    assert_eq!(
        report["snapshot_id"],
        store.snapshot().unwrap().fingerprint().unwrap()
    );
    assert_eq!(report["master"]["max_time_ms"], 1000);
    let cases = report["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 2);
    assert_eq!(cases[0]["cores_first"], false);
    assert_eq!(cases[1]["cores_first"], true);
    let mut proofs = 0;
    for case in cases {
        assert!(case["error"].is_null());
        assert_eq!(case["baseline"]["status"], case["cores"]["status"]);
        assert_eq!(
            case["verified_cuts"].as_u64().unwrap() as usize,
            case["cores"]["conflict_cores"].as_array().unwrap().len()
        );
        proofs += case["verified_cuts"].as_u64().unwrap();
    }
    assert!(proofs > 0);
}

#[test]
fn causal_segment_contracts_are_audited_and_legacy_segments_remain_readable() {
    let directory = tempfile::tempdir().unwrap();
    let store = DatasetStore::open(directory.path()).unwrap();
    let entries = collect_program(&store, &fixture(), None, &RolloutOptions::default()).unwrap();
    let original: RolloutRecord = store.load(&entries[0]).unwrap();
    let result = original.outcomes[0].result.as_ref().unwrap();
    assert_eq!(result.statistics.causal_certificates, result.segments.len());
    assert!(result.statistics.causal_links > 0);
    let mut forged = original.clone();
    forged.outcomes[0].result.as_mut().unwrap().segments[0]
        .causal
        .as_mut()
        .unwrap()
        .causal_links
        .clear();
    let entry = store.put(entries[0].metadata.clone(), &forged).unwrap();
    assert!(
        audit_record(&store, &entry)
            .unwrap_err()
            .to_string()
            .contains("causal contract")
    );
    let mut value = serde_json::to_value(&original).unwrap();
    for outcome in value["outcomes"].as_array_mut().unwrap() {
        for segment in outcome["result"]["segments"].as_array_mut().unwrap() {
            segment.as_object_mut().unwrap().remove("causal");
        }
    }
    let legacy: RolloutRecord = serde_json::from_value(value).unwrap();
    let entry = store.put(entries[0].metadata.clone(), &legacy).unwrap();
    audit_record(&store, &entry).unwrap();
}
