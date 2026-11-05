use ptu_plan::instrumentation::*;
use ptu_plan::master::*;
use ptu_plan::parsing::json::{Program, TaskOrdering};
use ptu_plan::solver::*;
use std::collections::BTreeMap;

fn fixture() -> Program {
    serde_json::from_str(include_str!("fixtures/solver.json")).unwrap()
}

#[test]
fn full_events_replay_every_successful_oracle_call_and_preserve_results() {
    let model = ground_reachable_program(&fixture(), &GroundingOptions::default()).unwrap();
    let partitions = vec![
        RootPartition {
            id: 0,
            roots: vec![0],
        },
        RootPartition {
            id: 1,
            roots: vec![1],
        },
    ];
    let expected = solve_sequential(&model, &partitions, &MasterOptions::default()).unwrap();
    let mut events = Vec::new();
    let actual = solve_sequential_observed(
        &model,
        &partitions,
        &MasterOptions::default(),
        TraceLevel::Full,
        &mut |e| events.push(e),
    )
    .unwrap();
    assert_eq!(actual.status, expected.status);
    assert_eq!(
        actual.certificate.as_ref().unwrap().decisions,
        expected.certificate.as_ref().unwrap().decisions
    );
    let mut requests = BTreeMap::new();
    let mut finishes = 0;
    for event in &events {
        match event {
            MasterEvent::OracleStarted {
                call,
                request: Some(request),
                ..
            } => {
                assert!(requests.insert(*call, request).is_none());
            }
            MasterEvent::OracleFinished {
                call,
                status,
                statistics,
                certificate,
            } => {
                finishes += 1;
                assert!(statistics.peak_frontier > 0);
                if *status == ProofStatus::Feasible {
                    validate_certificate(&model, requests[call], certificate.as_ref().unwrap())
                        .unwrap();
                }
            }
            _ => {}
        }
    }
    assert_eq!(finishes, actual.statistics.local_calls);
    assert_eq!(
        actual.statistics.oracle_feasible
            + actual.statistics.oracle_infeasible
            + actual.statistics.oracle_bounded
            + actual.statistics.local_unknown,
        actual.statistics.local_calls
    );
    assert!(matches!(
        events.last(),
        Some(MasterEvent::Finished {
            status: MasterStatus::Feasible,
            ..
        })
    ));
    let bytes = serde_json::to_vec(&events).unwrap();
    let _: Vec<MasterEvent> = serde_json::from_slice(&bytes).unwrap();
}

#[test]
fn summary_traces_omit_worlds_and_unknown_oracles_never_emit_proof_events() {
    let model = ground_reachable_program(&fixture(), &GroundingOptions::default()).unwrap();
    let partitions = vec![RootPartition {
        id: 0,
        roots: vec![0, 1],
    }];
    let mut events = Vec::new();
    let options = MasterOptions {
        local_budget: SearchBudget {
            max_expansions: 0,
            ..Default::default()
        },
        ..Default::default()
    };
    let result = solve_sequential_observed(
        &model,
        &partitions,
        &options,
        TraceLevel::Summary,
        &mut |e| events.push(e),
    )
    .unwrap();
    assert_eq!(result.status, MasterStatus::BudgetExhausted);
    assert!(events.iter().all(|e| !matches!(
        e,
        MasterEvent::CutVerified { .. }
            | MasterEvent::OracleStarted {
                request: Some(_),
                ..
            }
    )));
}

#[test]
fn search_counters_measure_frontier_guards_ordering_and_dead_ends() {
    let mut p = fixture();
    p.problem
        .as_mut()
        .unwrap()
        .init_tn
        .as_mut()
        .unwrap()
        .tn
        .orderings = TaskOrdering::Total;
    let model = ground_reachable_program(&p, &GroundingOptions::default()).unwrap();
    let result = solve_problem(&model, &SearchBudget::default()).unwrap();
    assert!(result.statistics.peak_frontier > 0);
    assert!(result.statistics.peak_pending >= 2);
    assert!(result.statistics.guard_rejections > 0);
    assert!(result.statistics.ordering_blocked > 0);
    assert!(result.statistics.max_decomposition_depth > 0);
    p.problem.as_mut().unwrap().init_state.clear();
    let model = ground_reachable_program(&p, &GroundingOptions::default()).unwrap();
    assert!(
        solve_problem(&model, &SearchBudget::default())
            .unwrap()
            .statistics
            .dead_ends
            > 0
    );
    let legacy: SearchStatistics = serde_json::from_str("{\"expanded\":2}").unwrap();
    assert_eq!(legacy.expanded, 2);
    assert_eq!(legacy.peak_frontier, 0);
}
