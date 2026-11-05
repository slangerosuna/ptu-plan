use ptu_plan::{
    agglomeration::AgglomerationOptions,
    analysis::{NetworkId, analyze_program},
    escalation::{EscalationOptions, Repair, solve_escalating},
    instrumentation::TraceLevel,
    interaction::InteractionGraph,
    master::{MasterGuidance, MasterOptions, MasterStatus},
    parsing::json::Program,
    partition::MicroPartition,
    solver::{GroundingOptions, ground_reachable_program, validate_problem_certificate},
};
fn fixture() -> (
    ptu_plan::analysis::HtnAnalysis,
    InteractionGraph,
    ptu_plan::solver::GroundProblem,
    Vec<MicroPartition>,
) {
    let p: Program = serde_json::from_str(include_str!("fixtures/interleaving.json")).unwrap();
    let analysis = analyze_program(&p).unwrap();
    let graph = InteractionGraph::from_analysis(&analysis);
    let seeds = graph
        .network_nodes(NetworkId::Initial)
        .into_iter()
        .enumerate()
        .map(|(id, n)| MicroPartition { id, nodes: vec![n] })
        .collect();
    let problem = ground_reachable_program(&p, &GroundingOptions::default()).unwrap();
    (analysis, graph, problem, seeds)
}
#[test]
fn boundary_failure_coarsens_and_replays_interleaved_root_solution() {
    let (analysis, graph, problem, seeds) = fixture();
    let budget = MasterOptions {
        max_boundary_atoms: 0,
        max_time_ms: 1000,
        max_local_calls: 16,
        ..Default::default()
    };
    let result = solve_escalating(
        &problem,
        &analysis,
        &graph,
        &seeds,
        &AgglomerationOptions::default(),
        &budget,
        &MasterGuidance::default(),
        &EscalationOptions::default(),
        TraceLevel::Full,
    )
    .unwrap();
    assert_eq!(result.result.status, MasterStatus::Feasible);
    assert_eq!(result.attempts.len(), 2);
    assert!(matches!(result.attempts[1].repair, Repair::BoundaryMerge));
    assert!(result.attempts[0].statistics.boundary_cutoffs > 0);
    assert_eq!(result.attempts[1].partitions.len(), 1);
    assert!(result.work.local_calls <= budget.max_local_calls);
    validate_problem_certificate(&problem, result.result.certificate.as_ref().unwrap()).unwrap();
    assert_eq!(result.result.certificate.as_ref().unwrap().actions.len(), 4);
}
#[test]
fn size_limits_and_shared_call_budgets_remain_unknown_without_false_cuts() {
    let (analysis, graph, problem, seeds) = fixture();
    let result = solve_escalating(
        &problem,
        &analysis,
        &graph,
        &seeds,
        &AgglomerationOptions {
            max_cluster_size: 1,
            ..Default::default()
        },
        &MasterOptions {
            max_boundary_atoms: 0,
            ..Default::default()
        },
        &MasterGuidance::default(),
        &EscalationOptions::default(),
        TraceLevel::Summary,
    )
    .unwrap();
    assert_eq!(result.result.status, MasterStatus::BudgetExhausted);
    assert_eq!(result.stop_reason, "NoRepair");
    assert_eq!(result.attempts.len(), 1);
    assert!(result.result.nogoods.is_empty());
    let budget = MasterOptions {
        max_local_calls: 1,
        max_time_ms: 1000,
        ..Default::default()
    };
    let result = solve_escalating(
        &problem,
        &analysis,
        &graph,
        &seeds,
        &AgglomerationOptions::default(),
        &budget,
        &MasterGuidance::default(),
        &EscalationOptions::default(),
        TraceLevel::Summary,
    )
    .unwrap();
    assert!(result.work.local_calls <= 1);
    assert_eq!(result.result.status, MasterStatus::BudgetExhausted);
    assert_eq!(result.stop_reason, "SharedBudget");
}
#[test]
fn local_unknown_splits_once_and_does_not_cycle_back_to_identical_partition() {
    let (analysis, graph, problem, seeds) = fixture();
    let merged = vec![MicroPartition {
        id: 10,
        nodes: seeds.iter().flat_map(|s| s.nodes.clone()).collect(),
    }];
    let mut budget = MasterOptions {
        max_local_calls: 8,
        max_time_ms: 1000,
        ..Default::default()
    };
    budget.local_budget.max_expansions = 0;
    let result = solve_escalating(
        &problem,
        &analysis,
        &graph,
        &merged,
        &AgglomerationOptions::default(),
        &budget,
        &MasterGuidance::default(),
        &EscalationOptions::default(),
        TraceLevel::Summary,
    )
    .unwrap();
    assert_eq!(result.result.status, MasterStatus::BudgetExhausted);
    assert!(
        result
            .attempts
            .iter()
            .any(|a| matches!(a.repair, Repair::SplitLocal))
    );
    assert_eq!(
        result
            .attempts
            .iter()
            .filter(|a| matches!(a.repair, Repair::SplitLocal))
            .count(),
        1
    );
    assert!(result.work.local_calls <= 8);
    assert!(
        result
            .attempts
            .iter()
            .all(|a| a.partitions.iter().all(|p| !p.roots.is_empty()))
    );
}
#[test]
fn planner_cli_reports_repartition_attempts() {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_ptu-plan"))
        .args([
            "tests/fixtures/interleaving.json",
            "--sequential",
            "--micro-max-size",
            "1",
            "--max-boundary-atoms",
            "0",
            "--repartition",
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
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["sequential"]["status"], "Feasible");
    assert_eq!(
        report["repartition"]["attempts"].as_array().unwrap().len(),
        2
    );
}

#[test]
fn whole_network_fallback_bridges_disconnected_goal_boundaries_but_obeys_size_cap() {
    let p: Program =
        serde_json::from_str(include_str!("fixtures/disconnected-goals.json")).unwrap();
    let analysis = analyze_program(&p).unwrap();
    let graph = InteractionGraph::from_analysis(&analysis);
    let problem = ground_reachable_program(&p, &GroundingOptions::default()).unwrap();
    let seeds: Vec<_> = graph
        .network_nodes(NetworkId::Initial)
        .into_iter()
        .enumerate()
        .map(|(id, n)| MicroPartition { id, nodes: vec![n] })
        .collect();
    let merge = AgglomerationOptions::default();
    let state =
        ptu_plan::agglomeration::observe(&analysis, &graph, NetworkId::Initial, &seeds, &merge)
            .unwrap();
    assert!(
        ptu_plan::agglomeration::legal_candidates(&graph, NetworkId::Initial, &state, &merge)
            .unwrap()
            .is_empty()
    );
    let budget = MasterOptions {
        max_time_ms: 1000,
        max_boundary_atoms: 0,
        ..Default::default()
    };
    let solved = solve_escalating(
        &problem,
        &analysis,
        &graph,
        &seeds,
        &merge,
        &budget,
        &MasterGuidance::default(),
        &EscalationOptions::default(),
        TraceLevel::Summary,
    )
    .unwrap();
    assert_eq!(solved.result.status, MasterStatus::Feasible);
    assert!(matches!(
        solved.attempts[1].repair,
        Repair::MonolithicFallback
    ));
    validate_problem_certificate(&problem, solved.result.certificate.as_ref().unwrap()).unwrap();
    for (config, merge) in [
        (
            EscalationOptions {
                monolithic_fallback: false,
                ..Default::default()
            },
            merge,
        ),
        (
            EscalationOptions::default(),
            AgglomerationOptions {
                max_cluster_size: 1,
                ..merge
            },
        ),
    ] {
        let unknown = solve_escalating(
            &problem,
            &analysis,
            &graph,
            &seeds,
            &merge,
            &budget,
            &MasterGuidance::default(),
            &config,
            TraceLevel::Summary,
        )
        .unwrap();
        assert_eq!(unknown.result.status, MasterStatus::BudgetExhausted);
        assert_eq!(unknown.attempts.len(), 1);
    }
}
