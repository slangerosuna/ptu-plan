use std::process::Command;

use ptu_plan::master::*;
use ptu_plan::parsing::json::Program;
use ptu_plan::solver::*;
use serde_json::{Value, json};

fn atom(name: &str) -> Value {
    json!({"Atom": {"name": name, "variables": []}})
}
fn sub(name: &str) -> Value {
    json!({"task": {"name": name}, "terms": []})
}
fn action(name: &str, guard: Value, effect: Value) -> Value {
    json!({"name": name, "parameters": [], "preconditions": guard, "effects": effect})
}
fn method(name: &str, task: &str, guard: Value, children: &[&str]) -> Value {
    json!({"name": {"name": name}, "task": {"name": task}, "task_terms": [], "params": [],
        "precondition": guard, "tn": {"subtasks": children.iter().map(|s| sub(s)).collect::<Vec<_>>(), "orderings": "Total"}})
}
fn program(
    actions: Vec<Value>,
    methods: Vec<Value>,
    compounds: &[&str],
    roots: &[&str],
    ordered: bool,
    goal: Value,
) -> Program {
    serde_json::from_value(json!({"domain": {
        "name": "master-fixture", "requirements": ["Hierarchy"],
        "predicates": (["p", "q", "extra"].iter().map(|s| json!({"name": s,"variables":[]})).collect::<Vec<_>>()),
        "compound_tasks": compounds.iter().map(|s| json!({"name": s, "parameters":[]})).collect::<Vec<_>>(),
        "actions": actions, "methods": methods, "functions": []},
        "problem": {"name":"test", "domain_name":"master-fixture", "requirements":["Hierarchy"],
        "objects":[], "init_state":[], "goal":goal,
        "init_tn":{"tn":{"subtasks":roots.iter().map(|s| sub(s)).collect::<Vec<_>>(),
            "orderings": if ordered { json!("Total") } else { json!({"Partial":[]}) }}}}
    })).unwrap()
}
fn backtracking() -> Program {
    program(
        vec![
            action("produce", Value::Null, atom("p")),
            action("consume", Value::Null, atom("q")),
        ],
        vec![
            method("skip", "choice", Value::Null, &[]),
            method("produce-p", "choice", Value::Null, &["produce"]),
            method("guarded-method", "finish", atom("p"), &["consume"]),
        ],
        &["choice", "finish"],
        &["choice", "finish"],
        true,
        atom("q"),
    )
}
fn model(p: &Program) -> GroundProblem {
    ground_program(p, &GroundingOptions::default()).unwrap()
}
fn singletons(count: usize) -> Vec<RootPartition> {
    (0..count)
        .map(|i| RootPartition {
            id: i,
            roots: vec![i],
        })
        .collect()
}

#[test]
fn revises_earlier_boundary_instead_of_accepting_independent_feasibility() {
    let model = model(&backtracking());
    let result = solve_sequential(&model, &singletons(2), &MasterOptions::default()).unwrap();
    assert_eq!(result.status, MasterStatus::Feasible);
    assert_eq!(result.segments.len(), 2);
    assert_eq!(
        result.segments[0]
            .outgoing
            .must_be_true
            .iter()
            .next()
            .unwrap()
            .predicate,
        "p"
    );
    assert!(result.segments[1].incoming.must_be_false.is_empty());
    assert_eq!(result.segments[1].incoming.must_be_true.len(), 1); // method guard, not action guard
    assert!(!result.nogoods.is_empty());
    for cut in &result.nogoods {
        assert_eq!(cut.proof, ProofStatus::Infeasible);
        let partition = &singletons(2)[cut.partition];
        let request = LocalRequest {
            network: model.initial_networks[cut.initial_network]
                .network
                .induced(&partition.roots)
                .unwrap(),
            initial_state: cut.initial_state.clone(),
            goal: cut.goal.clone(),
        };
        assert_eq!(
            solve_local(&model, &request, &SearchBudget::default())
                .unwrap()
                .status,
            ProofStatus::Infeasible
        );
    }
    let cert = result.certificate.unwrap();
    assert_eq!(
        cert.actions
            .iter()
            .map(|t| t.name.as_str())
            .collect::<Vec<_>>(),
        ["produce", "consume"]
    );
    validate_problem_certificate(&model, &cert).unwrap();
}

#[test]
fn tries_another_topological_block_order_when_the_first_consumer_is_disabled() {
    let mut p = program(
        vec![
            action("consumer", atom("p"), atom("q")),
            action("producer", Value::Null, atom("p")),
        ],
        vec![],
        &[],
        &["consumer", "producer"],
        false,
        atom("q"),
    );
    let model1 = model(&p);
    let result = solve_sequential(&model1, &singletons(2), &MasterOptions::default()).unwrap();
    assert_eq!(result.status, MasterStatus::Feasible);
    assert_eq!(result.segments[0].partition.id, 1);
    validate_problem_certificate(&model1, &result.certificate.unwrap()).unwrap();
    p.problem
        .as_mut()
        .unwrap()
        .init_tn
        .as_mut()
        .unwrap()
        .tn
        .orderings = ptu_plan::parsing::json::TaskOrdering::Total;
    assert_eq!(
        solve_sequential(&model(&p), &singletons(2), &MasterOptions::default())
            .unwrap()
            .status,
        MasterStatus::SequentiallyInfeasible
    );
}

#[test]
fn interleaving_failure_is_scoped_and_a_merge_recovers_the_global_solution() {
    let p = program(
        vec![
            action("make-p", Value::Null, atom("p")),
            action("need-q", atom("q"), Value::Null),
            action("make-q", Value::Null, atom("q")),
            action("need-p", atom("p"), Value::Null),
        ],
        vec![
            method("left", "a", Value::Null, &["make-p", "need-q"]),
            method("right", "b", Value::Null, &["make-q", "need-p"]),
        ],
        &["a", "b"],
        &["a", "b"],
        false,
        json!({"And":[atom("p"),atom("q")]}),
    );
    let model = model(&p);
    assert_eq!(
        solve_problem(&model, &SearchBudget::default())
            .unwrap()
            .status,
        ProofStatus::Feasible
    );
    let restricted = solve_sequential(&model, &singletons(2), &MasterOptions::default()).unwrap();
    assert_eq!(restricted.status, MasterStatus::SequentiallyInfeasible);
    assert!(restricted.certificate.is_none());
    let merged = solve_sequential(
        &model,
        &[RootPartition {
            id: 20,
            roots: vec![0, 1],
        }],
        &MasterOptions::default(),
    )
    .unwrap();
    assert_eq!(merged.status, MasterStatus::Feasible);
    validate_problem_certificate(&model, &merged.certificate.unwrap()).unwrap();
}

#[test]
fn no_cuts_from_unknown_results_and_bounded_proofs_keep_their_scope() {
    let model = model(&backtracking());
    for options in [
        MasterOptions {
            max_local_calls: 0,
            ..MasterOptions::default()
        },
        MasterOptions {
            max_master_nodes: 0,
            ..MasterOptions::default()
        },
        MasterOptions {
            max_time_ms: 0,
            ..MasterOptions::default()
        },
        MasterOptions {
            max_boundary_atoms: 0,
            ..MasterOptions::default()
        },
        MasterOptions {
            local_budget: SearchBudget {
                max_expansions: 0,
                ..SearchBudget::default()
            },
            ..MasterOptions::default()
        },
    ] {
        let result = solve_sequential(&model, &singletons(2), &options).unwrap();
        assert_eq!(result.status, MasterStatus::BudgetExhausted);
        assert!(result.nogoods.is_empty());
    }
    let exactness = ExactnessMode::Bounded {
        max_decomposition_depth: 0,
    };
    let bounded = solve_sequential(
        &model,
        &singletons(2),
        &MasterOptions {
            local_budget: SearchBudget {
                exactness,
                ..SearchBudget::default()
            },
            ..MasterOptions::default()
        },
    )
    .unwrap();
    assert_eq!(bounded.status, MasterStatus::InfeasibleWithinBound);
    assert!(
        bounded
            .nogoods
            .iter()
            .all(|n| n.exactness == exactness && n.proof == ProofStatus::InfeasibleWithinBound)
    );
}

#[test]
fn invalid_coverage_is_rejected_and_ordering_quotient_cycles_are_explicit() {
    let p = program(
        vec![action("step", Value::Null, Value::Null)],
        vec![],
        &[],
        &["step", "step", "step"],
        true,
        Value::Null,
    );
    let model = model(&p);
    for partitions in [
        vec![],
        vec![RootPartition {
            id: 0,
            roots: vec![0, 0, 1, 2],
        }],
        vec![RootPartition {
            id: 0,
            roots: vec![0, 1, 3],
        }],
        vec![
            RootPartition {
                id: 0,
                roots: vec![0],
            },
            RootPartition {
                id: 0,
                roots: vec![1, 2],
            },
        ],
    ] {
        assert!(solve_sequential(&model, &partitions, &MasterOptions::default()).is_err());
    }
    let result = solve_sequential(
        &model,
        &[
            RootPartition {
                id: 4,
                roots: vec![0, 2],
            },
            RootPartition {
                id: 5,
                roots: vec![1],
            },
        ],
        &MasterOptions::default(),
    )
    .unwrap();
    assert_eq!(result.status, MasterStatus::NotSequentiallyComposable);
    assert_eq!(result.statistics.local_calls, 0);
    let result = solve_sequential(&model, &singletons(3), &MasterOptions::default()).unwrap();
    assert_eq!(
        result
            .certificate
            .unwrap()
            .decisions
            .iter()
            .map(|d| d.occurrence)
            .collect::<Vec<_>>(),
        [0, 1, 2]
    );
}

#[test]
fn negative_contracts_and_frame_facts_survive_composition() {
    let p = program(
        vec![
            action("clear", Value::Null, json!({"Not":atom("p")})),
            action("finish", json!({"Not":atom("p")}), atom("q")),
        ],
        vec![],
        &[],
        &["clear", "finish"],
        true,
        json!({"And":[atom("q"),atom("extra"),{"Not":atom("p")}]}),
    );
    let mut model = model(&p);
    for predicate in ["p", "extra"] {
        model.initial_state.insert(GroundAtom {
            predicate: predicate.into(),
            arguments: vec![],
        });
    }
    let result = solve_sequential(&model, &singletons(2), &MasterOptions::default()).unwrap();
    assert_eq!(result.status, MasterStatus::Feasible);
    assert_eq!(
        result.segments[0]
            .outgoing
            .must_be_false
            .iter()
            .next()
            .unwrap()
            .predicate,
        "p"
    );
    assert!(result.segments[0].incoming.must_be_true.is_empty()); // unmentioned is unknown, not false
    let cert = result.certificate.unwrap();
    assert!(cert.final_state.iter().any(|a| a.predicate == "extra"));
    validate_problem_certificate(&model, &cert).unwrap();
}

#[test]
fn initial_parameter_groundings_and_nested_occurrence_ids_are_replayed() {
    let mut p: Program = serde_json::from_str(include_str!("fixtures/solver.json")).unwrap();
    let problem = p.problem.as_mut().unwrap();
    let initial = problem.init_tn.as_mut().unwrap();
    let mut root = initial.tn.subtasks[0].clone();
    root.terms[0].name = "?x".into();
    initial.tn.subtasks = vec![root];
    initial.tn.constraints = None;
    initial.parameters = Some(vec![ptu_plan::parsing::json::Symbol {
        name: "?x".into(),
        symbol_type: Some("item".into()),
    }]);
    problem.goal = Some(
        serde_json::from_value(json!({"Atom":{"name":"done","variables":[{"name":"b"}]}})).unwrap(),
    );
    let model = model(&p);
    let result = solve_sequential(&model, &singletons(1), &MasterOptions::default()).unwrap();
    assert_eq!(result.status, MasterStatus::Feasible);
    let cert = result.certificate.unwrap();
    assert_eq!(cert.initial_network, 1);
    validate_problem_certificate(&model, &cert).unwrap();
}

#[test]
fn sequential_cli_emits_composed_certificate_and_preserves_analysis_fields() {
    let output = Command::new(env!("CARGO_BIN_EXE_ptu-plan"))
        .args([
            "tests/fixtures/solver.json",
            "--analysis-json",
            "--sequential",
            "--agglomerate",
            "--micro-max-size",
            "1",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let record: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert!(record.get("analysis").is_some());
    assert!(record.get("graph").is_some());
    assert!(record.get("agglomeration").is_some());
    assert_eq!(record["sequential"]["status"], "Feasible");
    assert_eq!(
        record["sequential"]["certificate"]["actions"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
}

#[test]
fn composes_nested_repeated_tasks_and_successful_empty_refinements() {
    let p = program(
        vec![
            action("produce", Value::Null, atom("p")),
            action("consume", atom("p"), atom("q")),
        ],
        vec![
            method("outer-body", "outer", Value::Null, &["inner", "inner"]),
            method("inner-body", "inner", Value::Null, &["produce", "empty"]),
            method("empty-body", "empty", Value::Null, &[]),
            method("finish-body", "finish", Value::Null, &["consume"]),
        ],
        &["outer", "inner", "empty", "finish"],
        &["outer", "finish", "empty"],
        true,
        atom("q"),
    );
    let model = model(&p);
    let result = solve_sequential(&model, &singletons(3), &MasterOptions::default()).unwrap();
    assert_eq!(result.status, MasterStatus::Feasible);
    assert!(result.segments[2].certificate.actions.is_empty());
    let cert = result.certificate.unwrap();
    assert_eq!(cert.actions.len(), 3);
    validate_problem_certificate(&model, &cert).unwrap();
}

#[test]
fn boundary_enumeration_matches_monolithic_search_on_small_choice_models() {
    // Each method has one primitive child and no refinement guard, so every
    // global execution has an equivalent contiguous root-block execution.
    // Compare existence across varied guards, negative effects, and choices.
    let mut seed = 47u64;
    let mut next = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (seed >> 32) as usize
    };
    let literals = [
        Value::Null,
        atom("p"),
        json!({"Not":atom("p")}),
        atom("q"),
        json!({"Not":atom("q")}),
    ];
    for case in 0..48 {
        let mut actions = Vec::new();
        let mut methods = Vec::new();
        for task in ["a", "b", "c"] {
            for choice in 0..2 {
                let name = format!("{task}-{choice}");
                actions.push(action(
                    &name,
                    literals[next() % literals.len()].clone(),
                    literals[next() % literals.len()].clone(),
                ));
                methods.push(method(&format!("m-{name}"), task, Value::Null, &[&name]));
            }
        }
        let mut p = program(
            actions,
            methods,
            &["a", "b", "c"],
            &["a", "b", "c"],
            case % 2 == 0,
            literals[1 + next() % (literals.len() - 1)].clone(),
        );
        for predicate in ["p", "q"] {
            if next() % 2 == 0 {
                p.problem
                    .as_mut()
                    .unwrap()
                    .init_state
                    .push(ptu_plan::parsing::json::Predicate {
                        name: predicate.into(),
                        variables: vec![],
                    });
            }
        }
        let model = model(&p);
        let mono = solve_problem(&model, &SearchBudget::default()).unwrap();
        let reachable = ground_reachable_program(&p, &GroundingOptions::default()).unwrap();
        let reduced = solve_problem(&reachable, &SearchBudget::default()).unwrap();
        assert_eq!(
            mono.status, reduced.status,
            "reachable grounding case {case}"
        );
        if let Some(certificate) = reduced.certificate {
            validate_problem_certificate(&reachable, &certificate).unwrap();
        }
        let split = solve_sequential(&model, &singletons(3), &MasterOptions::default()).unwrap();
        let guided = solve_sequential_guided(
            &model,
            &singletons(3),
            &MasterOptions::default(),
            &MasterGuidance {
                conflict_core_checks: 8,
                ..Default::default()
            },
            ptu_plan::instrumentation::TraceLevel::Summary,
            None,
        )
        .unwrap();
        assert_eq!(guided.status, split.status, "core-enabled case {case}");
        if let Some(certificate) = &guided.certificate {
            validate_problem_certificate(&model, certificate).unwrap();
        }
        for core in &guided.conflict_cores {
            ptu_plan::conflicts::validate_conflict_core(
                &model,
                &singletons(3)[core.partition],
                core,
                &SearchBudget::default(),
            )
            .unwrap();
        }
        assert!(matches!(
            mono.status,
            ProofStatus::Feasible | ProofStatus::Infeasible
        ));
        assert!(matches!(
            split.status,
            MasterStatus::Feasible | MasterStatus::SequentiallyInfeasible
        ));
        assert_eq!(
            mono.status == ProofStatus::Feasible,
            split.status == MasterStatus::Feasible,
            "case {case}"
        );
        if let Some(certificate) = split.certificate {
            validate_problem_certificate(&model, &certificate).unwrap();
        }
    }
}

#[test]
fn cli_preserves_unknown_status_and_rejects_ambiguous_search_modes() {
    let output = Command::new(env!("CARGO_BIN_EXE_ptu-plan"))
        .args([
            "tests/fixtures/solver.json",
            "--analysis-json",
            "--sequential",
            "--max-local-calls",
            "0",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let record: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(record["sequential"]["status"], "BudgetExhausted");
    assert!(
        record["sequential"]["nogoods"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    for flags in [
        vec!["--solve", "--sequential"],
        vec!["--max-decomposition-depth", "1"],
    ] {
        let result = Command::new(env!("CARGO_BIN_EXE_ptu-plan"))
            .arg("tests/fixtures/solver.json")
            .args(flags)
            .output()
            .unwrap();
        assert!(!result.status.success());
    }
}

#[test]
fn caches_proven_failed_suffixes_across_equivalent_block_orders() {
    let p = program(
        vec![action("noop", Value::Null, Value::Null)],
        vec![],
        &[],
        &["noop", "noop", "noop", "noop", "noop", "noop", "noop"],
        false,
        atom("p"),
    );
    let model = model(&p);
    let result = solve_sequential(&model, &singletons(7), &MasterOptions::default()).unwrap();
    assert_eq!(result.status, MasterStatus::SequentiallyInfeasible);
    assert!(result.statistics.suffix_cache_hits > 0);
    assert!(result.statistics.local_calls < 1000);
    let unknown = solve_sequential(
        &model,
        &singletons(7),
        &MasterOptions {
            local_budget: SearchBudget {
                max_expansions: 0,
                ..SearchBudget::default()
            },
            ..MasterOptions::default()
        },
    )
    .unwrap();
    assert_eq!(unknown.status, MasterStatus::BudgetExhausted);
    assert_eq!(unknown.statistics.suffix_cache_hits, 0);
}

#[test]
fn benders_empty_core_skips_contracts_and_every_cut_reverifies() {
    let p = program(
        vec![
            action(
                "blocked",
                atom("extra"),
                json!({"And":[atom("p"),atom("q")]}),
            ),
            action("finish", json!({"And":[atom("p"),atom("q")]}), Value::Null),
        ],
        vec![],
        &[],
        &["blocked", "finish"],
        true,
        atom("q"),
    );
    let model = model(&p);
    let partitions = singletons(2);
    let baseline = solve_sequential(&model, &partitions, &MasterOptions::default()).unwrap();
    let mut events = Vec::new();
    let result = solve_sequential_guided(
        &model,
        &partitions,
        &MasterOptions::default(),
        &MasterGuidance {
            conflict_core_checks: 16,
            ..Default::default()
        },
        ptu_plan::instrumentation::TraceLevel::Full,
        Some(&mut |e| events.push(e)),
    )
    .unwrap();
    assert_eq!(result.status, baseline.status);
    assert!(result.statistics.core_hits > 0);
    assert!(result.statistics.local_calls < baseline.statistics.local_calls);
    assert!(
        result
            .conflict_cores
            .iter()
            .any(|c| c.core == BoundaryState::default())
    );
    for core in &result.conflict_cores {
        ptu_plan::conflicts::validate_conflict_core(
            &model,
            &partitions[core.partition],
            core,
            &SearchBudget::default(),
        )
        .unwrap();
    }
    assert_eq!(
        events
            .iter()
            .filter(|e| matches!(
                e,
                ptu_plan::instrumentation::MasterEvent::ConflictCoreVerified { .. }
            ))
            .count(),
        result.conflict_cores.len()
    );
}

#[test]
fn benders_literal_core_generalizes_only_outgoing_supersets_in_the_same_world() {
    let p = program(
        vec![
            action("q", Value::Null, atom("q")),
            action("p", Value::Null, atom("p")),
            action("finish", atom("p"), Value::Null),
        ],
        vec![
            method("skip", "choice", Value::Null, &[]),
            method("q", "choice", Value::Null, &["q"]),
            method("blocked", "choice", atom("extra"), &["p"]),
        ],
        &["choice"],
        &["choice", "finish"],
        true,
        atom("q"),
    );
    let model = model(&p);
    let partitions = singletons(2);
    let result = solve_sequential_guided(
        &model,
        &partitions,
        &MasterOptions::default(),
        &MasterGuidance {
            conflict_core_checks: 16,
            ..Default::default()
        },
        ptu_plan::instrumentation::TraceLevel::Summary,
        None,
    )
    .unwrap();
    assert_eq!(result.status, MasterStatus::SequentiallyInfeasible);
    let core = result
        .conflict_cores
        .iter()
        .find(|c| c.partition == 0 && c.core.must_be_true.iter().any(|a| a.predicate == "p"))
        .unwrap();
    assert_eq!(core.core.must_be_true.len(), 1);
    assert!(core.core.must_be_false.is_empty());
    for p in [false, true] {
        for q in [false, true] {
            let mut goal = BoundaryState::default();
            for (name, value) in [("p", p), ("q", q)] {
                let a = GroundAtom {
                    predicate: name.into(),
                    arguments: vec![],
                };
                if value {
                    goal.must_be_true.insert(a);
                } else {
                    goal.must_be_false.insert(a);
                }
            }
            let goal = goal.condition().unwrap();
            let matches = core.matches(0, 0, &model.initial_state, &goal, ExactnessMode::Unbounded);
            assert_eq!(matches, p);
            if matches {
                let request = LocalRequest {
                    network: model.initial_networks[0].network.induced(&[0]).unwrap(),
                    initial_state: model.initial_state.clone(),
                    goal,
                };
                assert_eq!(
                    solve_local(&model, &request, &SearchBudget::default())
                        .unwrap()
                        .status,
                    ProofStatus::Infeasible
                );
            }
        }
    }
    let mut other = model.initial_state.clone();
    other.insert(GroundAtom {
        predicate: "extra".into(),
        arguments: vec![],
    });
    assert!(!core.matches(0, 0, &other, &core.source_goal, ExactnessMode::Unbounded));
    assert!(!core.matches(
        1,
        0,
        &model.initial_state,
        &core.source_goal,
        ExactnessMode::Unbounded
    ));
    assert!(!core.matches(
        0,
        1,
        &model.initial_state,
        &core.source_goal,
        ExactnessMode::Unbounded
    ));
}

#[test]
fn core_probes_preserve_unknown_and_depth_bound_scopes() {
    let model = model(&backtracking());
    let partitions = singletons(2);
    let guidance = MasterGuidance {
        conflict_core_checks: 8,
        ..Default::default()
    };
    let unknown = solve_sequential_guided(
        &model,
        &partitions,
        &MasterOptions {
            local_budget: SearchBudget {
                max_expansions: 0,
                ..Default::default()
            },
            ..Default::default()
        },
        &guidance,
        ptu_plan::instrumentation::TraceLevel::Full,
        None,
    )
    .unwrap();
    assert_eq!(unknown.status, MasterStatus::BudgetExhausted);
    assert_eq!(unknown.statistics.core_checks, 0);
    assert!(unknown.conflict_cores.is_empty());
    let options = MasterOptions {
        local_budget: SearchBudget {
            exactness: ExactnessMode::Bounded {
                max_decomposition_depth: 0,
            },
            ..Default::default()
        },
        ..Default::default()
    };
    let bounded = solve_sequential_guided(
        &model,
        &partitions,
        &options,
        &guidance,
        ptu_plan::instrumentation::TraceLevel::Summary,
        None,
    )
    .unwrap();
    assert_eq!(bounded.status, MasterStatus::InfeasibleWithinBound);
    assert!(!bounded.conflict_cores.is_empty());
    for core in &bounded.conflict_cores {
        assert_eq!(core.proof, ProofStatus::InfeasibleWithinBound);
        ptu_plan::conflicts::validate_conflict_core(
            &model,
            &partitions[core.partition],
            core,
            &options.local_budget,
        )
        .unwrap();
        assert!(!core.matches(
            0,
            core.partition,
            &core.initial_state,
            &core.source_goal,
            ExactnessMode::Unbounded
        ));
        let mut forged = core.clone();
        forged.exactness = ExactnessMode::Unbounded;
        assert!(forged.validate_scope().is_err());
    }
}

#[test]
fn incoming_cores_reuse_failures_across_irrelevant_world_changes() {
    let p = program(
        vec![
            action("mark", Value::Null, atom("extra")),
            action("blocked", atom("p"), atom("q")),
            action("observe", atom("extra"), Value::Null),
        ],
        vec![
            method("skip", "prefix", Value::Null, &[]),
            method("mark", "prefix", Value::Null, &["mark"]),
        ],
        &["prefix"],
        &["prefix", "blocked", "observe"],
        true,
        atom("q"),
    );
    let model = model(&p);
    let partitions = singletons(3);
    let baseline = solve_sequential(&model, &partitions, &MasterOptions::default()).unwrap();
    let mut events = Vec::new();
    let result = solve_sequential_guided(
        &model,
        &partitions,
        &MasterOptions::default(),
        &MasterGuidance {
            conflict_core_checks: 8,
            ..Default::default()
        },
        ptu_plan::instrumentation::TraceLevel::Full,
        Some(&mut |e| events.push(e)),
    )
    .unwrap();
    assert_eq!(result.status, baseline.status);
    assert!(result.statistics.core_generalized_hits > 0);
    assert!(result.statistics.local_calls < baseline.statistics.local_calls);
    let core = result
        .conflict_cores
        .iter()
        .find(|c| c.partition == 1 && c.core == BoundaryState::default())
        .unwrap();
    assert_eq!(core.incoming.as_ref().unwrap().must_be_false.len(), 1);
    assert_eq!(
        core.incoming
            .as_ref()
            .unwrap()
            .must_be_false
            .iter()
            .next()
            .unwrap()
            .predicate,
        "p"
    );
    // Exhaust all closed-world assignments, independently checking every pruned world.
    for mask in 0..8 {
        let state: State = ["p", "q", "extra"]
            .into_iter()
            .enumerate()
            .filter(|(i, _)| mask & (1 << i) != 0)
            .map(|(_, s)| GroundAtom {
                predicate: s.into(),
                arguments: vec![],
            })
            .collect();
        assert_eq!(
            core.matches(0, 1, &state, &Condition::True, ExactnessMode::Unbounded),
            mask & 1 == 0
        );
        if core.matches(0, 1, &state, &Condition::True, ExactnessMode::Unbounded) {
            assert_eq!(
                solve_local(
                    &model,
                    &LocalRequest {
                        network: model.initial_networks[0].network.induced(&[1]).unwrap(),
                        initial_state: state,
                        goal: Condition::True
                    },
                    &SearchBudget::default()
                )
                .unwrap()
                .status,
                ProofStatus::Infeasible
            );
        }
    }
    let mut forged = core.clone();
    forged.incoming.as_mut().unwrap().must_be_false.clear();
    assert!(
        ptu_plan::conflicts::validate_conflict_core(
            &model,
            &partitions[1],
            &forged,
            &SearchBudget::default()
        )
        .is_err()
    );
    let mut legacy = serde_json::to_value(core).unwrap();
    legacy.as_object_mut().unwrap().remove("incoming");
    let legacy: ptu_plan::conflicts::VerifiedConflictCore = serde_json::from_value(legacy).unwrap();
    let mut different = core.initial_state.clone();
    different.insert(GroundAtom {
        predicate: "extra".into(),
        arguments: vec![],
    });
    assert!(!legacy.matches(0, 1, &different, &Condition::True, ExactnessMode::Unbounded));
    assert!(events.iter().any(|e| matches!(
        e,
        ptu_plan::instrumentation::MasterEvent::ConflictCoreApplied { .. }
    )));
}

#[test]
fn empty_core_jumps_a_wide_boundary_without_enumerating_it() {
    let names: Vec<_> = (0..16).map(|i| format!("x{i}")).collect();
    let literals: Vec<_> = names.iter().map(|s| atom(s)).collect();
    let mut p = program(
        vec![
            action("blocked", atom("extra"), json!({"And":literals})),
            action("finish", json!({"And":literals}), Value::Null),
        ],
        vec![],
        &[],
        &["blocked", "finish"],
        true,
        atom("x0"),
    );
    for name in names {
        p.domain
            .predicates
            .push(serde_json::from_value(json!({"name":name,"variables":[]})).unwrap());
    }
    let model = model(&p);
    let result = solve_sequential_guided(
        &model,
        &singletons(2),
        &MasterOptions {
            max_boundary_atoms: 16,
            max_local_calls: 4,
            ..Default::default()
        },
        &MasterGuidance {
            conflict_core_checks: 4,
            ..Default::default()
        },
        ptu_plan::instrumentation::TraceLevel::Summary,
        None,
    )
    .unwrap();
    assert_eq!(result.status, MasterStatus::SequentiallyInfeasible);
    assert_eq!(result.statistics.local_calls, 2);
    assert_eq!(result.statistics.interface_candidates, 2);
    assert_eq!(result.statistics.core_assignments_skipped, (1 << 16) - 1);
}

#[test]
fn incoming_scope_preserves_goal_frame_facts_even_without_guard_reads() {
    let p = program(
        vec![action("noop", Value::Null, Value::Null)],
        vec![],
        &[],
        &["noop"],
        true,
        atom("extra"),
    );
    let model = model(&p);
    let partitions = singletons(1);
    let result = solve_sequential_guided(
        &model,
        &partitions,
        &MasterOptions::default(),
        &MasterGuidance {
            conflict_core_checks: 8,
            ..Default::default()
        },
        ptu_plan::instrumentation::TraceLevel::Full,
        None,
    )
    .unwrap();
    assert_eq!(result.statistics.core_checks, 0); // Fixed final goal needs no deletion probes.
    let cut = &result.conflict_cores[0];
    assert_eq!(cut.verification_call, cut.source_call);
    assert_eq!(cut.removed_literals, 0);
    assert!(
        cut.incoming
            .as_ref()
            .unwrap()
            .must_be_false
            .iter()
            .any(|a| a.predicate == "extra")
    );
    let mut state = State::new();
    state.insert(GroundAtom {
        predicate: "extra".into(),
        arguments: vec![],
    });
    assert!(!cut.matches(0, 0, &state, &model.goal, ExactnessMode::Unbounded));
    assert_eq!(
        solve_local(
            &model,
            &LocalRequest {
                network: model.initial_networks[0].network.clone(),
                initial_state: state,
                goal: model.goal.clone()
            },
            &SearchBudget::default()
        )
        .unwrap()
        .status,
        ProofStatus::Feasible
    );
    let mut forged = cut.clone();
    forged.incoming = Some(BoundaryState::default());
    assert!(
        ptu_plan::conflicts::validate_conflict_core(
            &model,
            &partitions[0],
            &forged,
            &SearchBudget::default()
        )
        .is_err()
    );
}

#[test]
fn astronomical_cube_skip_counters_saturate_without_overflow() {
    let width = usize::BITS as usize - 1;
    let names: Vec<_> = (0..width).map(|i| format!("x{i}")).collect();
    let literals: Vec<_> = names.iter().map(|s| atom(s)).collect();
    let mut p = program(
        vec![action("blocked", atom("extra"), json!({"And":literals}))],
        vec![],
        &[],
        &["blocked", "blocked", "blocked"],
        false,
        json!({"And":literals}),
    );
    for name in names {
        p.domain
            .predicates
            .push(serde_json::from_value(json!({"name":name,"variables":[]})).unwrap());
    }
    let result = solve_sequential_guided(
        &model(&p),
        &singletons(3),
        &MasterOptions {
            max_boundary_atoms: width,
            max_local_calls: 8,
            ..Default::default()
        },
        &MasterGuidance {
            conflict_core_checks: 4,
            ..Default::default()
        },
        ptu_plan::instrumentation::TraceLevel::Summary,
        None,
    )
    .unwrap();
    assert_eq!(result.status, MasterStatus::SequentiallyInfeasible);
    assert_eq!(result.statistics.core_assignments_skipped, usize::MAX);
    assert_eq!(result.statistics.local_calls, 6);
}
