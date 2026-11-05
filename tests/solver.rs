use std::collections::BTreeSet;
use std::process::Command;

use ptu_plan::parsing::json::{
    Constraint, Formula, Function, NumberType, Predicate, Program, RequirementType, Subtask,
    Symbol, TaskOrdering,
};
use ptu_plan::solver::*;

fn fixture() -> Program {
    serde_json::from_str(include_str!("fixtures/solver.json")).unwrap()
}
fn sym(name: &str) -> Symbol {
    Symbol {
        name: name.into(),
        symbol_type: None,
    }
}
fn typed(name: &str, ty: &str) -> Symbol {
    Symbol {
        name: name.into(),
        symbol_type: Some(ty.into()),
    }
}
fn atom(name: &str, args: &[&str]) -> GroundAtom {
    GroundAtom {
        predicate: name.into(),
        arguments: args.iter().map(|s| (*s).into()).collect(),
    }
}
fn formula(name: &str, args: &[&str]) -> Formula {
    Formula::Atom(Predicate {
        name: name.into(),
        variables: args.iter().map(|s| sym(s)).collect(),
    })
}
fn model(program: &Program) -> GroundProblem {
    ground_program(program, &GroundingOptions::default()).unwrap()
}
fn solve(program: &Program) -> SolveResult {
    solve_problem(&model(program), &SearchBudget::default()).unwrap()
}

#[test]
fn solves_typed_negative_preconditions_and_quantified_goals_with_replay() {
    let model = model(&fixture());
    assert_eq!(model.actions.len(), 2);
    let result = solve_problem(&model, &SearchBudget::default()).unwrap();
    assert_eq!(result.status, ProofStatus::Feasible);
    let certificate = result.certificate.unwrap();
    assert_eq!(certificate.actions.len(), 2);
    assert!(certificate.final_state.contains(&atom("done", &["a"])));
    assert!(certificate.final_state.contains(&atom("done", &["b"])));
    assert!(!certificate.final_state.contains(&atom("ready", &["a"])));
    validate_problem_certificate(&model, &certificate).unwrap();
    assert_eq!(
        certificate
            .decisions
            .iter()
            .filter(|d| matches!(d.kind, DecisionKind::Refine { .. }))
            .count(),
        2
    );
}

#[test]
fn exhaustive_failure_and_budget_exhaustion_are_distinct() {
    let mut p = fixture();
    p.problem.as_mut().unwrap().init_state.clear();
    assert_eq!(solve(&p).status, ProofStatus::Infeasible);
    let model = model(&fixture());
    for budget in [
        SearchBudget {
            max_expansions: 0,
            ..SearchBudget::default()
        },
        SearchBudget {
            max_generated: 0,
            ..SearchBudget::default()
        },
        SearchBudget {
            max_time_ms: 0,
            ..SearchBudget::default()
        },
    ] {
        let result = solve_problem(&model, &budget).unwrap();
        assert_eq!(result.status, ProofStatus::BudgetExhausted);
        assert!(result.certificate.is_none());
    }
    assert!(
        ground_program(&fixture(), &GroundingOptions { max_instances: 1 })
            .unwrap_err()
            .to_string()
            .contains("budget exhausted")
    );
}

#[test]
fn decomposition_bounds_never_become_global_infeasibility_proofs() {
    let model = model(&fixture());
    let result = solve_problem(
        &model,
        &SearchBudget {
            exactness: ExactnessMode::Bounded {
                max_decomposition_depth: 0,
            },
            ..SearchBudget::default()
        },
    )
    .unwrap();
    assert_eq!(result.status, ProofStatus::InfeasibleWithinBound);
    let result = solve_problem(
        &model,
        &SearchBudget {
            exactness: ExactnessMode::Bounded {
                max_decomposition_depth: 1,
            },
            ..SearchBudget::default()
        },
    )
    .unwrap();
    assert_eq!(result.status, ProofStatus::Feasible);
}

#[test]
fn method_head_constants_and_equality_constraints_filter_bindings() {
    let mut p = fixture();
    p.domain.methods.remove(0);
    p.domain.methods[0].tn.constraints = Some(vec![Constraint::Equal("?x".into(), "a".into())]);
    assert_eq!(solve(&p).status, ProofStatus::Infeasible);
    p.problem
        .as_mut()
        .unwrap()
        .init_tn
        .as_mut()
        .unwrap()
        .tn
        .subtasks
        .truncate(1);
    p.problem.as_mut().unwrap().goal = Some(formula("done", &["a"]));
    assert_eq!(solve(&p).status, ProofStatus::Feasible);
    p.domain.methods[0].task_terms = vec![sym("a")];
    p.domain.methods[0].params.clear();
    p.domain.methods[0].tn.constraints = None;
    p.domain.methods[0].precondition = None;
    p.domain.methods[0].tn.subtasks[0].terms = vec![sym("a")];
    assert_eq!(solve(&p).status, ProofStatus::Feasible);
}

#[test]
fn initial_variables_are_existential_and_empty_typed_domains_are_vacuous() {
    let mut p = fixture();
    let initial = p.problem.as_mut().unwrap().init_tn.as_mut().unwrap();
    initial.parameters = Some(vec![typed("?root", "item")]);
    initial.tn.subtasks.truncate(1);
    initial.tn.subtasks[0].terms = vec![sym("?root")];
    p.problem.as_mut().unwrap().goal = Some(formula("done", &["b"]));
    let m = model(&p);
    assert_eq!(m.initial_networks.len(), 2);
    let certificate = solve_problem(&m, &SearchBudget::default())
        .unwrap()
        .certificate
        .unwrap();
    assert_eq!(
        m.initial_networks[certificate.initial_network].bindings["?root"],
        "b"
    );
    validate_problem_certificate(&m, &certificate).unwrap();

    p.problem.as_mut().unwrap().init_tn = None;
    p.problem.as_mut().unwrap().goal = Some(Formula::ForAll(
        vec![typed("?x", "empty")],
        Box::new(formula("done", &["?x"])),
    ));
    assert_eq!(solve(&p).status, ProofStatus::Feasible);
    p.problem.as_mut().unwrap().goal = Some(Formula::Exists(
        vec![typed("?x", "empty")],
        Box::new(formula("done", &["?x"])),
    ));
    assert_eq!(solve(&p).status, ProofStatus::Infeasible);
}

#[test]
fn quantifier_shadowing_and_simultaneous_add_delete_have_defined_semantics() {
    let mut p = fixture();
    p.domain.actions[0].effects = Some(Formula::And(vec![
        formula("done", &["?x"]),
        Formula::Not(Box::new(formula("done", &["?x"]))),
        Formula::ForAll(
            vec![typed("?x", "item")],
            Box::new(formula("done", &["?x"])),
        ),
    ]));
    p.domain.actions[0].preconditions = Some(Formula::And(vec![
        Formula::Imply(
            Box::new(Formula::Equals("?x".into(), "a".into())),
            Box::new(formula("ready", &["?x"])),
        ),
        Formula::Exists(
            vec![typed("?x", "item")],
            Box::new(Formula::Or(vec![formula("ready", &["?x"]), Formula::Empty])),
        ),
    ]));
    let result = solve(&p);
    assert_eq!(result.status, ProofStatus::Feasible);
    assert!(
        result
            .certificate
            .unwrap()
            .final_state
            .contains(&atom("done", &["a"]))
    );
}

#[test]
fn unsupported_semantics_and_invalid_bindings_fail_before_search() {
    type Mutation = fn(&mut Program);
    let cases: &[(&str, Mutation)] = &[
        ("unsupported", |p| {
            p.domain.actions[0].effects = Some(Formula::Probabilistic(
                NumberType::Real(0.5),
                Box::new(Formula::Empty),
            ))
        }),
        ("unsupported", |p| {
            p.domain.actions[0].effects = Some(Formula::Or(vec![]))
        }),
        ("unsupported", |p| {
            p.domain.actions[0].preconditions = Some(Formula::Xor(vec![]))
        }),
        ("numeric", |p| {
            p.domain.functions.push(Function {
                name: "cost".into(),
                variables: vec![],
            })
        }),
        ("unsupported solver requirement", |p| {
            p.domain.requirements.push(RequirementType::DomainAxioms)
        }),
        ("unknown type", |p| {
            p.problem.as_mut().unwrap().objects[0].symbol_type = Some("absent".into())
        }),
        ("cyclic type", |p| {
            p.domain.types.as_mut().unwrap()[0].symbol_type = Some("special".into())
        }),
        ("unbound variable", |p| {
            p.domain.actions[0].preconditions = Some(formula("ready", &["?missing"]))
        }),
        ("unknown object", |p| {
            p.problem.as_mut().unwrap().init_state[0].variables[0].name = "missing".into()
        }),
        ("unknown predicate", |p| {
            p.domain.actions[0].effects = Some(formula("missing", &["?x"]))
        }),
        ("arity mismatch", |p| {
            p.domain.actions[0].effects = Some(formula("done", &[]))
        }),
        ("required type", |p| {
            p.domain.predicates[0].variables[0].symbol_type = Some("special".into())
        }),
    ];
    for (expected, mutate) in cases {
        let mut p = fixture();
        mutate(&mut p);
        let error = ground_program(&p, &GroundingOptions::default())
            .unwrap_err()
            .to_string();
        assert!(error.contains(expected), "expected {expected}, got {error}");
    }
}

#[test]
fn local_partition_uses_concrete_input_and_sparse_terminal_boundary() {
    let m = model(&fixture());
    let boundary = BoundaryState {
        must_be_true: BTreeSet::from([atom("done", &["a"])]),
        must_be_false: BTreeSet::from([atom("ready", &["a"])]),
    };
    let request = LocalRequest {
        network: m.initial_networks[0].network.induced(&[0]).unwrap(),
        initial_state: m.initial_state.clone(),
        goal: boundary.condition().unwrap(),
    };
    let result = solve_local(&m, &request, &SearchBudget::default()).unwrap();
    assert_eq!(result.status, ProofStatus::Feasible);
    let certificate = result.certificate.unwrap();
    assert!(certificate.final_state.contains(&atom("ready", &["b"])));
    validate_certificate(&m, &request, &certificate).unwrap();
    let contradictory = BoundaryState {
        must_be_false: boundary.must_be_true.clone(),
        ..boundary
    };
    assert!(contradictory.condition().is_err());
    let mut invalid = request;
    invalid.initial_state.insert(atom("missing", &[]));
    assert!(solve_local(&m, &invalid, &SearchBudget::default()).is_err());
}

#[test]
fn induced_network_keeps_precedence_paths_through_other_partitions() {
    let task = GroundTask {
        name: "x".into(),
        arguments: vec![],
    };
    let network = GroundNetwork {
        tasks: vec![task.clone(), task.clone(), task],
        precedence: BTreeSet::from([(0, 1), (1, 2)]),
    };
    let local = network.induced(&[0, 2]).unwrap();
    assert_eq!(local.precedence, BTreeSet::from([(0, 1)]));
    assert!(network.induced(&[0, 0]).is_err());
    assert!(network.induced(&[3]).is_err());
    let cyclic = GroundNetwork {
        precedence: BTreeSet::from([(0, 1), (1, 0)]),
        ..network
    };
    assert!(cyclic.induced(&[0]).is_err());
}

#[test]
fn certificate_tampering_is_rejected() {
    let m = model(&fixture());
    let certificate = solve_problem(&m, &SearchBudget::default())
        .unwrap()
        .certificate
        .unwrap();
    let mut wrong = certificate.clone();
    wrong.final_state.clear();
    assert!(validate_problem_certificate(&m, &wrong).is_err());
    let mut wrong = certificate.clone();
    wrong.actions.pop();
    assert!(validate_problem_certificate(&m, &wrong).is_err());
    let mut wrong = certificate.clone();
    wrong.decisions[0].occurrence = 999;
    assert!(validate_problem_certificate(&m, &wrong).is_err());
    let mut wrong = certificate.clone();
    wrong.decisions[0].kind = DecisionKind::Refine { method: usize::MAX };
    assert!(validate_problem_certificate(&m, &wrong).is_err());
    let mut wrong = certificate;
    wrong.decisions.clear();
    assert!(validate_problem_certificate(&m, &wrong).is_err());
}

#[test]
fn recursive_methods_terminate_at_a_bound_and_retain_base_alternatives() {
    let mut p = fixture();
    let mut recursive = p.domain.methods[1].clone();
    recursive.name = sym("repeat");
    recursive.tn.subtasks[0].task = sym("finish");
    p.domain.methods = vec![recursive.clone()];
    let result = solve_problem(
        &model(&p),
        &SearchBudget {
            exactness: ExactnessMode::Bounded {
                max_decomposition_depth: 3,
            },
            ..SearchBudget::default()
        },
    )
    .unwrap();
    assert_eq!(result.status, ProofStatus::InfeasibleWithinBound);
    // Unbounded search can prove this no-progress recursive loop infeasible by
    // merging identical residual symbolic configurations.
    assert_eq!(solve(&p).status, ProofStatus::Infeasible);
    p.domain.methods.push(fixture().domain.methods[1].clone());
    assert_eq!(solve(&p).status, ProofStatus::Feasible);
}

#[test]
fn empty_methods_preserve_successor_order_and_goal_checks() {
    let mut p = fixture();
    p.domain.methods.remove(0);
    p.domain.methods[0].tn.subtasks.clear();
    p.problem
        .as_mut()
        .unwrap()
        .init_tn
        .as_mut()
        .unwrap()
        .tn
        .orderings = TaskOrdering::Total;
    p.problem.as_mut().unwrap().goal = None;
    let result = solve(&p);
    assert_eq!(result.status, ProofStatus::Feasible);
    assert!(result.certificate.unwrap().actions.is_empty());
    p.problem.as_mut().unwrap().goal = Some(formula("done", &["a"]));
    assert_eq!(solve(&p).status, ProofStatus::Infeasible);
}

#[test]
fn false_method_preconditions_and_unscheduled_actions_cannot_satisfy_a_request() {
    let mut p = fixture();
    // The empty method is guarded by locked(?x), which is absent. Even a true
    // terminal goal cannot justify applying this decomposition.
    p.domain.methods.truncate(1);
    p.problem.as_mut().unwrap().goal = None;
    assert_eq!(solve(&p).status, ProofStatus::Infeasible);

    p.problem.as_mut().unwrap().init_tn = None;
    p.problem.as_mut().unwrap().goal = Some(formula("done", &["a"]));
    assert_eq!(solve(&p).status, ProofStatus::Infeasible);

    let mut p = fixture();
    p.problem
        .as_mut()
        .unwrap()
        .init_tn
        .as_mut()
        .unwrap()
        .tn
        .constraints = Some(vec![Constraint::Equal("a".into(), "b".into())]);
    let m = model(&p);
    assert!(m.initial_networks.is_empty());
    assert_eq!(
        solve_problem(&m, &SearchBudget::default()).unwrap().status,
        ProofStatus::Infeasible
    );
}

#[test]
fn cli_solve_and_partition_emit_results_in_one_json_record() {
    let path = format!("{}/tests/fixtures/solver.json", env!("CARGO_MANIFEST_DIR"));
    let output = Command::new(env!("CARGO_BIN_EXE_ptu-plan"))
        .args([
            &path,
            "--solve",
            "--micropartition",
            "--micro-max-size",
            "1",
            "--view-weights",
            "1,2,3,1,1,2",
            "--analysis-json",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let json: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(json["solution"]["status"], "Feasible");
    assert_eq!(
        json["micropartitions"]["clusters"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    let output = Command::new(env!("CARGO_BIN_EXE_ptu-plan"))
        .args([&path, "--solve", "--max-expansions", "0", "--analysis-json"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let json: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(json["solution"]["status"], "BudgetExhausted");
    let output = Command::new(env!("CARGO_BIN_EXE_ptu-plan"))
        .args([&path, "--micropartition", "--view-weights", "1,2"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("six comma-separated values"));
}

#[test]
fn solver_interleaves_descendants_of_unordered_compound_roots() {
    let mut p: Program = serde_json::from_value(serde_json::json!({
        "domain": { "name":"interleave", "requirements":["Hierarchy"],
            "predicates": [ {"name":"p","variables":[]}, {"name":"q","variables":[]}, {"name":"left-done","variables":[]}, {"name":"right-done","variables":[]} ],
            "compound_tasks": [{"name":"left","parameters":[]},{"name":"right","parameters":[]}],
            "methods": [], "actions": [
                {"name":"lp","parameters":[],"effects":{"Atom":{"name":"p","variables":[]}}},
                {"name":"rp","parameters":[],"effects":{"Atom":{"name":"q","variables":[]}}},
                {"name":"lc","parameters":[],"preconditions":{"Atom":{"name":"q","variables":[]}},"effects":{"Atom":{"name":"left-done","variables":[]}}},
                {"name":"rc","parameters":[],"preconditions":{"Atom":{"name":"p","variables":[]}},"effects":{"Atom":{"name":"right-done","variables":[]}}}
            ], "functions":[] },
        "problem": {"name":"p","domain_name":"interleave","requirements":[],"objects":[],"init_state":[],
            "init_tn":{"tn":{"subtasks":[{"task":{"name":"left"},"terms":[]},{"task":{"name":"right"},"terms":[]}],"orderings":{"Partial":[]}}},
            "goal":{"And":[{"Atom":{"name":"left-done","variables":[]}},{"Atom":{"name":"right-done","variables":[]}}]}}
    })).unwrap();
    for (name, a, b) in [("left", "lp", "lc"), ("right", "rp", "rc")] {
        p.domain.methods.push(ptu_plan::parsing::json::Method {
            name: sym(name),
            task: sym(name),
            task_terms: vec![],
            params: vec![],
            precondition: None,
            tn: ptu_plan::parsing::json::Htn {
                subtasks: [a, b]
                    .iter()
                    .map(|name| Subtask {
                        id: None,
                        task: sym(name),
                        terms: vec![],
                    })
                    .collect(),
                orderings: TaskOrdering::Total,
                constraints: None,
            },
        });
    }
    let m = model(&p);
    let certificate = solve_problem(&m, &SearchBudget::default())
        .unwrap()
        .certificate
        .unwrap();
    let pos = |name| {
        certificate
            .actions
            .iter()
            .position(|task| task.name == name)
            .unwrap()
    };
    assert!(pos("lp") < pos("lc") && pos("rp") < pos("rc"));
    assert!(pos("rp") < pos("lc") && pos("lp") < pos("rc"));
    validate_problem_certificate(&m, &certificate).unwrap();
    p.problem
        .as_mut()
        .unwrap()
        .init_tn
        .as_mut()
        .unwrap()
        .tn
        .orderings = TaskOrdering::Total;
    assert_eq!(solve(&p).status, ProofStatus::Infeasible);
}

#[test]
fn reachable_grounding_avoids_irrelevant_cartesian_products_and_keeps_false_guards() {
    let mut p = fixture();
    p.domain.compound_tasks.push(ptu_plan::parsing::json::Task {
        name: "unused".into(),
        parameters: (0..8).map(|i| typed(&format!("?v{i}"), "item")).collect(),
    });
    let options = GroundingOptions { max_instances: 150 };
    assert!(
        ground_program(&p, &options)
            .unwrap_err()
            .0
            .contains("grounding budget exhausted")
    );
    let reachable = ground_reachable_program(&p, &options).unwrap();
    assert_eq!(reachable.grounding_mode, GroundingMode::InitialReachable);
    assert_eq!(reachable.actions.len(), 2);
    assert_eq!(reachable.methods.len(), 4); // includes initially false locked guards
    let cert = solve_problem(&reachable, &SearchBudget::default())
        .unwrap()
        .certificate
        .unwrap();
    validate_problem_certificate(&reachable, &cert).unwrap();
}

#[test]
fn reachable_grounding_unifies_repeated_head_variables_and_constants() {
    let mut p = fixture();
    p.domain.compound_tasks.push(ptu_plan::parsing::json::Task {
        name: "pair".into(),
        parameters: vec![typed("?left", "item"), typed("?right", "item")],
    });
    let mut m = p.domain.methods[1].clone();
    m.name = sym("same");
    m.task = sym("pair");
    m.task_terms = vec![sym("?x"), sym("?x")];
    p.domain.methods.push(m.clone());
    let problem = p.problem.as_mut().unwrap();
    problem.goal = Some(formula("done", &["a"]));
    let tn = &mut problem.init_tn.as_mut().unwrap().tn;
    tn.constraints = None;
    tn.subtasks.truncate(1);
    tn.subtasks[0].task = sym("pair");
    tn.subtasks[0].terms = vec![sym("a"), sym("b")];
    for same in [false, true] {
        p.problem
            .as_mut()
            .unwrap()
            .init_tn
            .as_mut()
            .unwrap()
            .tn
            .subtasks[0]
            .terms[1] = sym(if same { "a" } else { "b" });
        let reachable = ground_reachable_program(&p, &GroundingOptions::default()).unwrap();
        assert_eq!(
            solve_problem(&reachable, &SearchBudget::default())
                .unwrap()
                .status,
            if same {
                ProofStatus::Feasible
            } else {
                ProofStatus::Infeasible
            }
        );
    }
    m.name = sym("constant");
    m.task_terms = vec![sym("?x"), sym("a")];
    p.domain.methods.push(m);
    let reachable = ground_reachable_program(&p, &GroundingOptions::default()).unwrap();
    assert!(reachable.methods.iter().any(|m| m.name == "constant"));
    p.problem
        .as_mut()
        .unwrap()
        .init_tn
        .as_mut()
        .unwrap()
        .tn
        .subtasks[0]
        .terms = vec![sym("b"), sym("a")];
    p.problem.as_mut().unwrap().goal = Some(formula("done", &["b"]));
    let reachable = ground_reachable_program(&p, &GroundingOptions::default()).unwrap();
    let cert = solve_problem(&reachable, &SearchBudget::default())
        .unwrap()
        .certificate
        .unwrap();
    validate_problem_certificate(&reachable, &cert).unwrap();
}

#[test]
fn multiple_inheritance_accepts_diamonds_and_rejects_cycles() {
    let mut p = fixture();
    p.domain
        .types
        .as_mut()
        .unwrap()
        .extend([typed("special", "auxiliary"), sym("auxiliary")]);
    p.domain.actions.push(ptu_plan::parsing::json::Action {
        name: "aux-step".into(),
        parameters: vec![typed("?x", "auxiliary")],
        preconditions: None,
        effects: None,
    });
    let model = ground_program(&p, &GroundingOptions::default()).unwrap();
    let tasks: Vec<_> = model
        .actions
        .keys()
        .filter(|t| t.name == "aux-step")
        .collect();
    assert_eq!(tasks.len(), 1);
    assert_eq!(tasks[0].arguments, ["a"]);
    p.problem.as_mut().unwrap().domain_name = "FINISH-ITEMS".into();
    assert!(ground_reachable_program(&p, &GroundingOptions::default()).is_ok());
    p.domain
        .types
        .as_mut()
        .unwrap()
        .push(typed("auxiliary", "special"));
    assert!(
        ground_reachable_program(&p, &GroundingOptions::default())
            .unwrap_err()
            .0
            .contains("cyclic type hierarchy")
    );
}
