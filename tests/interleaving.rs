use ptu_plan::{
    interleaving::*,
    master::{MasterOptions, MasterStatus, RootPartition, solve_sequential},
    parsing::json::Program,
    solver::{causal::CausalEndpoint, *},
};
use serde_json::{Value, json};
fn atom(name: &str) -> Value {
    json!({"Atom":{"name":name,"variables":[]}})
}
fn neg(name: &str) -> Value {
    json!({"Not":atom(name)})
}
fn action(name: &str, guard: Value, effect: Value) -> Value {
    json!({"name":name,"parameters":[],"preconditions":guard,"effects":effect})
}
fn method(name: &str, task: &str, guard: Value, children: &[&str]) -> Value {
    json!({"name":{"name":name},"task":{"name":task},"params":[],"task_terms":[],"precondition":guard,
    "tn":{"subtasks":children.iter().map(|c|json!({"task":{"name":c},"terms":[]})).collect::<Vec<_>>(),"orderings":"Total"}})
}
fn program(
    actions: Vec<Value>,
    methods: Vec<Value>,
    compounds: &[&str],
    roots: &[&str],
    order: Value,
    initial: &[&str],
    goal: Value,
) -> Program {
    serde_json::from_value(json!({"domain":{"name":"pocl-test","requirements":["Hierarchy"],"functions":[],
        "predicates":(["p","q","r"].map(|name|json!({"name":name,"variables":[]}))),
        "actions":actions,"methods":methods,"compound_tasks":compounds.iter().map(|name|json!({"name":name,"parameters":[]})).collect::<Vec<_>>()},
        "problem":{"name":"case","domain_name":"pocl-test","requirements":[],"objects":[],
            "init_state":initial.iter().map(|name|json!({"name":name,"variables":[]})).collect::<Vec<_>>(),"goal":goal,
            "init_tn":{"tn":{"subtasks":roots.iter().map(|name|json!({"task":{"name":name},"terms":[]})).collect::<Vec<_>>(),"orderings":order}}}})).unwrap()
}
fn unordered() -> Value {
    json!({"Partial":[]})
}
fn singletons(n: usize) -> Vec<RootPartition> {
    (0..n)
        .map(|i| RootPartition {
            id: i,
            roots: vec![i],
        })
        .collect()
}
fn ground(p: &Program) -> GroundProblem {
    ground_reachable_program(p, &GroundingOptions::default()).unwrap()
}
fn fixture() -> Program {
    serde_json::from_str(include_str!("fixtures/interleaving.json")).unwrap()
}
fn feasible(problem: &GroundProblem, partitions: &[RootPartition]) -> InterleavingCertificate {
    let result = solve_interleaved(problem, partitions, &InterleavingOptions::default()).unwrap();
    assert_eq!(
        result.status,
        ProofStatus::Feasible,
        "{:?}",
        result.statistics
    );
    assert_eq!(result.statistics.replay_checks, 1);
    let cert = result.certificate.unwrap();
    validate_interleaving_certificate(problem, &cert).unwrap();
    validate_problem_certificate(problem, &cert.plan).unwrap();
    cert
}
#[test]
fn cross_partition_support_recovers_the_non_contiguous_signal_exchange() {
    let problem = ground(&fixture());
    let partitions = singletons(2);
    assert_eq!(
        solve_sequential(&problem, &partitions, &MasterOptions::default())
            .unwrap()
            .status,
        MasterStatus::SequentiallyInfeasible
    );
    let cert = feasible(&problem, &partitions);
    assert_eq!(cert.plan.actions.len(), 4);
    let cross = cert
        .causal_links
        .iter()
        .filter(|link| match (&link.producer, &link.consumer) {
            (CausalEndpoint::Step(a), CausalEndpoint::Step(b)) => {
                cert.steps[*a].partition != cert.steps[*b].partition
            }
            _ => false,
        })
        .count();
    assert_eq!(cross, 2);
    let serialized = serde_json::to_vec(&cert).unwrap();
    let reloaded = serde_json::from_slice(&serialized).unwrap();
    validate_interleaving_certificate(&problem, &reloaded).unwrap();
}
#[test]
fn method_guards_can_import_facts_produced_after_the_global_start() {
    let p = program(
        vec![
            action("produce", Value::Null, atom("p")),
            action("finish", Value::Null, atom("q")),
        ],
        vec![method("guarded", "consumer", atom("p"), &["finish"])],
        &["consumer"],
        &["consumer", "produce"],
        unordered(),
        &[],
        atom("q"),
    );
    let problem = ground(&p);
    let cert = feasible(&problem, &singletons(2));
    assert_eq!(cert.plan.actions[0].name, "produce");
    assert!(cert.causal_links.iter().any(|link| match link.consumer {
        CausalEndpoint::Step(i) =>
            matches!(cert.steps[i].decision.kind, DecisionKind::Refine { .. })
                && link.literal.atom.predicate == "p",
        _ => false,
    }));
}
#[test]
fn opposite_requirements_within_one_partition_accept_intermediate_external_writes() {
    let p = program(
        vec![
            action("need-p", atom("p"), Value::Null),
            action("need-not-p", neg("p"), atom("q")),
            action("make-p", Value::Null, atom("p")),
            action("clear-p", Value::Null, neg("p")),
        ],
        vec![
            method("left", "consumer", Value::Null, &["need-p", "need-not-p"]),
            method("right", "producer", Value::Null, &["make-p", "clear-p"]),
        ],
        &["consumer", "producer"],
        &["consumer", "producer"],
        unordered(),
        &[],
        atom("q"),
    );
    let problem = ground(&p);
    let cert = feasible(&problem, &singletons(2));
    assert_eq!(
        cert.plan
            .actions
            .iter()
            .map(|a| a.name.as_str())
            .collect::<Vec<_>>(),
        ["make-p", "need-p", "clear-p", "need-not-p"]
    );
    assert!(
        cert.causal_links
            .iter()
            .any(|link| !link.literal.positive && matches!(link.producer, CausalEndpoint::Step(_)))
    );
}
#[test]
fn promotion_and_demotion_each_solve_forced_threat_cases() {
    let actions = vec![
        action("produce", Value::Null, atom("p")),
        action("use", atom("p"), atom("q")),
        action("clobber", Value::Null, neg("p")),
    ];
    for promote in [true, false] {
        let p = program(
            actions.clone(),
            vec![],
            &[],
            &["produce", "use", "clobber"],
            unordered(),
            &[],
            if promote {
                json!({"And":[atom("q"),neg("p")]})
            } else {
                atom("q")
            },
        );
        let mut problem = ground(&p);
        if !promote {
            problem.initial_networks[0]
                .network
                .precedence
                .insert((2, 1));
        }
        let result =
            solve_interleaved(&problem, &singletons(3), &InterleavingOptions::default()).unwrap();
        assert_eq!(result.status, ProofStatus::Feasible);
        assert!(result.statistics.threat_choices >= 2);
        assert!(result.statistics.cycle_rejections > 0);
        let cert = result.certificate.unwrap();
        validate_interleaving_certificate(&problem, &cert).unwrap();
        let names: Vec<_> = cert.plan.actions.iter().map(|a| a.name.as_str()).collect();
        assert_eq!(
            names,
            if promote {
                vec!["produce", "use", "clobber"]
            } else {
                vec!["clobber", "produce", "use"]
            }
        );
    }
}
#[test]
fn cyclic_partition_quotients_do_not_forbid_valid_root_interleavings() {
    let p = program(
        vec![
            action("make-p", Value::Null, atom("p")),
            action("use-p", atom("p"), atom("q")),
            action("clear-p", Value::Null, neg("p")),
            action("finish", neg("p"), atom("r")),
        ],
        vec![],
        &[],
        &["make-p", "use-p", "clear-p", "finish"],
        json!("Total"),
        &[],
        atom("r"),
    );
    let problem = ground(&p);
    let partitions = vec![
        RootPartition {
            id: 17,
            roots: vec![0, 2],
        },
        RootPartition {
            id: 81,
            roots: vec![1, 3],
        },
    ];
    assert_eq!(
        solve_sequential(&problem, &partitions, &MasterOptions::default())
            .unwrap()
            .status,
        MasterStatus::NotSequentiallyComposable
    );
    let cert = feasible(&problem, &partitions);
    assert_eq!(cert.plan.actions.len(), 4);
}
#[test]
fn complete_failures_are_global_but_depth_and_resource_cutoffs_remain_scoped() {
    let p = program(
        vec![
            action("p-from-q", atom("q"), atom("p")),
            action("q-from-p", atom("p"), atom("q")),
        ],
        vec![],
        &[],
        &["p-from-q", "q-from-p"],
        unordered(),
        &[],
        atom("p"),
    );
    let problem = ground(&p);
    let result =
        solve_interleaved(&problem, &singletons(2), &InterleavingOptions::default()).unwrap();
    assert_eq!(result.status, ProofStatus::Infeasible);
    assert!(result.statistics.cycle_rejections > 0);
    let recursive = program(
        vec![],
        vec![method("again", "loop", Value::Null, &["loop"])],
        &["loop"],
        &["loop"],
        unordered(),
        &[],
        Value::Null,
    );
    let recursive = ground(&recursive);
    let bounded = InterleavingOptions {
        exactness: ExactnessMode::Bounded {
            max_decomposition_depth: 2,
        },
        ..Default::default()
    };
    let result = solve_interleaved(&recursive, &singletons(1), &bounded).unwrap();
    assert_eq!(result.status, ProofStatus::InfeasibleWithinBound);
    assert_eq!(result.statistics.depth_cutoffs, 1);
    let capped = InterleavingOptions {
        max_events: 4,
        ..Default::default()
    };
    let result = solve_interleaved(&recursive, &singletons(1), &capped).unwrap();
    assert_eq!(result.status, ProofStatus::BudgetExhausted);
    assert!(result.statistics.resource_cutoffs > 0);
    let problem = ground(&fixture());
    for options in [
        InterleavingOptions {
            max_nodes: 0,
            ..Default::default()
        },
        InterleavingOptions {
            max_generated: 0,
            ..Default::default()
        },
        InterleavingOptions {
            max_time_ms: 0,
            ..Default::default()
        },
        InterleavingOptions {
            max_frontier: 0,
            ..Default::default()
        },
        InterleavingOptions {
            max_events: 1,
            ..Default::default()
        },
    ] {
        let result = solve_interleaved(&problem, &singletons(2), &options).unwrap();
        assert_eq!(result.status, ProofStatus::BudgetExhausted);
        assert!(result.certificate.is_none());
    }
}
#[test]
fn boolean_guards_and_orderings_agree_with_monolithic_search_on_128_models() {
    let guards = [
        Value::Null,
        atom("p"),
        neg("p"),
        json!({"Not":{"And":[atom("p"),neg("q")]}}),
    ];
    for a in &guards {
        for b in &guards {
            for bits in 0..4 {
                for ordered in [false, true] {
                    let initial: Vec<_> = ["p", "q"]
                        .into_iter()
                        .enumerate()
                        .filter_map(|(i, p)| (bits & (1 << i) != 0).then_some(p))
                        .collect();
                    let p = program(
                        vec![
                            action("a", a.clone(), atom("p")),
                            action("b", b.clone(), atom("q")),
                            action("c", Value::Null, neg("p")),
                        ],
                        vec![],
                        &[],
                        &["a", "b", "c"],
                        if ordered { json!("Total") } else { unordered() },
                        &initial,
                        json!({"And":[atom("q"),neg("p")]}),
                    );
                    let problem = ground(&p);
                    let expected = solve_problem(&problem, &SearchBudget::default()).unwrap();
                    let actual = solve_interleaved(
                        &problem,
                        &singletons(3),
                        &InterleavingOptions::default(),
                    )
                    .unwrap();
                    assert_eq!(
                        actual.status, expected.status,
                        "guards {a} {b}, bits {bits}, ordered {ordered}"
                    );
                    if let Some(cert) = actual.certificate {
                        validate_interleaving_certificate(&problem, &cert).unwrap();
                    }
                }
            }
        }
    }
}
#[test]
fn certificates_reject_changed_support_hierarchy_ownership_order_and_replay() {
    let problem = ground(&fixture());
    let cert = feasible(&problem, &singletons(2));
    let mut bad = cert.clone();
    bad.causal_links.clear();
    assert!(validate_interleaving_certificate(&problem, &bad).is_err());
    let mut bad = cert.clone();
    bad.steps[0].partition = 999;
    assert!(validate_interleaving_certificate(&problem, &bad).is_err());
    let mut bad = cert.clone();
    bad.steps[0].children.clear();
    assert!(validate_interleaving_certificate(&problem, &bad).is_err());
    let mut bad = cert.clone();
    bad.ordering.clear();
    assert!(validate_interleaving_certificate(&problem, &bad).is_err());
    let mut bad = cert.clone();
    bad.plan.actions.clear();
    assert!(validate_interleaving_certificate(&problem, &bad).is_err());
    let mut bad = cert.clone();
    bad.linearization.fill(0);
    assert!(validate_interleaving_certificate(&problem, &bad).is_err());
    let mut bad = cert.clone();
    bad.exactness = ExactnessMode::Bounded {
        max_decomposition_depth: 0,
    };
    assert!(validate_interleaving_certificate(&problem, &bad).is_err());
    let mut bad = cert.clone();
    bad.partitions[1].roots[0] = 0;
    assert!(validate_interleaving_certificate(&problem, &bad).is_err());
}
#[test]
fn global_goal_empty_refinements_and_add_wins_follow_exact_semantics() {
    let p = program(
        vec![action(
            "both",
            Value::Null,
            json!({"And":[atom("p"),neg("p")]}),
        )],
        vec![method("empty", "empty", Value::Null, &[])],
        &["empty"],
        &["empty", "both"],
        json!("Total"),
        &[],
        atom("p"),
    );
    let problem = ground(&p);
    let cert = feasible(&problem, &singletons(2));
    assert_eq!(cert.plan.actions.len(), 1);
    let mut problem = problem;
    problem.goal = Condition::Not(Box::new(Condition::Atom(GroundAtom {
        predicate: "p".into(),
        arguments: vec![],
    })));
    assert_eq!(
        solve_interleaved(&problem, &singletons(2), &InterleavingOptions::default())
            .unwrap()
            .status,
        ProofStatus::Infeasible
    );
    let empty = program(vec![], vec![], &[], &[], unordered(), &[], Value::Null);
    let problem = ground(&empty);
    let cert = feasible(&problem, &[]);
    assert!(cert.plan.actions.is_empty());
}
#[test]
fn interleaving_cli_returns_a_replayable_certificate_and_budget_status() {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_ptu-plan"))
        .args([
            "tests/fixtures/interleaving.json",
            "--interleave",
            "--micro-max-size",
            "1",
            "--analysis-json",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: Value = serde_json::from_slice(&output.stdout).unwrap();
    let result: InterleavingResult = serde_json::from_value(value["interleaving"].clone()).unwrap();
    assert_eq!(result.status, ProofStatus::Feasible);
    validate_interleaving_certificate(&ground(&fixture()), result.certificate.as_ref().unwrap())
        .unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_ptu-plan"))
        .args([
            "tests/fixtures/interleaving.json",
            "--interleave",
            "--micro-max-size",
            "1",
            "--max-pocl-events",
            "1",
            "--analysis-json",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let value: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(value["interleaving"]["status"], "BudgetExhausted");
}

fn source(
    problem: &GroundProblem,
    partition: RootPartition,
    initial: State,
) -> ptu_plan::master::LocalSegment {
    let request = LocalRequest {
        network: problem.initial_networks[0]
            .network
            .induced(&partition.roots)
            .unwrap(),
        initial_state: initial.clone(),
        goal: Condition::True,
    };
    let certificate = solve_local(problem, &request, &SearchBudget::default())
        .unwrap()
        .certificate
        .unwrap();
    let causal = causal::causal_contract(problem, &request, &certificate, partition.id).unwrap();
    ptu_plan::master::LocalSegment {
        partition,
        incoming: BoundaryState {
            must_be_true: initial,
            must_be_false: State::new(),
        },
        outgoing: BoundaryState::default(),
        request,
        certificate,
        causal: Some(causal),
    }
}
#[test]
fn supplied_local_certificates_reconcile_with_event_local_imports_and_auditable_evidence() {
    let problem = ground(&fixture());
    let partitions = singletons(2);
    let sources: Vec<_> = partitions
        .into_iter()
        .zip(["q", "p"])
        .map(|(partition, p)| {
            source(
                &problem,
                partition,
                [GroundAtom {
                    predicate: p.into(),
                    arguments: vec![],
                }]
                .into_iter()
                .collect(),
            )
        })
        .collect();
    let result =
        reconcile_local_certificates(&problem, 0, &sources, &InterleavingOptions::default())
            .unwrap();
    assert_eq!(result.scope, SearchScope::SuppliedLocalCertificates);
    assert_eq!(result.status, ProofStatus::Feasible);
    let cert = result.certificate.unwrap();
    assert_eq!(cert.local_sources.as_ref().unwrap().len(), 2);
    assert_eq!(cert.plan.actions.len(), 4);
    validate_interleaving_certificate(&problem, &cert).unwrap();
    let mut bad = cert.clone();
    bad.local_sources.as_mut().unwrap()[0]
        .causal
        .as_mut()
        .unwrap()
        .causal_links
        .clear();
    assert!(validate_interleaving_certificate(&problem, &bad).is_err());
    let mut legacy = sources.clone();
    for source in &mut legacy {
        source.causal = None;
    }
    let result =
        reconcile_local_certificates(&problem, 0, &legacy, &InterleavingOptions::default())
            .unwrap();
    assert_eq!(result.status, ProofStatus::Feasible);
}
#[test]
fn failed_supplied_certificates_do_not_imply_failure_of_other_decompositions() {
    let p = program(
        vec![
            action("make-p", Value::Null, atom("p")),
            action("need-p", atom("p"), atom("q")),
        ],
        vec![
            method("skip", "choice", Value::Null, &[]),
            method("produce", "choice", Value::Null, &["make-p"]),
        ],
        &["choice"],
        &["choice", "need-p"],
        unordered(),
        &[],
        atom("q"),
    );
    let problem = ground(&p);
    let partitions = singletons(2);
    let sources = vec![
        source(&problem, partitions[0].clone(), State::new()),
        source(
            &problem,
            partitions[1].clone(),
            [GroundAtom {
                predicate: "p".into(),
                arguments: vec![],
            }]
            .into_iter()
            .collect(),
        ),
    ];
    assert!(sources[0].certificate.actions.is_empty());
    let result =
        reconcile_local_certificates(&problem, 0, &sources, &InterleavingOptions::default())
            .unwrap();
    assert_eq!(result.status, ProofStatus::Infeasible);
    assert_eq!(result.scope, SearchScope::SuppliedLocalCertificates);
    let result = solve_interleaved(&problem, &partitions, &InterleavingOptions::default()).unwrap();
    assert_eq!(result.status, ProofStatus::Feasible);
    assert_eq!(result.scope, SearchScope::AllDecompositions);
    let bounded = InterleavingOptions {
        exactness: ExactnessMode::Bounded {
            max_decomposition_depth: 0,
        },
        ..Default::default()
    };
    assert!(reconcile_local_certificates(&problem, 0, &sources, &bounded).is_err());
}
#[test]
fn a_dropped_alternative_is_unknown_and_poor_early_methods_can_be_revised() {
    let p = program(
        vec![action("produce", Value::Null, atom("p"))],
        vec![
            method("blocked", "choice", atom("q"), &[]),
            method("produce", "choice", Value::Null, &["produce"]),
        ],
        &["choice"],
        &["choice"],
        unordered(),
        &[],
        atom("p"),
    );
    let problem = ground(&p);
    feasible(&problem, &singletons(1));
    let capped = InterleavingOptions {
        max_frontier: 1,
        ..Default::default()
    };
    let mut p = p;
    p.domain.methods.reverse();
    let problem = ground(&p);
    let result = solve_interleaved(&problem, &singletons(1), &capped).unwrap();
    assert_eq!(result.status, ProofStatus::BudgetExhausted);
    assert!(result.statistics.resource_cutoffs > 0);
}
