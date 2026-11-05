use std::process::Command;

use ptu_plan::analysis::{
    FormulaFeature, HtnAnalysis, Literal, MethodId, NetworkId, OrderingClass, RecursionClass,
    TaskId, analyze_program,
};
use ptu_plan::interaction::{
    EdgeKind, GraphView, HierarchyRelation, InteractionGraph, NodeId, NodeKind, ThreatTarget,
    ViewWeights,
};
use ptu_plan::parsing::json::{Formula, Predicate, Program, Subtask, Symbol, Task, TaskOrdering};

fn fixture() -> Program {
    serde_json::from_str(include_str!("fixtures/interaction.json")).unwrap()
}

fn symbol(name: &str) -> Symbol {
    Symbol {
        name: name.into(),
        symbol_type: None,
    }
}

fn literal(predicate: &str, positive: bool) -> Literal {
    Literal {
        predicate: predicate.into(),
        positive,
    }
}

fn atom(name: &str) -> Formula {
    Formula::Atom(Predicate {
        name: name.into(),
        variables: Vec::new(),
    })
}

fn task(analysis: &HtnAnalysis, name: &str) -> TaskId {
    analysis
        .tasks
        .iter()
        .find(|task| task.name == name)
        .unwrap()
        .id
}

fn recursive_program(ordering: TaskOrdering) -> Program {
    let mut program = fixture();
    let mut recursive = program.domain.methods[0].clone();
    recursive.name = symbol("repeat");
    recursive.precondition = None;
    recursive.tn.subtasks = vec![
        Subtask {
            id: Some(symbol("step")),
            task: symbol("make"),
            terms: vec![],
        },
        Subtask {
            id: Some(symbol("again")),
            task: symbol("job"),
            terms: vec![],
        },
    ];
    recursive.tn.orderings = ordering;
    recursive.tn.constraints = None;
    program.domain.methods.push(recursive);
    program
}

#[test]
fn resolves_networks_without_inventing_list_order() {
    let analysis = analyze_program(&fixture()).unwrap();
    let network = &analysis.methods[0].network;
    assert_eq!(network.id, NetworkId::Method(MethodId(0)));
    assert_eq!(network.ordering, OrderingClass::Partial);
    assert!(network.precedes(0, 1));
    assert!(!network.precedes(1, 2));
    assert!(!network.precedes(0, 2));
    assert_eq!(network.constraints.len(), 1);
    assert_eq!(
        analysis.initial_network.as_ref().unwrap().ordering,
        OrderingClass::Unordered
    );
    assert_eq!(analysis.signature.ordering, OrderingClass::Partial);
    assert!(!analysis.signature.fully_unordered);
    assert!(!analysis.signature.has_lifted_variables);
    assert_eq!(analysis.signature.recursion, RecursionClass::Acyclic);
    assert_eq!(analysis.signature.reachable_tasks, 4);
    assert_eq!(analysis.signature.ordered_pairs, 1);
    assert_eq!(analysis.signature.possible_ordered_pairs, 4);
}

#[test]
fn recognizes_total_orders_through_transitive_closure() {
    let mut program = fixture();
    program.domain.methods[0].tn.orderings = TaskOrdering::Partial(vec![
        ("a".into(), "b".into()),
        ("b".into(), "c".into()),
        ("a".into(), "b".into()),
    ]);
    let analysis = analyze_program(&program).unwrap();
    let network = &analysis.methods[0].network;
    assert_eq!(network.ordering, OrderingClass::Total);
    assert_eq!(network.precedence.len(), 2);
    assert_eq!(network.transitive_precedence.len(), 3);
    assert!(network.precedes(0, 2));
    program.domain.methods[0].tn.orderings = TaskOrdering::Total;
    let total = analyze_program(&program).unwrap();
    assert_eq!(
        network.transitive_precedence,
        total.methods[0].network.transitive_precedence
    );
}

#[test]
fn rejects_invalid_structural_references_with_context() {
    type Mutation = fn(&mut Program);
    let cases: &[(&str, Mutation)] = &[
        ("unknown task", |p| {
            p.domain.methods[0].tn.subtasks[0].task = symbol("missing")
        }),
        ("expects 0 arguments", |p| {
            p.domain.methods[0].tn.subtasks[0].terms.push(symbol("x"))
        }),
        ("head expects", |p| {
            p.domain.methods[0].task_terms.push(symbol("x"))
        }),
        ("refines primitive", |p| {
            p.domain.methods[0].task = symbol("make")
        }),
        ("refines unknown", |p| {
            p.domain.methods[0].task = symbol("missing")
        }),
        ("duplicate task", |p| {
            p.domain.compound_tasks[0].name = "make".into()
        }),
        ("duplicate method", |p| {
            p.domain.methods.push(p.domain.methods[0].clone())
        }),
        ("duplicate subtask label", |p| {
            p.domain.methods[0].tn.subtasks[1].id = Some(symbol("a"))
        }),
        ("unknown subtask label", |p| {
            p.domain.methods[0].tn.subtasks[1].id = None
        }),
        ("cyclic task ordering", |p| {
            p.domain.methods[0].tn.orderings =
                TaskOrdering::Partial(vec![("a".into(), "b".into()), ("b".into(), "a".into())])
        }),
        ("cyclic task ordering", |p| {
            p.domain.methods[0].tn.orderings = TaskOrdering::Partial(vec![("a".into(), "a".into())])
        }),
        ("names domain", |p| {
            p.problem.as_mut().unwrap().domain_name = "other".into()
        }),
    ];
    for (expected, mutate) in cases {
        let mut program = fixture();
        mutate(&mut program);
        let error = analyze_program(&program).unwrap_err().to_string();
        assert!(error.contains(expected), "expected {expected}, got {error}");
    }
}

#[test]
fn recursion_summaries_reach_a_fixed_point_and_tail_position_is_checked() {
    let mut program = recursive_program(TaskOrdering::Total);
    let analysis = analyze_program(&program).unwrap();
    assert_eq!(analysis.signature.recursion, RecursionClass::TailRecursive);
    let job = &analysis.tasks[task(&analysis, "job").0];
    assert!(job.summary.conditions.contains(&literal("ready", true)));
    assert!(job.summary.effects.contains(&literal("p", false)));
    assert!(
        analysis.methods[1]
            .summary
            .effects
            .contains(&literal("q", true))
    );
    assert_eq!(analysis.signature.max_method_branching, 2);

    program.domain.methods[1].tn.orderings = TaskOrdering::Partial(vec![]);
    assert_eq!(
        analyze_program(&program).unwrap().signature.recursion,
        RecursionClass::GeneralRecursive
    );
    program.domain.methods[1].tn.orderings =
        TaskOrdering::Partial(vec![("again".into(), "step".into())]);
    assert_eq!(
        analyze_program(&program).unwrap().signature.recursion,
        RecursionClass::GeneralRecursive
    );
    program.domain.methods[1].tn.subtasks[0].task = symbol("job");
    assert_eq!(
        analyze_program(&program).unwrap().signature.recursion,
        RecursionClass::GeneralRecursive
    );
}

#[test]
fn detects_mutual_recursion_and_ignores_unreachable_structure_in_signature() {
    let mut program = recursive_program(TaskOrdering::Total);
    program.domain.compound_tasks.push(Task {
        name: "other".into(),
        parameters: vec![],
    });
    program.domain.methods[1].tn.subtasks[1].task = symbol("other");
    let mut back = program.domain.methods[1].clone();
    back.name = symbol("back");
    back.task = symbol("other");
    back.tn.subtasks.remove(0);
    back.tn.subtasks[0].task = symbol("job");
    program.domain.methods.push(back);
    let analysis = analyze_program(&program).unwrap();
    assert_eq!(analysis.signature.largest_recursive_scc, 2);
    assert_eq!(analysis.signature.recursion, RecursionClass::TailRecursive);
    let other = &analysis.tasks[task(&analysis, "other").0];
    assert!(other.summary.effects.contains(&literal("q", true)));

    let initial = &mut program
        .problem
        .as_mut()
        .unwrap()
        .init_tn
        .as_mut()
        .unwrap()
        .tn;
    initial.subtasks.truncate(1);
    initial.subtasks[0].task = symbol("make");
    let analysis = analyze_program(&program).unwrap();
    assert_eq!(analysis.signature.reachable_tasks, 1);
    assert_eq!(analysis.signature.reachable_methods, 0);
    assert_eq!(analysis.signature.recursion, RecursionClass::Acyclic);
    assert!(analysis.signature.fully_unordered);
    assert!(
        analysis
            .recursion_groups
            .iter()
            .any(|group| group.recursive)
    );
    let graph = InteractionGraph::from_analysis(&analysis);
    assert!(graph.nodes.iter().any(|node| !node.reachable));
    program.problem = None;
    assert_eq!(
        analyze_program(&program)
            .unwrap()
            .signature
            .largest_recursive_scc,
        2
    );
}

#[test]
fn empty_networks_and_missing_initial_network_are_well_defined() {
    let program: Program = serde_json::from_str(include_str!("fixtures/minimal.json")).unwrap();
    let analysis = analyze_program(&program).unwrap();
    assert_eq!(analysis.signature.reachable_tasks, 1);
    assert!(analysis.signature.fully_unordered);
    let mut program = fixture();
    program.problem.as_mut().unwrap().init_tn = None;
    let analysis = analyze_program(&program).unwrap();
    assert_eq!(analysis.signature.reachable_tasks, 0);
    assert_eq!(analysis.signature.max_network_size, 0);
    assert!(analysis.signature.fully_unordered);
    program.domain.methods[0].tn.subtasks.clear();
    program.domain.methods[0].tn.orderings = TaskOrdering::Total;
    assert!(analyze_program(&program).is_ok());
}

#[test]
fn fully_unordered_considers_all_reachable_method_bodies() {
    let mut program = fixture();
    program.domain.methods[0].tn.orderings = TaskOrdering::Partial(vec![]);
    assert!(analyze_program(&program).unwrap().signature.fully_unordered);
    let mut ordered = program.domain.methods[0].clone();
    ordered.name = symbol("ordered-alternative");
    ordered.tn.orderings = TaskOrdering::Total;
    program.domain.methods.push(ordered);
    assert!(!analyze_program(&program).unwrap().signature.fully_unordered);
}

#[test]
fn formula_features_and_lifted_terms_are_explicit() {
    let mut program = fixture();
    program.domain.methods[0].params.push(symbol("?unused"));
    program.domain.actions[1].preconditions = Some(Formula::Imply(
        Box::new(atom("p")),
        Box::new(Formula::Exists(
            vec![symbol("?x")],
            Box::new(Formula::Or(vec![atom("q"), atom("ready")])),
        )),
    ));
    let analysis = analyze_program(&program).unwrap();
    let summary = &analysis.tasks[task(&analysis, "use").0].summary;
    assert!(summary.conditions.contains(&literal("p", false)));
    assert!(summary.conditions.contains(&literal("q", true)));
    assert!(
        summary
            .formula_features
            .contains(&FormulaFeature::Implication)
    );
    assert!(
        summary
            .formula_features
            .contains(&FormulaFeature::Quantification)
    );
    assert!(
        summary
            .formula_features
            .contains(&FormulaFeature::Disjunction)
    );
    assert!(analysis.signature.has_lifted_variables);
    assert!(!analysis.warnings.is_empty());
    assert!(serde_json::to_string(&analysis).is_ok());
}

#[test]
fn retains_directed_typed_parallel_edges_and_negative_support() {
    let mut program = fixture();
    // Erasing p can support a negative p condition, while making p threatens it.
    program.domain.actions[1].preconditions = Some(Formula::And(vec![
        atom("p"),
        Formula::Not(Box::new(atom("p"))),
    ]));
    let analysis = analyze_program(&program).unwrap();
    let graph = InteractionGraph::from_analysis(&analysis);
    let nodes = graph.network_nodes(NetworkId::Method(MethodId(0)));
    let has = |source, target, kind| {
        graph
            .edges
            .iter()
            .any(|edge| edge.source == source && edge.target == target && edge.kind == kind)
    };
    assert!(has(nodes[0], nodes[1], EdgeKind::Ordering));
    assert!(has(
        nodes[0],
        nodes[1],
        EdgeKind::Support {
            literal: literal("p", true)
        }
    ));
    assert!(has(
        nodes[2],
        nodes[1],
        EdgeKind::Support {
            literal: literal("p", false)
        }
    ));
    assert!(has(
        nodes[0],
        nodes[1],
        EdgeKind::Threat {
            literal: literal("p", false),
            target: ThreatTarget::Condition
        }
    ));
    assert!(has(
        nodes[1],
        nodes[0],
        EdgeKind::Threat {
            literal: literal("p", true),
            target: ThreatTarget::Effect
        }
    ));
    assert!(has(
        nodes[0],
        nodes[1],
        EdgeKind::SharedFluent {
            predicate: "p".into()
        }
    ));
    assert!(!has(nodes[1], nodes[0], EdgeKind::Ordering));
    assert_eq!(graph.edges_in_view(GraphView::Resource).count(), 0);
    assert!(
        graph
            .edges_in_view(GraphView::SharedState)
            .all(|edge| !edge.kind.is_directed())
    );
    assert!(graph.edges.iter().any(|edge| edge.kind
        == EdgeKind::Hierarchy {
            relation: HierarchyRelation::InstanceOf
        }));
}

#[test]
fn method_alternatives_and_repeated_occurrences_keep_separate_identity() {
    let mut program = fixture();
    let mut alternative = program.domain.methods[0].clone();
    alternative.name = symbol("alternative");
    alternative.tn.subtasks[1].task = symbol("make");
    program.domain.methods.push(alternative);
    let graph = InteractionGraph::from_analysis(&analyze_program(&program).unwrap());
    let first = graph.network_nodes(NetworkId::Method(MethodId(0)));
    let second = graph.network_nodes(NetworkId::Method(MethodId(1)));
    assert_ne!(second[0], second[1]);
    assert!(
        graph
            .edges
            .iter()
            .all(|edge| !(first.contains(&edge.source) && second.contains(&edge.target)))
    );
    for edge in graph
        .edges
        .iter()
        .filter(|edge| edge.kind.view() != GraphView::Hierarchy)
    {
        let network = |node: NodeId| match &graph.nodes[node.0].kind {
            NodeKind::Occurrence { network, .. } => *network,
            _ => panic!("interaction endpoint must be an occurrence"),
        };
        assert_eq!(network(edge.source), network(edge.target));
    }
}

#[test]
fn affinity_sums_selected_views_without_changing_the_graph() {
    let graph = InteractionGraph::from_analysis(&analyze_program(&fixture()).unwrap());
    let before = serde_json::to_string(&graph).unwrap();
    let nodes = graph.network_nodes(NetworkId::Method(MethodId(0)));
    let weights = ViewWeights {
        ordering: 2.0,
        support: 3.0,
        threat: 0.0,
        resource: 0.0,
        hierarchy: 0.0,
        shared_state: 5.0,
    };
    let matrix = graph.symmetric_affinity(&nodes, &weights).unwrap();
    assert_eq!(matrix.nodes, nodes);
    assert_eq!(
        matrix.values,
        vec![
            vec![0.0, 10.0, 5.0],
            vec![10.0, 0.0, 5.0],
            vec![5.0, 5.0, 0.0]
        ]
    );
    assert_eq!(serde_json::to_string(&graph).unwrap(), before);
    let reverse = graph
        .symmetric_affinity(&[nodes[1], nodes[0]], &weights)
        .unwrap();
    assert_eq!(reverse.values, vec![vec![0.0, 10.0], vec![10.0, 0.0]]);
    assert!(
        graph
            .symmetric_affinity(&[], &weights)
            .unwrap()
            .values
            .is_empty()
    );
}

#[test]
fn affinity_rejects_invalid_weights_and_node_selections() {
    let graph = InteractionGraph::from_analysis(&analyze_program(&fixture()).unwrap());
    let nodes = graph.network_nodes(NetworkId::Method(MethodId(0)));
    for weight in [-1.0, f64::NAN, f64::INFINITY] {
        let weights = ViewWeights {
            support: weight,
            ..ViewWeights::default()
        };
        assert!(graph.symmetric_affinity(&nodes, &weights).is_err());
    }
    assert!(
        graph
            .symmetric_affinity(&[nodes[0], nodes[0]], &ViewWeights::default())
            .is_err()
    );
    assert!(
        graph
            .symmetric_affinity(&[NodeId(usize::MAX)], &ViewWeights::default())
            .is_err()
    );
    let weights = ViewWeights {
        ordering: f64::MAX,
        support: f64::MAX,
        ..ViewWeights::default()
    };
    assert!(graph.symmetric_affinity(&nodes, &weights).is_err());
}

#[test]
fn cli_emits_machine_readable_analysis_and_reports_invalid_htns() {
    let fixture_path = format!(
        "{}/tests/fixtures/interaction.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let output = Command::new(env!("CARGO_BIN_EXE_ptu-plan"))
        .args(["--analysis-json", &fixture_path])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let record: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(record["analysis"]["signature"]["ordering"], "Partial");
    assert_eq!(record["graph"]["nodes"].as_array().unwrap().len(), 10);
    let output = Command::new(env!("CARGO_BIN_EXE_ptu-plan"))
        .args(["--analyze", &fixture_path])
        .output()
        .unwrap();
    assert!(output.status.success());
    assert!(String::from_utf8_lossy(&output.stdout).contains("Interaction graph:"));

    let mut invalid = fixture();
    invalid.domain.methods[0].tn.subtasks[0].task = symbol("missing");
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("invalid.json");
    std::fs::write(&path, serde_json::to_vec(&invalid).unwrap()).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_ptu-plan"))
        .arg("--analysis-json")
        .arg(path)
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("unknown task 'missing'"));
}
