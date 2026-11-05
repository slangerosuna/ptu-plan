use ptu_plan::agglomeration::*;
use ptu_plan::analysis::{HtnAnalysis, NetworkId, analyze_program};
use ptu_plan::interaction::{Edge, EdgeKind, InteractionGraph, NodeId, ViewWeights};
use ptu_plan::parsing::json::{Program, Subtask, Symbol, TaskOrdering};
use ptu_plan::partition::MicroPartition;

fn symbol(s: &str) -> Symbol {
    Symbol {
        name: s.into(),
        symbol_type: None,
    }
}
fn setup(roots: &[&str], ordered: bool) -> (HtnAnalysis, InteractionGraph, Vec<MicroPartition>) {
    let mut p: Program = serde_json::from_str(include_str!("fixtures/interaction.json")).unwrap();
    let tn = &mut p.problem.as_mut().unwrap().init_tn.as_mut().unwrap().tn;
    tn.subtasks = roots
        .iter()
        .enumerate()
        .map(|(i, task)| Subtask {
            id: Some(symbol(&format!("r{i}"))),
            task: symbol(task),
            terms: vec![],
        })
        .collect();
    tn.orderings = if ordered {
        TaskOrdering::Total
    } else {
        TaskOrdering::Partial(vec![])
    };
    let analysis = analyze_program(&p).unwrap();
    let graph = InteractionGraph::from_analysis(&analysis);
    let seeds = graph
        .network_nodes(NetworkId::Initial)
        .into_iter()
        .enumerate()
        .map(|(i, n)| MicroPartition {
            id: i,
            nodes: vec![n],
        })
        .collect();
    (analysis, graph, seeds)
}
fn run(
    a: &HtnAnalysis,
    g: &InteractionGraph,
    s: &[MicroPartition],
    o: &AgglomerationOptions,
) -> Agglomeration {
    greedy_agglomerate(a, g, NetworkId::Initial, s, o).unwrap()
}
fn baseline() -> AgglomerationOptions {
    AgglomerationOptions {
        interface_weight: 0.0,
        size_penalty: 0.0,
        unordered_loss_penalty: 0.0,
        recursive_mix_penalty: 0.0,
        ..AgglomerationOptions::default()
    }
}
fn replace_edges(g: &mut InteractionGraph, s: &[MicroPartition], links: &[(usize, usize, usize)]) {
    g.edges.clear();
    for &(a, b, count) in links {
        for i in 0..count {
            g.edges.push(Edge {
                source: s[a].nodes[0],
                target: s[b].nodes[0],
                kind: EdgeKind::SharedFluent {
                    predicate: format!("edge-{i}"),
                },
            });
        }
    }
}

#[test]
fn merges_strongest_neighbors_and_stops_at_size_or_target_with_a_dendrogram() {
    let (a, mut g, s) = setup(&["make", "make", "make", "make"], false);
    replace_edges(&mut g, &s, &[(0, 1, 4), (1, 2, 1), (2, 3, 3)]);
    let o = AgglomerationOptions {
        max_cluster_size: 2,
        ..baseline()
    };
    let result = run(&a, &g, &s, &o);
    assert_eq!(result.merges.len(), 2);
    assert_eq!(result.merges[0].children, [0, 1]);
    assert_eq!(result.merges[1].children, [2, 3]);
    assert!(result.clusters.iter().all(|c| c.nodes.len() == 2));
    assert_eq!(result.quotient_edges.len(), 1);
    let stopped = run(
        &a,
        &g,
        &s,
        &AgglomerationOptions {
            target_clusters: 3,
            ..o
        },
    );
    assert_eq!(stopped.merges.len(), 1);
    assert_eq!(
        serde_json::to_value(&result).unwrap(),
        serde_json::to_value(run(&a, &g, &s, &o)).unwrap()
    );
}

#[test]
fn retains_parallel_directional_types_and_does_not_merge_unconnected_peers() {
    let (a, g, s) = setup(&["make", "use", "erase"], false);
    let result = run(
        &a,
        &g,
        &s,
        &AgglomerationOptions {
            min_score: 1e9,
            ..baseline()
        },
    );
    let selected: Vec<_> = g
        .edges
        .iter()
        .filter(|e| {
            s.iter().any(|c| c.nodes.contains(&e.source))
                && s.iter().any(|c| c.nodes.contains(&e.target))
        })
        .cloned()
        .collect();
    assert_eq!(
        result
            .quotient_edges
            .iter()
            .map(|e| e.original.clone())
            .collect::<Vec<_>>(),
        selected
    );
    assert!(
        selected
            .iter()
            .any(|e| matches!(e.kind, EdgeKind::Support { .. }))
    );
    assert!(
        selected
            .iter()
            .any(|e| matches!(e.kind, EdgeKind::Threat { .. }))
    );
    let mut disconnected = g.clone();
    disconnected.edges.clear();
    assert!(
        run(
            &a,
            &disconnected,
            &s,
            &AgglomerationOptions {
                min_score: -100.0,
                ..baseline()
            }
        )
        .merges
        .is_empty()
    );
}

#[test]
fn repairs_cyclic_seed_contraction_and_rejects_new_cyclic_merges() {
    let (a, mut g, s) = setup(&["make", "make", "make"], true);
    // Tempting nonconvex merge would produce A -> B -> A.
    g.edges.push(Edge {
        source: s[0].nodes[0],
        target: s[2].nodes[0],
        kind: EdgeKind::SharedFluent {
            predicate: "heavy".into(),
        },
    });
    let result = run(
        &a,
        &g,
        &s,
        &AgglomerationOptions {
            target_clusters: 2,
            weights: ViewWeights {
                ordering: 0.0,
                shared_state: 100.0,
                support: 0.0,
                threat: 0.0,
                resource: 0.0,
                hierarchy: 0.0,
            },
            ..baseline()
        },
    );
    assert!(!result.merges.iter().any(|m| m.children == vec![0, 2]));
    let invalid = vec![
        MicroPartition {
            id: 10,
            nodes: vec![s[0].nodes[0], s[2].nodes[0]],
        },
        MicroPartition {
            id: 11,
            nodes: s[1].nodes.clone(),
        },
    ];
    let repaired = run(
        &a,
        &g,
        &invalid,
        &AgglomerationOptions {
            min_score: 1e9,
            ..baseline()
        },
    );
    assert_eq!(repaired.clusters.len(), 1);
    assert!(matches!(
        repaired.merges[0].reason,
        MergeReason::OrderingCycle
    ));
    assert!(
        greedy_agglomerate(
            &a,
            &g,
            NetworkId::Initial,
            &invalid,
            &AgglomerationOptions {
                max_cluster_size: 2,
                ..baseline()
            }
        )
        .is_err()
    );
}

#[test]
fn scores_unordered_mass_and_exposes_recursive_structural_features() {
    let (a, g, s) = setup(&["make", "erase"], true);
    let result = run(
        &a,
        &g,
        &s,
        &AgglomerationOptions {
            target_clusters: 2,
            ..baseline()
        },
    );
    assert!(result.features.values().all(|f| f.fully_unordered));
    let merged = run(&a, &g, &s, &baseline());
    assert_eq!(merged.merges[0].score.unordered_mass_lost, 2);
    assert!(!merged.features.values().next().unwrap().fully_unordered);
    let penalized = run(
        &a,
        &g,
        &s,
        &AgglomerationOptions {
            unordered_loss_penalty: 1000.0,
            ..baseline()
        },
    );
    assert!(penalized.merges.is_empty());

    let mut p: Program = serde_json::from_str(include_str!("fixtures/interaction.json")).unwrap();
    let mut recursive = p.domain.methods[0].clone();
    recursive.name = symbol("recurse");
    recursive.tn.subtasks = vec![Subtask {
        id: None,
        task: symbol("job"),
        terms: vec![],
    }];
    recursive.tn.orderings = TaskOrdering::Partial(vec![]);
    recursive.tn.constraints = None;
    p.domain.methods.push(recursive);
    let a = analyze_program(&p).unwrap();
    let g = InteractionGraph::from_analysis(&a);
    let s: Vec<_> = g
        .network_nodes(NetworkId::Initial)
        .into_iter()
        .enumerate()
        .map(|(id, n)| MicroPartition { id, nodes: vec![n] })
        .collect();
    let result = run(&a, &g, &s, &baseline());
    assert!(result.merges[0].score.recursive_mix);
    assert!(result.features.values().next().unwrap().recursive);
}

#[test]
fn rejects_invalid_coverage_penalties_and_weights() {
    let (a, g, s) = setup(&["make", "use"], false);
    let bad = [
        vec![],
        vec![s[0].clone()],
        vec![s[0].clone(), s[0].clone()],
        vec![
            MicroPartition {
                id: 0,
                nodes: vec![NodeId(usize::MAX)],
            },
            s[1].clone(),
        ],
    ];
    for seeds in bad {
        assert!(greedy_agglomerate(&a, &g, NetworkId::Initial, &seeds, &baseline()).is_err());
    }
    for options in [
        AgglomerationOptions {
            size_penalty: f64::NAN,
            ..baseline()
        },
        AgglomerationOptions {
            max_cluster_size: 0,
            ..baseline()
        },
        AgglomerationOptions {
            weights: ViewWeights {
                support: -1.0,
                ..ViewWeights::default()
            },
            ..baseline()
        },
    ] {
        assert!(greedy_agglomerate(&a, &g, NetworkId::Initial, &s, &options).is_err());
    }
}
