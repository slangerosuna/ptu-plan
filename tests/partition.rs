use std::collections::BTreeSet;

use ptu_plan::analysis::{NetworkId, analyze_program};
use ptu_plan::interaction::{AffinityMatrix, InteractionGraph, NodeId, ViewWeights};
use ptu_plan::parsing::json::Program;
use ptu_plan::partition::{SpectralOptions, micropartition_network, spectral_partition};

fn matrix(size: usize, edges: &[(usize, usize, f64)]) -> AffinityMatrix {
    let mut values = vec![vec![0.0; size]; size];
    for &(a, b, w) in edges {
        values[a][b] = w;
        values[b][a] = w;
    }
    AffinityMatrix {
        nodes: (0..size).map(NodeId).collect(),
        values,
    }
}

#[test]
fn normalized_laplacian_splits_a_known_path_at_its_middle() {
    let affinity = matrix(4, &[(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)]);
    let result = spectral_partition(
        &affinity,
        &SpectralOptions {
            max_cluster_size: 2,
            ..SpectralOptions::default()
        },
    )
    .unwrap();
    assert_eq!(
        result
            .clusters
            .iter()
            .map(|c| c.nodes.clone())
            .collect::<Vec<_>>(),
        vec![vec![NodeId(0), NodeId(1)], vec![NodeId(2), NodeId(3)]]
    );
    assert_eq!(result.cut_weight, 1.0);
    assert!((result.normalized_cut - 2.0 / 3.0).abs() < 1e-9);
    assert_eq!(result.splits.len(), 1);
    assert!((result.splits[0].fiedler_value - 0.5).abs() < 1e-8);
}

#[test]
fn spectral_sweep_finds_weakly_connected_dense_groups() {
    let mut edges = Vec::new();
    for offset in [0, 4] {
        for i in 0..4 {
            for j in i + 1..4 {
                edges.push((offset + i, offset + j, 10.0));
            }
        }
    }
    edges.push((3, 4, 0.1));
    let result = spectral_partition(
        &matrix(8, &edges),
        &SpectralOptions {
            max_cluster_size: 4,
            min_cluster_size: 2,
            ..SpectralOptions::default()
        },
    )
    .unwrap();
    assert_eq!(result.clusters.len(), 2);
    assert_eq!(
        result.clusters[0].nodes,
        (0..4).map(NodeId).collect::<Vec<_>>()
    );
    assert_eq!(
        result.clusters[1].nodes,
        (4..8).map(NodeId).collect::<Vec<_>>()
    );
    assert!((result.cut_weight - 0.1).abs() < 1e-10);
}

#[test]
fn disconnected_components_and_isolates_never_require_an_eigensolve() {
    let result = spectral_partition(
        &matrix(5, &[(0, 1, 1.0), (2, 3, 1.0)]),
        &SpectralOptions {
            min_cluster_size: 3,
            ..SpectralOptions::default()
        },
    )
    .unwrap();
    assert_eq!(result.clusters.len(), 3);
    assert_eq!(result.clusters[2].nodes, vec![NodeId(4)]);
    assert!(result.splits.is_empty());
    assert_eq!(result.cut_weight, 0.0);
    assert_eq!(result.normalized_cut, 0.0);
    assert!(
        spectral_partition(&matrix(0, &[]), &SpectralOptions::default())
            .unwrap()
            .clusters
            .is_empty()
    );
}

#[test]
fn recursive_bisection_respects_size_bounds_and_preserves_every_node_once() {
    let edges: Vec<_> = (0..10).map(|i| (i, i + 1, 1.0)).collect();
    let affinity = matrix(11, &edges);
    let options = SpectralOptions {
        max_cluster_size: 2,
        min_cluster_size: 2,
        ..SpectralOptions::default()
    };
    let result = spectral_partition(&affinity, &options).unwrap();
    assert!(
        result
            .clusters
            .iter()
            .all(|cluster| !cluster.nodes.is_empty() && cluster.nodes.len() <= 2)
    );
    let nodes: Vec<_> = result
        .clusters
        .iter()
        .flat_map(|c| c.nodes.clone())
        .collect();
    assert_eq!(nodes.len(), 11);
    assert_eq!(
        nodes.into_iter().collect::<BTreeSet<_>>(),
        affinity.nodes.iter().copied().collect()
    );
    assert_eq!(
        result
            .clusters
            .iter()
            .map(|c| c.nodes.clone())
            .collect::<Vec<_>>(),
        spectral_partition(&affinity, &options)
            .unwrap()
            .clusters
            .iter()
            .map(|c| c.nodes.clone())
            .collect::<Vec<_>>()
    );
}

#[test]
fn canonicalization_makes_input_permutations_equivalent() {
    let affinity = matrix(4, &[(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)]);
    let permuted = AffinityMatrix {
        nodes: affinity.nodes.iter().rev().copied().collect(),
        values: affinity
            .values
            .iter()
            .rev()
            .map(|row| row.iter().rev().copied().collect())
            .collect(),
    };
    let options = SpectralOptions {
        max_cluster_size: 2,
        ..SpectralOptions::default()
    };
    assert_eq!(
        serde_json::to_value(spectral_partition(&affinity, &options).unwrap()).unwrap(),
        serde_json::to_value(spectral_partition(&permuted, &options).unwrap()).unwrap()
    );
}

#[test]
fn weighted_graph_views_change_partitions_without_mutating_typed_edges() {
    let program: Program = serde_json::from_str(include_str!("fixtures/solver.json")).unwrap();
    let graph = InteractionGraph::from_analysis(&analyze_program(&program).unwrap());
    let original = serde_json::to_string(&graph).unwrap();
    let none = ViewWeights {
        ordering: 0.0,
        support: 0.0,
        threat: 0.0,
        resource: 0.0,
        hierarchy: 0.0,
        shared_state: 0.0,
    };
    let isolated = micropartition_network(
        &graph,
        NetworkId::Initial,
        &SpectralOptions {
            weights: none,
            ..SpectralOptions::default()
        },
    )
    .unwrap();
    assert_eq!(isolated.clusters.len(), 2);
    let joined = micropartition_network(
        &graph,
        NetworkId::Initial,
        &SpectralOptions {
            weights: ViewWeights {
                shared_state: 1.0,
                ..none
            },
            ..SpectralOptions::default()
        },
    )
    .unwrap();
    assert_eq!(joined.clusters.len(), 1);
    assert_eq!(serde_json::to_string(&graph).unwrap(), original);
}

#[test]
fn malformed_affinities_and_invalid_options_return_errors() {
    let valid = matrix(2, &[(0, 1, 1.0)]);
    let options = SpectralOptions::default();
    for bad in [
        AffinityMatrix {
            nodes: vec![NodeId(0)],
            values: vec![],
        },
        AffinityMatrix {
            nodes: vec![NodeId(0), NodeId(0)],
            values: valid.values.clone(),
        },
        AffinityMatrix {
            nodes: valid.nodes.clone(),
            values: vec![vec![0.0, 1.0], vec![2.0, 0.0]],
        },
        matrix(2, &[(0, 1, -1.0)]),
        matrix(2, &[(0, 1, f64::NAN)]),
        matrix(2, &[(0, 1, f64::INFINITY)]),
        matrix(2, &[(0, 0, 1.0)]),
        matrix(3, &[(0, 1, f64::MAX), (1, 2, f64::MAX)]),
    ] {
        assert!(spectral_partition(&bad, &options).is_err());
    }
    for options in [
        SpectralOptions {
            max_cluster_size: 0,
            ..options
        },
        SpectralOptions {
            min_cluster_size: 0,
            ..options
        },
        SpectralOptions {
            min_cluster_size: 9,
            ..options
        },
        SpectralOptions {
            eigen_tolerance: f64::NAN,
            ..options
        },
        SpectralOptions {
            eigen_tolerance: 0.0,
            ..options
        },
        SpectralOptions {
            max_eigen_iterations: 0,
            ..options
        },
    ] {
        assert!(spectral_partition(&valid, &options).is_err());
    }
}
