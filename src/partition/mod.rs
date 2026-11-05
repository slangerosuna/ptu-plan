//! Deterministic dense normalized-cut micro-partitions of a weighted multi-view
//! affinity. These clusters propose search structure and never remove HTN edges.

use std::collections::{BTreeSet, VecDeque};
use std::fmt;

use nalgebra::{DMatrix, linalg::SymmetricEigen};
use serde::{Deserialize, Serialize};

use crate::analysis::NetworkId;
use crate::interaction::{AffinityMatrix, InteractionGraph, NodeId, ViewWeights};

#[derive(Debug, Clone, Copy, Serialize)]
pub struct SpectralOptions {
    pub max_cluster_size: usize,
    /// Lower bound on connected bisection sides when possible. Disconnected
    /// components and isolates can be smaller; the maximum is always enforced.
    pub min_cluster_size: usize,
    pub weights: ViewWeights,
    pub eigen_tolerance: f64,
    pub max_eigen_iterations: usize,
}

impl Default for SpectralOptions {
    fn default() -> Self {
        Self {
            max_cluster_size: 8,
            min_cluster_size: 1,
            weights: ViewWeights::default(),
            eigen_tolerance: 1e-10,
            max_eigen_iterations: 10_000,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MicroPartition {
    pub id: usize,
    pub nodes: Vec<NodeId>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SpectralSplit {
    pub size: usize,
    pub left_size: usize,
    pub right_size: usize,
    pub fiedler_value: f64,
    pub normalized_cut: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct MicroPartitioning {
    pub clusters: Vec<MicroPartition>,
    pub splits: Vec<SpectralSplit>,
    pub cut_weight: f64,
    pub normalized_cut: f64,
}

/// Partition peer task occurrences only, avoiding accidental clustering of
/// alternative methods together. The typed graph is borrowed and preserved.
pub fn micropartition_network(
    graph: &InteractionGraph,
    network: NetworkId,
    options: &SpectralOptions,
) -> Result<MicroPartitioning, PartitionError> {
    let nodes = graph.network_nodes(network);
    let affinity = graph
        .symmetric_affinity(&nodes, &options.weights)
        .map_err(|error| PartitionError(error.to_string()))?;
    spectral_partition(&affinity, options)
}

/// Recursive normalized-Laplacian bisection with a Fiedler-vector sweep cut.
/// Input node order is canonicalized; cluster IDs follow smallest node ID.
pub fn spectral_partition(
    affinity: &AffinityMatrix,
    options: &SpectralOptions,
) -> Result<MicroPartitioning, PartitionError> {
    if options.max_cluster_size == 0
        || options.min_cluster_size == 0
        || options.min_cluster_size > options.max_cluster_size
    {
        return Err(PartitionError(
            "require 1 <= min_cluster_size <= max_cluster_size".into(),
        ));
    }
    if !options.eigen_tolerance.is_finite()
        || options.eigen_tolerance <= 0.0
        || options.max_eigen_iterations == 0
    {
        return Err(PartitionError(
            "eigensolver tolerance and iteration limit must be positive".into(),
        ));
    }
    let n = affinity.nodes.len();
    if affinity.values.len() != n || affinity.values.iter().any(|row| row.len() != n) {
        return Err(PartitionError(
            "affinity must be square and match node IDs".into(),
        ));
    }
    if affinity.nodes.iter().collect::<BTreeSet<_>>().len() != n {
        return Err(PartitionError("duplicate affinity node ID".into()));
    }
    for i in 0..n {
        for j in 0..n {
            let weight = affinity.values[i][j];
            if !weight.is_finite()
                || weight < 0.0
                || weight != affinity.values[j][i]
                || (i == j && weight != 0.0)
            {
                return Err(PartitionError(
                    "affinity must be finite, nonnegative, symmetric, and have a zero diagonal"
                        .into(),
                ));
            }
        }
    }
    let mut canonical: Vec<_> = (0..n).collect();
    canonical.sort_by_key(|&i| affinity.nodes[i]);
    let nodes: Vec<_> = canonical.iter().map(|&i| affinity.nodes[i]).collect();
    let values: Vec<Vec<_>> = canonical
        .iter()
        .map(|&i| canonical.iter().map(|&j| affinity.values[i][j]).collect())
        .collect();
    let total: f64 = values.iter().flatten().sum();
    if !total.is_finite() {
        return Err(PartitionError("affinity volume overflowed".into()));
    }
    let mut pending = VecDeque::from([canonical
        .iter()
        .enumerate()
        .map(|(i, _)| i)
        .collect::<Vec<_>>()]);
    let mut groups = Vec::new();
    let mut splits = Vec::new();
    while let Some(group) = pending.pop_front() {
        if group.is_empty() {
            continue;
        }
        let components = components(&group, &values);
        if components.len() > 1 {
            pending.extend(components);
            continue;
        }
        if group.len() <= options.max_cluster_size {
            groups.push(group);
            continue;
        }
        let (left, right, split) = bisect(&group, &values, options)?;
        pending.push_back(left);
        pending.push_back(right);
        splits.push(split);
    }
    for group in &mut groups {
        group.sort_unstable();
    }
    groups.sort_by_key(|group| group[0]);
    let mut membership = vec![0; n];
    for (id, group) in groups.iter().enumerate() {
        for &node in group {
            membership[node] = id;
        }
    }
    let mut cut_weight = 0.0;
    for i in 0..n {
        for j in i + 1..n {
            if membership[i] != membership[j] {
                cut_weight += values[i][j];
            }
        }
    }
    let mut normalized_cut = 0.0;
    for (id, group) in groups.iter().enumerate() {
        let volume: f64 = group.iter().map(|&i| values[i].iter().sum::<f64>()).sum();
        let cut: f64 = group
            .iter()
            .map(|&i| {
                (0..n)
                    .filter(|&j| membership[j] != id)
                    .map(|j| values[i][j])
                    .sum::<f64>()
            })
            .sum();
        if volume > 0.0 {
            normalized_cut += cut / volume;
        }
    }
    Ok(MicroPartitioning {
        clusters: groups
            .into_iter()
            .enumerate()
            .map(|(id, group)| MicroPartition {
                id,
                nodes: group.into_iter().map(|i| nodes[i]).collect(),
            })
            .collect(),
        splits,
        cut_weight,
        normalized_cut,
    })
}

fn components(group: &[usize], values: &[Vec<f64>]) -> Vec<Vec<usize>> {
    let mut unseen: BTreeSet<_> = group.iter().copied().collect();
    let mut found = Vec::new();
    while let Some(&start) = unseen.first() {
        unseen.remove(&start);
        let mut component = vec![start];
        let mut queue = VecDeque::from([start]);
        while let Some(from) = queue.pop_front() {
            let neighbors: Vec<_> = unseen
                .iter()
                .copied()
                .filter(|&to| values[from][to] > 0.0)
                .collect();
            for to in neighbors {
                unseen.remove(&to);
                component.push(to);
                queue.push_back(to);
            }
        }
        component.sort_unstable();
        found.push(component);
    }
    found
}

fn bisect(
    group: &[usize],
    values: &[Vec<f64>],
    options: &SpectralOptions,
) -> Result<(Vec<usize>, Vec<usize>, SpectralSplit), PartitionError> {
    let n = group.len();
    let degree: Vec<f64> = group
        .iter()
        .map(|&i| group.iter().map(|&j| values[i][j]).sum())
        .collect();
    let mut laplacian = DMatrix::identity(n, n);
    for i in 0..n {
        for j in 0..n {
            if i != j {
                laplacian[(i, j)] =
                    -(values[group[i]][group[j]] / degree[i].sqrt()) / degree[j].sqrt();
            }
        }
    }
    let eigen = SymmetricEigen::try_new(
        laplacian,
        options.eigen_tolerance,
        options.max_eigen_iterations,
    )
    .ok_or_else(|| PartitionError("normalized Laplacian eigensolver did not converge".into()))?;
    if eigen.eigenvalues.iter().any(|x| !x.is_finite())
        || eigen.eigenvectors.iter().any(|x| !x.is_finite())
    {
        return Err(PartitionError("nonfinite spectral decomposition".into()));
    }
    let mut eigen_order: Vec<_> = (0..n).collect();
    eigen_order.sort_by(|&a, &b| {
        eigen.eigenvalues[a]
            .total_cmp(&eigen.eigenvalues[b])
            .then(a.cmp(&b))
    });
    let fiedler = eigen_order[1];
    let mut coordinates: Vec<f64> = (0..n)
        .map(|i| eigen.eigenvectors[(i, fiedler)] / degree[i].sqrt())
        .collect();
    if coordinates
        .iter()
        .find(|x| x.abs() > options.eigen_tolerance)
        .is_some_and(|x| *x < 0.0)
    {
        for value in &mut coordinates {
            *value = -*value;
        }
    }
    let mut order: Vec<_> = (0..n).collect();
    order.sort_by(|&a, &b| {
        coordinates[a]
            .total_cmp(&coordinates[b])
            .then(group[a].cmp(&group[b]))
    });
    let volume: f64 = degree.iter().sum();
    let minimum = options.min_cluster_size.min(n / 2);
    let mut best: Option<(f64, usize)> = None;
    let mut left_volume = 0.0;
    let mut cut = 0.0;
    for size in 1..n {
        let added = order[size - 1];
        left_volume += degree[added];
        let internal: f64 = order[..size - 1]
            .iter()
            .map(|&j| values[group[added]][group[j]])
            .sum();
        cut = (cut + degree[added] - 2.0 * internal).max(0.0);
        if size < minimum || n - size < minimum {
            continue;
        }
        // Sum directly on the right to avoid cancellation near the sweep end.
        let right_volume: f64 = order[size..].iter().map(|&j| degree[j]).sum();
        let score = cut / left_volume + cut / right_volume;
        if !score.is_finite() || volume <= 0.0 {
            return Err(PartitionError("invalid normalized cut arithmetic".into()));
        }
        let better = best.is_none_or(|(old, old_size)| {
            score < old - 1e-12
                || ((score - old).abs() <= 1e-12
                    && size.abs_diff(n - size) < old_size.abs_diff(n - old_size))
        });
        if better {
            best = Some((score, size));
        }
    }
    let (score, split) =
        best.ok_or_else(|| PartitionError("no valid spectral sweep cut".into()))?;
    Ok((
        order[..split].iter().map(|&i| group[i]).collect(),
        order[split..].iter().map(|&i| group[i]).collect(),
        SpectralSplit {
            size: n,
            left_size: split,
            right_size: n - split,
            fiedler_value: eigen.eigenvalues[fiedler].max(0.0),
            normalized_cut: score,
        },
    ))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartitionError(pub String);
impl fmt::Display for PartitionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for PartitionError {}
