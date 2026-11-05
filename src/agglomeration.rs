//! Greedy structural baseline over the original typed quotient graph.
//! Scores propose partitions; they never assert solvability or complexity bounds.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::analysis::{HtnAnalysis, NetworkId, OrderingClass};
use crate::interaction::{Edge, EdgeKind, InteractionGraph, NodeId, NodeKind, ViewWeights};
use crate::partition::{MicroPartition, PartitionError};

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
pub struct AgglomerationOptions {
    pub max_cluster_size: usize,
    pub target_clusters: usize,
    pub min_score: f64,
    pub weights: ViewWeights,
    pub interface_weight: f64,
    pub size_penalty: f64,
    pub unordered_loss_penalty: f64,
    pub recursive_mix_penalty: f64,
}

impl Default for AgglomerationOptions {
    fn default() -> Self {
        Self {
            max_cluster_size: 32,
            target_clusters: 1,
            min_score: 0.0,
            weights: ViewWeights::default(),
            interface_weight: 1.0,
            size_penalty: 0.1,
            unordered_loss_penalty: 1.0,
            recursive_mix_penalty: 0.5,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ClusterFeatures {
    pub size: usize,
    pub fluent_families: BTreeSet<String>,
    pub boundary_families: BTreeSet<String>,
    pub fully_unordered: bool,
    pub recursive: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct QuotientEdge {
    pub source_cluster: usize,
    pub target_cluster: usize,
    /// Preserve direction, endpoints, predicate, and relation of every edge.
    pub original: Edge,
}

#[derive(Debug, Clone, Serialize)]
pub enum MergeReason {
    Greedy,
    Policy,
    /// Repair spectral seeds whose contraction creates an ordering cycle.
    OrderingCycle,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct MergeScore {
    pub coupling: f64,
    pub interface_reduction: f64,
    pub size_cost: f64,
    pub unordered_mass_lost: usize,
    pub recursive_mix: bool,
    pub total: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct MergeStep {
    pub children: Vec<usize>,
    pub parent: MicroPartition,
    pub reason: MergeReason,
    pub score: MergeScore,
}

#[derive(Debug, Clone, Serialize)]
pub struct Agglomeration {
    pub seeds: Vec<MicroPartition>,
    pub merges: Vec<MergeStep>,
    pub clusters: Vec<MicroPartition>,
    pub features: BTreeMap<usize, ClusterFeatures>,
    pub quotient_edges: Vec<QuotientEdge>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MergeCandidate {
    pub left: usize,
    pub right: usize,
    pub score: MergeScore,
}

/// Validate/repair seeds and observe the current level without greedy merges.
pub fn observe(
    analysis: &HtnAnalysis,
    graph: &InteractionGraph,
    network: NetworkId,
    clusters: &[MicroPartition],
    options: &AgglomerationOptions,
) -> Result<Agglomeration, PartitionError> {
    greedy_agglomerate(
        analysis,
        graph,
        network,
        clusters,
        &AgglomerationOptions {
            target_clusters: usize::MAX,
            ..*options
        },
    )
}

/// Legal adjacent contractions, including scores below the greedy threshold.
/// Learned policies may rank these but cannot bypass size or ordering checks.
pub fn legal_candidates(
    graph: &InteractionGraph,
    network: NetworkId,
    state: &Agglomeration,
    options: &AgglomerationOptions,
) -> Result<Vec<MergeCandidate>, PartitionError> {
    let nodes = graph.network_nodes(network);
    let affinity = graph
        .symmetric_affinity(&nodes, &options.weights)
        .map_err(|e| PartitionError(e.to_string()))?;
    let positions = nodes.iter().enumerate().map(|(i, &n)| (n, i)).collect();
    candidates(
        graph,
        &state.clusters,
        &state.features,
        &affinity.values,
        &positions,
        options,
    )
}

pub fn merged_clusters(
    clusters: &[MicroPartition],
    candidate: &MergeCandidate,
) -> Result<Vec<MicroPartition>, PartitionError> {
    if candidate.left == candidate.right {
        return Err(PartitionError("merge endpoints must differ".into()));
    }
    let members: Vec<_> = clusters
        .iter()
        .enumerate()
        .filter(|(_, c)| c.id == candidate.left || c.id == candidate.right)
        .map(|(i, _)| i)
        .collect();
    if members.len() != 2 {
        return Err(PartitionError("unknown merge endpoints".into()));
    }
    let mut result = clusters.to_vec();
    let mut next = clusters.iter().map(|c| c.id).max().unwrap_or(0);
    merge(
        &mut result,
        &members,
        &mut next,
        &mut Vec::new(),
        MergeReason::Greedy,
        candidate.score.clone(),
    )?;
    Ok(result)
}

/// Merge adjacent clusters deterministically, rejecting contractions that would
/// prevent contiguous block execution. Initial quotient SCCs are repaired first,
/// even below the score threshold; the hard size limit also applies to repairs.
pub fn greedy_agglomerate(
    analysis: &HtnAnalysis,
    graph: &InteractionGraph,
    network: NetworkId,
    seeds: &[MicroPartition],
    options: &AgglomerationOptions,
) -> Result<Agglomeration, PartitionError> {
    if options.max_cluster_size == 0
        || options.target_clusters == 0
        || !options.min_score.is_finite()
        || [
            options.interface_weight,
            options.size_penalty,
            options.unordered_loss_penalty,
            options.recursive_mix_penalty,
        ]
        .iter()
        .any(|x| !x.is_finite() || *x < 0.0)
    {
        return Err(PartitionError(
            "invalid agglomeration size, score, or penalty".into(),
        ));
    }
    let peers = graph.network_nodes(network);
    // Also validates the weights and graph endpoints used by the projection.
    let affinity = graph
        .symmetric_affinity(&peers, &options.weights)
        .map_err(|e| PartitionError(e.to_string()))?;
    let positions: BTreeMap<_, _> = peers.iter().enumerate().map(|(i, &n)| (n, i)).collect();
    let mut seen = BTreeSet::new();
    let mut ids = BTreeSet::new();
    let mut clusters = seeds.to_vec();
    for cluster in &mut clusters {
        if cluster.nodes.is_empty()
            || cluster.nodes.len() > options.max_cluster_size
            || !ids.insert(cluster.id)
        {
            return Err(PartitionError(
                "empty, oversized, or duplicate seed cluster".into(),
            ));
        }
        cluster.nodes.sort();
        for &node in &cluster.nodes {
            if !positions.contains_key(&node) || !seen.insert(node) {
                return Err(PartitionError(
                    "seed clusters must cover each peer occurrence exactly once".into(),
                ));
            }
        }
    }
    if seen.len() != peers.len() {
        return Err(PartitionError(
            "seed clusters do not cover the network".into(),
        ));
    }
    clusters.sort_by_key(|c| c.id);
    let mut next = ids.last().copied().unwrap_or(0);
    let mut merges = Vec::new();
    // Contract entire SCCs, not arbitrary pairs: pairwise contraction can leave
    // a cyclic intermediate quotient even when the full repair is legal.
    loop {
        let reach = quotient_ordering(graph, &clusters);
        let component = (0..clusters.len()).find_map(|i| {
            let members: Vec<_> = (0..clusters.len())
                .filter(|&j| i == j || (reach[i][j] && reach[j][i]))
                .collect();
            (members.len() > 1).then_some(members)
        });
        let Some(members) = component else { break };
        let size: usize = members.iter().map(|&i| clusters[i].nodes.len()).sum();
        if size > options.max_cluster_size {
            return Err(PartitionError(format!(
                "repairing an ordering cycle requires {size} roots, above max_cluster_size"
            )));
        }
        merge(
            &mut clusters,
            &members,
            &mut next,
            &mut merges,
            MergeReason::OrderingCycle,
            MergeScore::default(),
        )?;
    }
    while clusters.len() > options.target_clusters {
        let features = features(analysis, graph, &clusters);
        let choices = candidates(
            graph,
            &clusters,
            &features,
            &affinity.values,
            &positions,
            options,
        )?;
        let mut best: Option<MergeCandidate> = None;
        for choice in choices {
            if choice.score.total >= options.min_score
                && best
                    .as_ref()
                    .is_none_or(|old| choice.score.total > old.score.total)
            {
                best = Some(choice);
            }
        }
        let Some(choice) = best else { break };
        let a = clusters.iter().position(|c| c.id == choice.left).unwrap();
        let b = clusters.iter().position(|c| c.id == choice.right).unwrap();
        let score = choice.score;
        merge(
            &mut clusters,
            &[a, b],
            &mut next,
            &mut merges,
            MergeReason::Greedy,
            score,
        )?;
    }
    let features = features(analysis, graph, &clusters);
    let owner = ownership(&clusters);
    let quotient_edges = graph
        .edges
        .iter()
        .filter_map(|edge| {
            let (&source_cluster, &target_cluster) =
                (owner.get(&edge.source)?, owner.get(&edge.target)?);
            (source_cluster != target_cluster).then(|| QuotientEdge {
                source_cluster,
                target_cluster,
                original: edge.clone(),
            })
        })
        .collect();
    Ok(Agglomeration {
        seeds: seeds.to_vec(),
        merges,
        clusters,
        features,
        quotient_edges,
    })
}

fn candidates(
    graph: &InteractionGraph,
    clusters: &[MicroPartition],
    features: &BTreeMap<usize, ClusterFeatures>,
    affinity: &[Vec<f64>],
    positions: &BTreeMap<NodeId, usize>,
    options: &AgglomerationOptions,
) -> Result<Vec<MergeCandidate>, PartitionError> {
    // Compute closure once per level, not once for every candidate pair.
    let reach = quotient_ordering(graph, clusters);
    let mut family_counts = BTreeMap::<&str, usize>::new();
    for feature in features.values() {
        for family in &feature.fluent_families {
            *family_counts.entry(family).or_default() += 1;
        }
    }
    let mut result = Vec::new();
    for a in 0..clusters.len() {
        for b in a + 1..clusters.len() {
            let left = &clusters[a];
            let right = &clusters[b];
            if left.nodes.len() + right.nodes.len() > options.max_cluster_size {
                continue;
            }
            let coupling: f64 = left
                .nodes
                .iter()
                .flat_map(|n| {
                    right
                        .nodes
                        .iter()
                        .map(|m| affinity[positions[n]][positions[m]])
                })
                .sum();
            if coupling <= 0.0 {
                continue;
            }
            if merge_creates_cycle(&reach, a, b) {
                continue;
            }
            let lf = &features[&left.id];
            let rf = &features[&right.id];
            let boundary_after = lf
                .fluent_families
                .union(&rf.fluent_families)
                .filter(|family| {
                    family_counts[family.as_str()]
                        > usize::from(lf.fluent_families.contains(*family))
                            + usize::from(rf.fluent_families.contains(*family))
                })
                .count();
            let merged_unordered =
                lf.fully_unordered && rf.fully_unordered && !reach[a][b] && !reach[b][a];
            let unordered_mass_lost = if merged_unordered {
                0
            } else {
                usize::from(lf.fully_unordered) * lf.size
                    + usize::from(rf.fully_unordered) * rf.size
            };
            let mut score = MergeScore {
                coupling,
                interface_reduction: (lf.boundary_families.len() + rf.boundary_families.len())
                    as f64
                    - boundary_after as f64,
                size_cost: (lf.size as f64 * rf.size as f64) / options.max_cluster_size as f64,
                unordered_mass_lost,
                recursive_mix: lf.recursive != rf.recursive,
                total: 0.0,
            };
            score.total = coupling + options.interface_weight * score.interface_reduction
                - options.size_penalty * score.size_cost
                - options.unordered_loss_penalty * score.unordered_mass_lost as f64
                - options.recursive_mix_penalty * f64::from(score.recursive_mix);
            if !score.total.is_finite() {
                return Err(PartitionError("agglomeration score overflow".into()));
            }
            result.push(MergeCandidate {
                left: left.id,
                right: right.id,
                score,
            });
        }
    }
    Ok(result)
}

// Contracting vertices in a DAG creates a cycle exactly when a path between
// them has an external intermediate vertex. A direct edge alone is legal.
fn merge_creates_cycle(reach: &[Vec<bool>], a: usize, b: usize) -> bool {
    (0..reach.len()).any(|via| {
        via != a
            && via != b
            && ((reach[a][via] && reach[via][b]) || (reach[b][via] && reach[via][a]))
    })
}

fn merge(
    clusters: &mut Vec<MicroPartition>,
    members: &[usize],
    next: &mut usize,
    history: &mut Vec<MergeStep>,
    reason: MergeReason,
    score: MergeScore,
) -> Result<(), PartitionError> {
    *next = next
        .checked_add(1)
        .ok_or_else(|| PartitionError("cluster ID overflow".into()))?;
    let children = members.iter().map(|&i| clusters[i].id).collect();
    let mut nodes: Vec<_> = members
        .iter()
        .flat_map(|&i| clusters[i].nodes.iter().copied())
        .collect();
    nodes.sort();
    let parent = MicroPartition { id: *next, nodes };
    for &i in members.iter().rev() {
        clusters.remove(i);
    }
    clusters.push(parent.clone());
    clusters.sort_by_key(|c| c.id);
    history.push(MergeStep {
        children,
        parent,
        reason,
        score,
    });
    Ok(())
}

fn ownership(clusters: &[MicroPartition]) -> BTreeMap<NodeId, usize> {
    clusters
        .iter()
        .flat_map(|c| c.nodes.iter().map(move |&n| (n, c.id)))
        .collect()
}

fn quotient_ordering(graph: &InteractionGraph, clusters: &[MicroPartition]) -> Vec<Vec<bool>> {
    let owner: BTreeMap<_, _> = clusters
        .iter()
        .enumerate()
        .flat_map(|(i, c)| c.nodes.iter().map(move |&n| (n, i)))
        .collect();
    let mut reach = vec![vec![false; clusters.len()]; clusters.len()];
    for edge in &graph.edges {
        if edge.kind == EdgeKind::Ordering
            && let (Some(&a), Some(&b)) = (owner.get(&edge.source), owner.get(&edge.target))
            && a != b
        {
            reach[a][b] = true;
        }
    }
    for k in 0..reach.len() {
        for i in 0..reach.len() {
            for j in 0..reach.len() {
                reach[i][j] |= reach[i][k] && reach[k][j];
            }
        }
    }
    reach
}

fn features(
    analysis: &HtnAnalysis,
    graph: &InteractionGraph,
    clusters: &[MicroPartition],
) -> BTreeMap<usize, ClusterFeatures> {
    let mut result: BTreeMap<_, _> = clusters
        .iter()
        .map(|c| (c.id, features_for(analysis, graph, &c.nodes)))
        .collect();
    for c in clusters {
        let outside: BTreeSet<_> = result
            .iter()
            .filter(|(id, _)| **id != c.id)
            .flat_map(|(_, f)| f.fluent_families.iter().cloned())
            .collect();
        let f = result.get_mut(&c.id).unwrap();
        f.boundary_families = f.fluent_families.intersection(&outside).cloned().collect();
    }
    result
}

fn features_for(
    analysis: &HtnAnalysis,
    graph: &InteractionGraph,
    nodes: &[NodeId],
) -> ClusterFeatures {
    let selected: BTreeSet<_> = nodes.iter().copied().collect();
    let mut tasks = BTreeSet::new();
    let mut pending = Vec::new();
    let mut fluent_families = BTreeSet::new();
    for node in &graph.nodes {
        if selected.contains(&node.id) {
            if let NodeKind::Occurrence { task, .. } = &node.kind {
                pending.push(*task);
            }
            fluent_families.extend(
                node.summary
                    .fluent_families()
                    .into_iter()
                    .map(str::to_owned),
            );
        }
    }
    let mut fully_unordered = !graph.edges.iter().any(|e| {
        e.kind == EdgeKind::Ordering && selected.contains(&e.source) && selected.contains(&e.target)
    });
    while let Some(task) = pending.pop() {
        if !tasks.insert(task) {
            continue;
        }
        for &method in &analysis.tasks[task.0].methods {
            let body = &analysis.methods[method.0].network;
            fully_unordered &= body.ordering == OrderingClass::Unordered;
            pending.extend(body.occurrences.iter().map(|o| o.task));
        }
    }
    let recursive = analysis
        .recursion_groups
        .iter()
        .any(|g| g.recursive && g.tasks.iter().any(|t| tasks.contains(t)));
    ClusterFeatures {
        size: nodes.len(),
        fluent_families,
        boundary_families: BTreeSet::new(),
        fully_unordered,
        recursive,
    }
}

#[cfg(test)]
mod tests {
    use super::merge_creates_cycle;

    #[test]
    fn cached_reachability_matches_exhaustive_dag_contraction() {
        let n = 5;
        let pairs: Vec<_> = (0..n)
            .flat_map(|a| (a + 1..n).map(move |b| (a, b)))
            .collect();
        for mask in 0..(1usize << pairs.len()) {
            let mut edges = vec![vec![false; n]; n];
            for (bit, &(a, b)) in pairs.iter().enumerate() {
                edges[a][b] = mask & (1 << bit) != 0;
            }
            let closure = |mut matrix: Vec<Vec<bool>>| {
                for k in 0..matrix.len() {
                    for a in 0..matrix.len() {
                        for b in 0..matrix.len() {
                            matrix[a][b] |= matrix[a][k] && matrix[k][b];
                        }
                    }
                }
                matrix
            };
            let reach = closure(edges.clone());
            for &(a, b) in &pairs {
                let mut contracted = vec![vec![false; n]; n];
                for (i, row) in edges.iter().enumerate() {
                    for (j, &edge) in row.iter().enumerate() {
                        let left = if i == b { a } else { i };
                        let right = if j == b { a } else { j };
                        if left != right {
                            contracted[left][right] |= edge;
                        }
                    }
                }
                let actual = closure(contracted);
                assert_eq!(
                    merge_creates_cycle(&reach, a, b),
                    (0..n).any(|i| actual[i][i]),
                    "mask {mask}, pair {a},{b}"
                );
            }
        }
    }
}
