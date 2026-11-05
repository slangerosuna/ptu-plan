//! Directed, typed structural views plus a separate spectral affinity projection.
//!
//! Causal edges are possible predicate-family interactions, not causal links.
//! They ignore argument unification, applicability, and ordering feasibility.
//! Interactions are generated only between peers in the same task network;
//! alternative method bodies never masquerade as simultaneous occurrences.

mod build;

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::analysis::{Literal, MethodId, NetworkId, TaskId, TaskKind, TaskSummary};
use crate::parsing::json::Symbol;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Deserialize, Serialize)]
#[serde(transparent)]
pub struct NodeId(pub usize);

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum NodeKind {
    TaskSchema {
        task: TaskId,
        kind: TaskKind,
    },
    Method {
        method: MethodId,
        task: TaskId,
    },
    Occurrence {
        network: NetworkId,
        index: usize,
        task: TaskId,
        terms: Vec<Symbol>,
    },
}

#[derive(Debug, Clone, Serialize)]
pub struct Node {
    pub id: NodeId,
    pub label: String,
    pub kind: NodeKind,
    pub reachable: bool,
    pub summary: TaskSummary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum GraphView {
    Ordering,
    Support,
    Threat,
    Resource,
    Hierarchy,
    SharedState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum HierarchyRelation {
    /// Task schema -> alternative method.
    Decomposes,
    /// Method -> child occurrence.
    Contains,
    /// Occurrence -> referenced task schema.
    InstanceOf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ThreatTarget {
    Condition,
    Effect,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum EdgeKind {
    Ordering,
    /// Source can produce the polarity that the target may read.
    Support {
        literal: Literal,
    },
    /// Source may produce the complement of a target condition or effect.
    Threat {
        literal: Literal,
        target: ThreatTarget,
    },
    /// Extension point only: numeric/resource inference is not implemented.
    Resource {
        resource: String,
    },
    Hierarchy {
        relation: HierarchyRelation,
    },
    /// Undirected; stored once with the smaller node ID first.
    SharedFluent {
        predicate: String,
    },
}

impl EdgeKind {
    pub fn view(&self) -> GraphView {
        match self {
            Self::Ordering => GraphView::Ordering,
            Self::Support { .. } => GraphView::Support,
            Self::Threat { .. } => GraphView::Threat,
            Self::Resource { .. } => GraphView::Resource,
            Self::Hierarchy { .. } => GraphView::Hierarchy,
            Self::SharedFluent { .. } => GraphView::SharedState,
        }
    }

    pub fn is_directed(&self) -> bool {
        !matches!(self, Self::SharedFluent { .. } | Self::Resource { .. })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Edge {
    pub source: NodeId,
    pub target: NodeId,
    pub kind: EdgeKind,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct InteractionGraph {
    pub nodes: Vec<Node>,
    /// Parallel edges retain distinct types, predicates, and threat targets.
    pub edges: Vec<Edge>,
}

/// Nonnegative weights used only for the spectral projection.
#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
pub struct ViewWeights {
    pub ordering: f64,
    pub support: f64,
    pub threat: f64,
    pub resource: f64,
    pub hierarchy: f64,
    pub shared_state: f64,
}

impl Default for ViewWeights {
    fn default() -> Self {
        Self {
            ordering: 1.0,
            support: 1.0,
            threat: 1.0,
            resource: 1.0,
            hierarchy: 1.0,
            shared_state: 1.0,
        }
    }
}

impl ViewWeights {
    fn get(&self, view: GraphView) -> f64 {
        match view {
            GraphView::Ordering => self.ordering,
            GraphView::Support => self.support,
            GraphView::Threat => self.threat,
            GraphView::Resource => self.resource,
            GraphView::Hierarchy => self.hierarchy,
            GraphView::SharedState => self.shared_state,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct AffinityMatrix {
    /// Row/column order, preserving the caller's selection order.
    pub nodes: Vec<NodeId>,
    pub values: Vec<Vec<f64>>,
}

impl InteractionGraph {
    pub fn edges_in_view(&self, view: GraphView) -> impl Iterator<Item = &Edge> {
        self.edges
            .iter()
            .filter(move |edge| edge.kind.view() == view)
    }

    pub fn network_nodes(&self, network: NetworkId) -> Vec<NodeId> {
        self.nodes
            .iter()
            .filter(|node| matches!(&node.kind, NodeKind::Occurrence { network: id, .. } if *id == network))
            .map(|node| node.id)
            .collect()
    }

    /// Return a separate dense matrix for selected nodes. Each stored edge
    /// adds its view weight to both off-diagonal entries (sum symmetrization,
    /// not averaging). Parallel edges add; unselected endpoints are omitted.
    /// The original graph and its directed edge types remain untouched.
    pub fn symmetric_affinity(
        &self,
        nodes: &[NodeId],
        weights: &ViewWeights,
    ) -> Result<AffinityMatrix, GraphError> {
        use std::collections::BTreeMap;
        for view in [
            GraphView::Ordering,
            GraphView::Support,
            GraphView::Threat,
            GraphView::Resource,
            GraphView::Hierarchy,
            GraphView::SharedState,
        ] {
            let weight = weights.get(view);
            if !weight.is_finite() || weight < 0.0 {
                return Err(GraphError(format!(
                    "invalid {view:?} affinity weight: {weight}"
                )));
            }
        }
        let mut indices = BTreeMap::new();
        for (index, &node) in nodes.iter().enumerate() {
            if self.nodes.get(node.0).is_none_or(|entry| entry.id != node) {
                return Err(GraphError(format!("unknown node {}", node.0)));
            }
            if indices.insert(node, index).is_some() {
                return Err(GraphError(format!(
                    "duplicate node {} in affinity selection",
                    node.0
                )));
            }
        }
        let mut values = vec![vec![0.0; nodes.len()]; nodes.len()];
        for edge in &self.edges {
            if let (Some(&left), Some(&right)) =
                (indices.get(&edge.source), indices.get(&edge.target))
                && left != right
            {
                let weight = weights.get(edge.kind.view());
                values[left][right] += weight;
                values[right][left] += weight;
                if !values[left][right].is_finite() {
                    return Err(GraphError("affinity weight sum overflowed".into()));
                }
            }
        }
        Ok(AffinityMatrix {
            nodes: nodes.to_vec(),
            values,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphError(pub String);

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for GraphError {}
