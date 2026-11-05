use crate::agglomeration::{Agglomeration, AgglomerationOptions, MergeCandidate, legal_candidates};
use crate::analysis::{HtnAnalysis, NetworkId, TaskKind};
use crate::interaction::{EdgeKind, GraphView, InteractionGraph, NodeKind};
use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const FEATURE_VERSION: u32 = 1;
pub const NODE_FEATURES: usize = 12;
pub const PAIR_FEATURES: usize = 12;
pub const GLOBAL_FEATURES: usize = 6;
pub const CONTEXT_FEATURES: usize = 9;
pub const RELATIONS: usize = 12;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterInput {
    pub id: usize,
    pub features: Vec<f64>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relation {
    pub source: usize,
    pub target: usize,
    pub channel: usize,
    pub weight: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionInput {
    /// Canonical node positions, not IDs or task schema indices. None is STOP.
    pub pair: Option<(usize, usize)>,
    pub features: Vec<f64>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hilg: Option<super::hilg::InstanceGraph>,
    pub feature_version: u32,
    pub nodes: Vec<ClusterInput>,
    pub relations: Vec<Relation>,
    pub global: Vec<f64>,
    pub context: Vec<f64>,
    pub actions: Vec<ActionInput>,
}

pub fn view_index(view: GraphView) -> usize {
    match view {
        GraphView::Ordering => 0,
        GraphView::Support => 1,
        GraphView::Threat => 2,
        GraphView::Resource => 3,
        GraphView::Hierarchy => 4,
        GraphView::SharedState => 5,
    }
}
fn signed_log(x: f64) -> f64 {
    x.signum() * x.abs().ln_1p()
}

impl Observation {
    pub fn set_search_context(&mut self, options: &crate::master::MasterOptions) {
        let (bounded, depth) = match options.local_budget.exactness {
            crate::solver::ExactnessMode::Unbounded => (0.0, 0),
            crate::solver::ExactnessMode::Bounded {
                max_decomposition_depth,
            } => (1.0, max_decomposition_depth),
        };
        self.context = vec![
            (options.max_master_nodes as f64).ln_1p(),
            (options.max_local_calls as f64).ln_1p(),
            (options.max_boundary_atoms as f64).ln_1p(),
            (options.local_budget.max_expansions as f64).ln_1p(),
            (options.local_budget.max_generated as f64).ln_1p(),
            bounded,
            (depth as f64).ln_1p(),
            (options.max_time_ms as f64).ln_1p(),
            (options.local_budget.max_time_ms as f64).ln_1p(),
        ];
    }
    pub fn validate(&self) -> Result<()> {
        if self.feature_version != FEATURE_VERSION
            || self.global.len() != GLOBAL_FEATURES
            || self.context.len() != CONTEXT_FEATURES
            || self.actions.is_empty()
        {
            bail!("incompatible observation feature schema");
        }
        if let Some(graph) = &self.hilg {
            graph.validate(self.nodes.len())?;
        }
        let mut ids = BTreeSet::new();
        for node in &self.nodes {
            if node.features.len() != NODE_FEATURES || !ids.insert(node.id) {
                bail!("invalid cluster features");
            }
        }
        let mut pairs = BTreeSet::new();
        let mut stops = 0;
        for action in &self.actions {
            if action.features.len() != PAIR_FEATURES {
                bail!("invalid action features");
            }
            if let Some((a, b)) = action.pair {
                if a >= b || b >= self.nodes.len() || !pairs.insert((a, b)) {
                    bail!("invalid or duplicate merge action");
                }
            } else {
                stops += 1;
            }
        }
        if stops != 1 {
            bail!("observation must contain exactly one stop action");
        }
        if self.relations.iter().any(|r| {
            r.source >= self.nodes.len()
                || r.target >= self.nodes.len()
                || r.source == r.target
                || r.channel >= RELATIONS
                || !r.weight.is_finite()
                || r.weight <= 0.0
        }) {
            bail!("invalid typed quotient relation");
        }
        if self
            .global
            .iter()
            .chain(&self.context)
            .chain(self.nodes.iter().flat_map(|n| &n.features))
            .chain(self.actions.iter().flat_map(|a| &a.features))
            .any(|v| !v.is_finite())
        {
            bail!("nonfinite model features");
        }
        Ok(())
    }

    pub fn build(
        analysis: &HtnAnalysis,
        graph: &InteractionGraph,
        state: &Agglomeration,
        options: &AgglomerationOptions,
    ) -> Result<(Self, Vec<MergeCandidate>)> {
        let mut candidates = legal_candidates(graph, NetworkId::Initial, state, options)?;
        candidates.sort_by_key(|c| (c.left, c.right));
        let positions: BTreeMap<_, _> = state
            .clusters
            .iter()
            .enumerate()
            .map(|(i, c)| (c.id, i))
            .collect();
        let owner: BTreeMap<_, _> = state
            .clusters
            .iter()
            .flat_map(|c| c.nodes.iter().map(|&n| (n, positions[&c.id])))
            .collect();
        let mut nodes = Vec::new();
        for cluster in &state.clusters {
            let selected: BTreeSet<_> = cluster.nodes.iter().copied().collect();
            let mut primitive = 0;
            let mut conditions = BTreeSet::new();
            let mut effects = BTreeSet::new();
            let mut branching = 0;
            for node in graph.nodes.iter().filter(|n| selected.contains(&n.id)) {
                if let NodeKind::Occurrence { task, .. } = &node.kind {
                    primitive += usize::from(analysis.tasks[task.0].kind == TaskKind::Primitive);
                    branching = branching.max(analysis.tasks[task.0].methods.len());
                }
                conditions.extend(node.summary.conditions.iter().cloned());
                effects.extend(node.summary.effects.iter().cloned());
            }
            let f = &state.features[&cluster.id];
            let size = cluster.nodes.len() as f64;
            let ordered = graph
                .edges
                .iter()
                .filter(|e| {
                    e.kind == EdgeKind::Ordering
                        && selected.contains(&e.source)
                        && selected.contains(&e.target)
                })
                .count();
            let density = if size > 1.0 {
                ordered as f64 / (size * (size - 1.0) / 2.0)
            } else {
                0.0
            };
            nodes.push(ClusterInput {
                id: cluster.id,
                features: vec![
                    size.ln_1p(),
                    primitive as f64 / size,
                    1.0 - primitive as f64 / size,
                    (f.fluent_families.len() as f64).ln_1p(),
                    (f.boundary_families.len() as f64).ln_1p(),
                    f64::from(f.fully_unordered),
                    f64::from(f.recursive),
                    (conditions.len() as f64).ln_1p(),
                    (effects.len() as f64).ln_1p(),
                    (branching as f64).ln_1p(),
                    density,
                    size / (owner.len().max(1) as f64),
                ],
            });
        }
        let mut counts = BTreeMap::<(usize, usize, usize), f64>::new();
        let mut pair_counts = BTreeMap::<(usize, usize), [f64; 6]>::new();
        let mut ordering = BTreeSet::new();
        for edge in &graph.edges {
            let (Some(&a), Some(&b)) = (owner.get(&edge.source), owner.get(&edge.target)) else {
                continue;
            };
            if a == b {
                continue;
            }
            let view = view_index(edge.kind.view());
            *counts.entry((a, b, 2 * view)).or_default() += 1.0;
            *counts.entry((b, a, 2 * view + 1)).or_default() += 1.0;
            if !edge.kind.is_directed() {
                *counts.entry((b, a, 2 * view)).or_default() += 1.0;
                *counts.entry((a, b, 2 * view + 1)).or_default() += 1.0;
            }
            pair_counts.entry((a.min(b), a.max(b))).or_default()[view] += 1.0;
            if edge.kind == EdgeKind::Ordering {
                ordering.insert((a, b));
            }
        }
        let relations = counts
            .into_iter()
            .map(|((source, target, channel), weight)| Relation {
                source,
                target,
                channel,
                weight: weight.ln_1p(),
            })
            .collect();
        let mut actions = Vec::new();
        for candidate in &candidates {
            let pair = (positions[&candidate.left], positions[&candidate.right]);
            let s = &candidate.score;
            let mut features = vec![
                s.coupling.ln_1p(),
                signed_log(s.interface_reduction),
                s.size_cost.ln_1p(),
                (s.unordered_mass_lost as f64).ln_1p(),
                f64::from(s.recursive_mix),
                signed_log(s.total),
            ];
            features.extend(
                pair_counts
                    .get(&pair)
                    .unwrap_or(&[0.0; 6])
                    .iter()
                    .map(|x| x.ln_1p()),
            );
            actions.push(ActionInput {
                pair: Some(pair),
                features,
            });
        }
        actions.push(ActionInput {
            pair: None,
            features: vec![0.0; PAIR_FEATURES],
        });
        let count = nodes.len().max(1) as f64;
        let roots = owner.len().max(1) as f64;
        let global = vec![
            (nodes.len() as f64).ln_1p(),
            (owner.len() as f64).ln_1p(),
            (state.quotient_edges.len() as f64).ln_1p(),
            (ordering.len() as f64).ln_1p(),
            state.features.values().filter(|f| f.recursive).count() as f64 / count,
            state
                .features
                .values()
                .filter(|f| f.fully_unordered)
                .map(|f| f.size)
                .sum::<usize>() as f64
                / roots,
        ];
        let mut observation = Self {
            hilg: None,
            feature_version: FEATURE_VERSION,
            nodes,
            relations,
            global,
            context: vec![0.0; CONTEXT_FEATURES],
            actions,
        };
        observation.hilg = Some(super::hilg::InstanceGraph::build(
            analysis,
            graph,
            state,
            &observation,
        )?);
        observation.validate()?;
        Ok((observation, candidates))
    }
}
