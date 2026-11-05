//! Ground Boolean HTN partial-order causal-link search across root partitions.
//! Structural refinements are proposals; only protected support and global HTN
//! replay establish feasibility. Resource cutoffs never prove infeasibility.
mod order;
mod search;
mod structure;
mod verify;

use crate::{
    master::RootPartition,
    solver::{
        causal::{CausalLink, Literal},
        *,
    },
};
pub use search::{reconcile_local_certificates, solve_interleaved};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
pub use verify::validate_interleaving_certificate;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct InterleavingOptions {
    /// Counts both structural and POCL states.
    pub max_nodes: usize,
    pub max_generated: usize,
    pub max_time_ms: u64,
    pub max_events: usize,
    pub max_frontier: usize,
    pub exactness: ExactnessMode,
}
impl Default for InterleavingOptions {
    fn default() -> Self {
        Self {
            max_nodes: 100_000,
            max_generated: 500_000,
            max_time_ms: 30_000,
            max_events: 256,
            max_frontier: 1024,
            exactness: ExactnessMode::Unbounded,
        }
    }
}
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InterleavingStatistics {
    pub nodes: usize,
    pub generated: usize,
    pub structural_nodes: usize,
    pub local_refinements: usize,
    pub complete_decompositions: usize,
    pub causal_nodes: usize,
    pub support_choices: usize,
    pub cross_partition_links: usize,
    pub threat_choices: usize,
    pub condition_choices: usize,
    pub cycle_rejections: usize,
    pub unsupported_conditions: usize,
    pub depth_cutoffs: usize,
    pub resource_cutoffs: usize,
    pub peak_frontier: usize,
    pub replay_checks: usize,
    pub elapsed_ms: u128,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanStep {
    /// Local-plan owner; root occurrence ownership is inherited by descendants.
    pub partition: usize,
    pub task: GroundTask,
    /// Stable decomposition occurrence, remapped for the final linearization.
    pub decision: Decision,
    pub children: Vec<usize>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterleavingCertificate {
    pub schema_version: u32,
    pub initial_network: usize,
    pub exactness: ExactnessMode,
    pub partitions: Vec<RootPartition>,
    pub steps: Vec<PlanStep>,
    /// 0=initial boundary, event i=i+1, steps.len()+1=goal boundary.
    pub ordering: BTreeSet<(usize, usize)>,
    pub causal_links: Vec<CausalLink>,
    pub guard_witnesses: Vec<BoundaryState>,
    pub goal_witness: BoundaryState,
    pub linearization: Vec<usize>,
    pub plan: PlanCertificate,
    /// Supplied local evidence constrains method choices and internal order;
    /// its output facts are not implicitly required at the global finish.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub local_sources: Option<Vec<crate::master::LocalSegment>>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SearchScope {
    AllDecompositions,
    SuppliedLocalCertificates,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterleavingResult {
    pub status: ProofStatus,
    pub exactness: ExactnessMode,
    pub scope: SearchScope,
    pub options: InterleavingOptions,
    pub certificate: Option<InterleavingCertificate>,
    pub statistics: InterleavingStatistics,
}

#[derive(Clone)]
struct ProposedStep {
    task: GroundTask,
    partition: usize,
    depth: usize,
    /// None denotes a compound occurrence whose method has not been chosen.
    kind: Option<DecisionKind>,
    children: Vec<usize>,
}
#[derive(Clone)]
struct Proposal {
    initial: usize,
    steps: Vec<ProposedStep>,
}
struct FixedPlan {
    proposal: Proposal,
    guards: Vec<Condition>,
    add: Vec<State>,
    delete: Vec<State>,
    writers: std::collections::BTreeMap<Literal, Vec<usize>>,
    clobberers: std::collections::BTreeMap<Literal, Vec<usize>>,
    base: BTreeSet<(usize, usize)>,
}
#[derive(Clone)]
struct Obligation {
    consumer: usize,
    condition: Condition,
    positive: bool,
}
#[derive(Debug, Clone, PartialEq, Eq)]
struct Link {
    producer: usize,
    consumer: usize,
    literal: Literal,
}
#[derive(Clone)]
struct PartialPlan {
    order: order::Ordering,
    open: Vec<Obligation>,
    links: Vec<Link>,
}
