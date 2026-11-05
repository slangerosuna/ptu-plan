//! Exact progression search for finite, typed, deterministic Boolean HDDL.
//!
//! Only enabled tasks can execute or refine. Method preconditions are evaluated
//! at refinement time. Concrete states use the closed-world convention; an
//! unknown boundary state must be resolved by a master before calling this API.

pub mod causal;
mod ground;
mod search;

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use serde::{Deserialize, Serialize};

pub use ground::{ground_program, ground_reachable_program};
pub(crate) use search::sample_structure;
pub use search::{
    certify_decisions, solve_local, solve_problem, solve_with_plan, trace_certificate,
    validate_certificate, validate_problem_certificate,
};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Deserialize, Serialize)]
pub struct GroundAtom {
    pub predicate: String,
    pub arguments: Vec<String>,
}

pub type State = BTreeSet<GroundAtom>;
pub type Bindings = BTreeMap<String, String>;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Deserialize, Serialize)]
pub struct GroundTask {
    pub name: String,
    pub arguments: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub enum Condition {
    True,
    False,
    Atom(GroundAtom),
    Not(Box<Condition>),
    And(Vec<Condition>),
    Or(Vec<Condition>),
}

impl Condition {
    pub fn holds(&self, state: &State) -> bool {
        match self {
            Self::True => true,
            Self::False => false,
            Self::Atom(atom) => state.contains(atom),
            Self::Not(inner) => !inner.holds(state),
            Self::And(terms) => terms.iter().all(|term| term.holds(state)),
            Self::Or(terms) => terms.iter().any(|term| term.holds(state)),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize)]
pub struct BoundaryState {
    pub must_be_true: State,
    pub must_be_false: State,
}

impl BoundaryState {
    pub fn condition(&self) -> Result<Condition, SolverError> {
        if !self.must_be_true.is_disjoint(&self.must_be_false) {
            return Err(SolverError("contradictory boundary literals".into()));
        }
        Ok(Condition::And(
            self.must_be_true
                .iter()
                .cloned()
                .map(Condition::Atom)
                .chain(
                    self.must_be_false
                        .iter()
                        .cloned()
                        .map(|atom| Condition::Not(Box::new(Condition::Atom(atom)))),
                )
                .collect(),
        ))
    }
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct GroundNetwork {
    pub tasks: Vec<GroundTask>,
    pub precedence: BTreeSet<(usize, usize)>,
}

impl GroundNetwork {
    /// Induce a local network from root indices, retaining precedence paths
    /// through omitted roots. This is not a proof that partitions compose.
    pub fn induced(&self, roots: &[usize]) -> Result<Self, SolverError> {
        let mut selected = BTreeMap::new();
        for (new, &old) in roots.iter().enumerate() {
            if old >= self.tasks.len() || selected.insert(old, new).is_some() {
                return Err(SolverError("invalid or duplicate partition root".into()));
            }
        }
        let closure = network_closure(self)?;
        let mut precedence = BTreeSet::new();
        for (&before, &left) in &selected {
            for (&after, &right) in &selected {
                if closure[before][after] {
                    precedence.insert((left, right));
                }
            }
        }
        Ok(Self {
            tasks: roots.iter().map(|&id| self.tasks[id].clone()).collect(),
            precedence,
        })
    }
}

pub(crate) fn network_closure(network: &GroundNetwork) -> Result<Vec<Vec<bool>>, SolverError> {
    let n = network.tasks.len();
    let mut closure = vec![vec![false; n]; n];
    for &(before, after) in &network.precedence {
        if before >= n || after >= n {
            return Err(SolverError(
                "task ordering endpoint is outside the network".into(),
            ));
        }
        closure[before][after] = true;
    }
    for via in 0..n {
        for from in 0..n {
            if closure[from][via] {
                for to in 0..n {
                    closure[from][to] |= closure[via][to];
                }
            }
        }
    }
    if (0..n).any(|i| closure[i][i]) {
        return Err(SolverError("cyclic task ordering".into()));
    }
    Ok(closure)
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GroundAction {
    pub task: GroundTask,
    pub precondition: Condition,
    pub add: State,
    pub delete: State,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GroundMethod {
    pub id: usize,
    pub name: String,
    pub bindings: Bindings,
    pub task: GroundTask,
    pub precondition: Condition,
    pub network: GroundNetwork,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct InitialNetwork {
    pub bindings: Bindings,
    pub network: GroundNetwork,
}

/// Fully compiled model. Create through `ground_program`; the search never
/// uses the analyzer's approximate summaries as symbolic semantics.
#[derive(Debug, Clone)]
pub struct GroundProblem {
    pub actions: BTreeMap<GroundTask, GroundAction>,
    pub methods: Vec<GroundMethod>,
    pub initial_networks: Vec<InitialNetwork>,
    pub initial_state: State,
    pub goal: Condition,
    pub(crate) tasks: BTreeSet<GroundTask>,
    pub(crate) methods_by_task: BTreeMap<GroundTask, Vec<usize>>,
    pub(crate) context: ground::Context,
    pub grounding_mode: GroundingMode,
}

/// Grounded method IDs depend on this strategy. Persist it with certificates.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
pub enum GroundingMode {
    #[default]
    Full,
    InitialReachable,
}

impl GroundingMode {
    pub fn is_full(&self) -> bool {
        *self == Self::Full
    }
}

#[derive(Debug, Clone, Copy)]
pub struct GroundingOptions {
    /// Counts task, method, initial-network, and formula expansion work.
    pub max_instances: usize,
}

impl Default for GroundingOptions {
    fn default() -> Self {
        Self {
            max_instances: 100_000,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LocalRequest {
    pub network: GroundNetwork,
    /// Complete concrete state, not a partial incoming contract.
    pub initial_state: State,
    pub goal: Condition,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum ExactnessMode {
    Unbounded,
    /// Root depth is zero; a compound at the bound cannot refine.
    Bounded {
        max_decomposition_depth: usize,
    },
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
pub struct SearchBudget {
    pub max_expansions: usize,
    pub max_generated: usize,
    pub max_time_ms: u64,
    pub exactness: ExactnessMode,
}

impl Default for SearchBudget {
    fn default() -> Self {
        Self {
            max_expansions: 100_000,
            max_generated: 500_000,
            max_time_ms: 5_000,
            exactness: ExactnessMode::Unbounded,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum ProofStatus {
    Feasible,
    Infeasible,
    InfeasibleWithinBound,
    BudgetExhausted,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct SearchStatistics {
    pub expanded: usize,
    pub generated: usize,
    pub duplicates: usize,
    pub action_applications: usize,
    pub method_refinements: usize,
    pub depth_cutoffs: usize,
    pub dead_ends: usize,
    pub guard_rejections: usize,
    pub ordering_blocked: usize,
    pub peak_frontier: usize,
    pub peak_pending: usize,
    pub max_decomposition_depth: usize,
    pub elapsed_ms: u128,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub enum DecisionKind {
    Execute,
    Refine { method: usize },
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct Decision {
    /// Stable occurrence ID within this certificate, including refinements.
    pub occurrence: usize,
    pub kind: DecisionKind,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PlanCertificate {
    pub initial_network: usize,
    /// Refinements identify grounded methods and their bindings in the model.
    pub decisions: Vec<Decision>,
    pub actions: Vec<GroundTask>,
    pub final_state: State,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ExecutionStep {
    pub decision: Decision,
    pub task: GroundTask,
    pub before: State,
    pub after: State,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SolveResult {
    pub status: ProofStatus,
    pub exactness: ExactnessMode,
    pub certificate: Option<PlanCertificate>,
    pub statistics: SearchStatistics,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SolverError(pub String);

impl fmt::Display for SolverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for SolverError {}
