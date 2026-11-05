//! Static, lifted HTN features. These summaries are not solvability proofs.
//!
//! IDs index the corresponding vectors and are stable for a fixed AST order.
//! The original parser AST remains the source of exact formula semantics.

mod analyze;
mod formulas;
mod structure;

use std::collections::BTreeSet;
use std::fmt;

use serde::Serialize;

use crate::parsing::json::{Constraint, Symbol};

pub use analyze::analyze_program;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct TaskId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct MethodId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub enum NetworkId {
    Method(MethodId),
    Initial,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum TaskKind {
    Primitive,
    Compound,
}

/// Predicate-family access, intentionally ignoring argument bindings and types.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct Literal {
    pub predicate: String,
    pub positive: bool,
}

impl Literal {
    pub fn complement(&self) -> Self {
        Self {
            predicate: self.predicate.clone(),
            positive: !self.positive,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum FormulaFeature {
    Disjunction,
    ExclusiveDisjunction,
    Implication,
    Quantification,
    Probabilistic,
    Equality,
}

/// May-read / may-write unions over all reachable decompositions. Neither set
/// asserts necessary conditions, guaranteed effects, or feasible refinements.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct TaskSummary {
    pub conditions: BTreeSet<Literal>,
    pub effects: BTreeSet<Literal>,
    pub formula_features: BTreeSet<FormulaFeature>,
}

impl TaskSummary {
    pub(crate) fn extend(&mut self, other: &Self) {
        self.conditions.extend(other.conditions.iter().cloned());
        self.effects.extend(other.effects.iter().cloned());
        self.formula_features.extend(&other.formula_features);
    }

    pub fn fluent_families(&self) -> BTreeSet<&str> {
        self.conditions
            .iter()
            .chain(&self.effects)
            .map(|literal| literal.predicate.as_str())
            .collect()
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct TaskAnalysis {
    pub id: TaskId,
    pub name: String,
    pub kind: TaskKind,
    pub parameters: Vec<Symbol>,
    pub methods: Vec<MethodId>,
    pub summary: TaskSummary,
}

#[derive(Debug, Clone, Serialize)]
pub struct MethodAnalysis {
    pub id: MethodId,
    pub name: String,
    pub task: TaskId,
    pub parameters: Vec<Symbol>,
    pub task_terms: Vec<Symbol>,
    pub network: NetworkAnalysis,
    pub summary: TaskSummary,
}

#[derive(Debug, Clone, Serialize)]
pub struct TaskOccurrence {
    /// Index within this network; independent of the optional HDDL label.
    pub index: usize,
    pub label: Option<String>,
    pub task: TaskId,
    pub terms: Vec<Symbol>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct Precedence {
    pub before: usize,
    pub after: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum OrderingClass {
    /// No nontrivial precedence; includes empty and singleton networks.
    Unordered,
    Total,
    Partial,
}

#[derive(Debug, Clone, Serialize)]
pub struct NetworkAnalysis {
    pub id: NetworkId,
    pub parameters: Vec<Symbol>,
    pub occurrences: Vec<TaskOccurrence>,
    /// Input edges, deduplicated; total-order inputs become consecutive edges.
    pub precedence: Vec<Precedence>,
    pub transitive_precedence: Vec<Precedence>,
    pub ordering: OrderingClass,
    pub constraints: Vec<Constraint>,
}

impl NetworkAnalysis {
    pub fn precedes(&self, before: usize, after: usize) -> bool {
        self.transitive_precedence
            .binary_search(&Precedence { before, after })
            .is_ok()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum RecursionClass {
    Acyclic,
    /// Each recursive method has at most one same-SCC call, after every peer.
    TailRecursive,
    GeneralRecursive,
}

#[derive(Debug, Clone, Serialize)]
pub struct RecursionGroup {
    pub tasks: Vec<TaskId>,
    pub recursive: bool,
    pub tail_recursive: bool,
}

/// Syntactic features over problem-reachable schemas (all schemas for domain-
/// only analysis). No formal complexity bound is assigned by this scaffold.
#[derive(Debug, Clone, Serialize)]
pub struct ComplexitySignature {
    pub recursion: RecursionClass,
    pub ordering: OrderingClass,
    /// Reachable schema/network parameters or variable terms remain lifted,
    /// even when the problem's initial task arguments happen to be ground.
    pub has_lifted_variables: bool,
    pub fully_unordered: bool,
    pub reachable_tasks: usize,
    pub reachable_methods: usize,
    pub largest_recursive_scc: usize,
    pub max_method_branching: usize,
    pub max_network_size: usize,
    pub ordered_pairs: usize,
    pub possible_ordered_pairs: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct HtnAnalysis {
    /// Original instance for instance-aware, lifted learning graphs; omitted from reports.
    #[serde(skip)]
    pub instance: Option<crate::parsing::json::Program>,
    pub domain_name: String,
    pub problem_name: Option<String>,
    pub tasks: Vec<TaskAnalysis>,
    pub methods: Vec<MethodAnalysis>,
    pub initial_network: Option<NetworkAnalysis>,
    pub reachable_tasks: BTreeSet<TaskId>,
    pub recursion_groups: Vec<RecursionGroup>,
    pub signature: ComplexitySignature,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnalysisError(pub String);

impl fmt::Display for AnalysisError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for AnalysisError {}
