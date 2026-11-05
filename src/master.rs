//! Sequential partial-state master. Each partition executes as one contiguous
//! block. Exhaustive failure is scoped to that restriction, never global HTN
//! infeasibility. Only proven local failures create contextual no-goods.

use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use crate::conflicts::{
    CoreIndex, VerifiedConflictCore, after_forbidden_cube, conjunction,
    incoming_projection_from_reads,
};
use crate::instrumentation::{InterfaceTrainingContext, MasterEvent, OraclePurpose, TraceLevel};
use serde::{Deserialize, Serialize};

use crate::analysis::NetworkId;
use crate::interaction::{InteractionGraph, NodeKind};
use crate::partition::MicroPartition;
use crate::solver::{
    BoundaryState, Condition, Decision, DecisionKind, ExactnessMode, GroundNetwork, GroundProblem,
    GroundTask, LocalRequest, PlanCertificate, ProofStatus, SearchBudget, SolverError, State,
    certify_decisions, network_closure, solve_local, validate_problem_certificate,
};

pub struct InterfaceQuery<'a> {
    pub problem: &'a GroundProblem,
    pub network: &'a GroundNetwork,
    pub incoming: &'a State,
    pub boundary: &'a [crate::solver::GroundAtom],
    pub future_reads: &'a State,
}

/// A proposal only chooses the first exhaustive valuation. Omitted atoms keep
/// their incoming truth value. Predictions never create cuts.
pub trait InterfaceGuide {
    fn propose(&self, query: &InterfaceQuery<'_>) -> Result<BoundaryState, SolverError>;
}

#[derive(Default)]
pub struct MasterGuidance<'a> {
    pub interface: Option<&'a dyn InterfaceGuide>,
    /// Maximum deletion checks per proven local failure. Zero disables extraction.
    pub conflict_core_checks: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct RootPartition {
    pub id: usize,
    /// Indices into the original initial task network, not schema IDs.
    pub roots: Vec<usize>,
}

pub fn initial_partitions(
    graph: &InteractionGraph,
    clusters: &[MicroPartition],
) -> Result<Vec<RootPartition>, SolverError> {
    clusters
        .iter()
        .map(|cluster| {
            let mut roots = Vec::new();
            for id in &cluster.nodes {
                match graph.nodes.iter().find(|n| n.id == *id).map(|n| &n.kind) {
                    Some(NodeKind::Occurrence {
                        network: NetworkId::Initial,
                        index,
                        ..
                    }) => roots.push(*index),
                    _ => {
                        return Err(SolverError(
                            "master partitions require initial-network occurrences".into(),
                        ));
                    }
                }
            }
            roots.sort();
            Ok(RootPartition {
                id: cluster.id,
                roots,
            })
        })
        .collect()
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
pub struct MasterOptions {
    pub max_master_nodes: usize,
    pub max_local_calls: usize,
    /// Exhaustive enumeration is exponential in this number. Exceeding it is
    /// an unknown result and does not justify any no-good.
    pub max_boundary_atoms: usize,
    pub max_time_ms: u64,
    pub local_budget: SearchBudget,
}

impl Default for MasterOptions {
    fn default() -> Self {
        Self {
            max_master_nodes: 10_000,
            max_local_calls: 1_000,
            max_boundary_atoms: 12,
            max_time_ms: 30_000,
            local_budget: SearchBudget::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum MasterStatus {
    Feasible,
    /// Exhaustive search of all block orders and contracts for these partitions.
    SequentiallyInfeasible,
    InfeasibleWithinBound,
    BudgetExhausted,
    NotSequentiallyComposable,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LocalSegment {
    pub partition: RootPartition,
    /// Explicit true/false projection onto local reads. Unlisted facts remain
    /// unknown in this contract; the oracle receives the complete witness state.
    pub incoming: BoundaryState,
    pub outgoing: BoundaryState,
    pub request: LocalRequest,
    pub certificate: PlanCertificate,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub causal: Option<crate::solver::causal::CausalContract>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct VerifiedNogood {
    pub initial_network: usize,
    pub partition: usize,
    pub initial_state: State,
    pub goal: Condition,
    /// No generalization to different incoming worlds or omitted literals.
    pub exactness: ExactnessMode,
    pub proof: ProofStatus,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct MasterStatistics {
    pub master_nodes: usize,
    pub local_calls: usize,
    pub interface_candidates: usize,
    pub nogood_hits: usize,
    pub local_expanded: usize,
    pub local_generated: usize,
    pub local_unknown: usize,
    pub boundary_cutoffs: usize,
    pub suffix_cache_hits: usize,
    pub max_boundary_atoms: usize,
    pub oracle_feasible: usize,
    pub oracle_infeasible: usize,
    pub oracle_bounded: usize,
    pub core_checks: usize,
    pub core_hits: usize,
    pub core_generalized_hits: usize,
    pub core_assignments_skipped: usize,
    pub core_active: usize,
    pub core_subsumed: usize,
    pub core_redundant: usize,
    pub core_incoming_literals: usize,
    pub core_literals_removed: usize,
    pub interface_proposals: usize,
    pub interface_inference_us: u128,
    pub causal_certificates: usize,
    pub causal_links: usize,
    pub causal_imports: usize,
    pub causal_threats: usize,
    pub causal_time_us: u128,
    pub elapsed_ms: u128,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MasterResult {
    pub status: MasterStatus,
    pub exactness: ExactnessMode,
    pub certificate: Option<PlanCertificate>,
    pub segments: Vec<LocalSegment>,
    pub nogoods: Vec<VerifiedNogood>,
    #[serde(default)]
    pub conflict_cores: Vec<VerifiedConflictCore>,
    pub statistics: MasterStatistics,
}

#[derive(Default)]
pub(crate) struct Footprint {
    pub(crate) reads: State,
    pub(crate) writes: State,
}

/// Enumerate block topological orders and relevant boundary valuations. A single
/// local witness per valuation suffices: all facts read by any future action,
/// refinement guard, or final goal are retained, and unwritten facts are framed
/// by the concrete incoming world. This argument uses exact grounded atoms,
/// never predicate-family interaction summaries.
pub fn solve_sequential(
    problem: &GroundProblem,
    partitions: &[RootPartition],
    options: &MasterOptions,
) -> Result<MasterResult, SolverError> {
    run_master(
        problem,
        partitions,
        options,
        &MasterGuidance::default(),
        None,
        TraceLevel::Summary,
    )
}

pub fn solve_sequential_observed(
    problem: &GroundProblem,
    partitions: &[RootPartition],
    options: &MasterOptions,
    level: TraceLevel,
    observer: &mut dyn FnMut(MasterEvent),
) -> Result<MasterResult, SolverError> {
    run_master(
        problem,
        partitions,
        options,
        &MasterGuidance::default(),
        Some(observer),
        level,
    )
}

pub fn solve_sequential_guided<'a>(
    problem: &'a GroundProblem,
    partitions: &'a [RootPartition],
    options: &'a MasterOptions,
    guidance: &'a MasterGuidance<'a>,
    level: TraceLevel,
    observer: Option<&'a mut dyn FnMut(MasterEvent)>,
) -> Result<MasterResult, SolverError> {
    run_master(problem, partitions, options, guidance, observer, level)
}

fn run_master<'a>(
    problem: &'a GroundProblem,
    partitions: &'a [RootPartition],
    options: &'a MasterOptions,
    guidance: &'a MasterGuidance<'a>,
    observer: Option<&'a mut dyn FnMut(MasterEvent)>,
    level: TraceLevel,
) -> Result<MasterResult, SolverError> {
    let mut ctx = Master {
        problem,
        partitions,
        options,
        guidance,
        start: Instant::now(),
        initial: 0,
        footprints: Vec::new(),
        goal_atoms: State::new(),
        predecessors: Vec::new(),
        result: MasterResult {
            status: MasterStatus::SequentiallyInfeasible,
            exactness: options.local_budget.exactness,
            certificate: None,
            segments: Vec::new(),
            nogoods: Vec::new(),
            conflict_cores: Vec::new(),
            statistics: MasterStatistics::default(),
        },
        unknown: false,
        bounded: false,
        stopped: false,
        uncertain_branches: 0,
        infeasible_suffixes: BTreeSet::new(),
        core_index: CoreIndex::default(),
        observer,
        trace_level: level,
    };
    atoms(&problem.goal, &mut ctx.goal_atoms);
    let mut cyclic = false;
    for (initial, network) in problem.initial_networks.iter().enumerate() {
        ctx.initial = initial;
        if let Some(observer) = &mut ctx.observer {
            observer(MasterEvent::InitialNetwork {
                initial_network: initial,
                partitions: partitions.to_vec(),
            });
        }
        ctx.infeasible_suffixes.clear();
        ctx.predecessors = partition_order(&network.network, partitions)?;
        if cyclic_order(&ctx.predecessors) {
            cyclic = true;
            continue;
        }
        ctx.footprints = partitions
            .iter()
            .map(|p| footprint(problem, &network.network, p))
            .collect();
        let remaining = (0..partitions.len()).collect();
        if let Some(segments) = ctx.search(&network.network, &remaining, &problem.initial_state)? {
            let certificate = compose(problem, &network.network, initial, &segments)?;
            ctx.result.status = MasterStatus::Feasible;
            ctx.result.certificate = Some(certificate);
            ctx.result.segments = segments;
            break;
        }
        if ctx.stopped {
            break;
        }
    }
    if ctx.result.certificate.is_none() {
        ctx.result.status = if ctx.unknown {
            MasterStatus::BudgetExhausted
        } else if ctx.bounded {
            MasterStatus::InfeasibleWithinBound
        } else if cyclic {
            MasterStatus::NotSequentiallyComposable
        } else {
            MasterStatus::SequentiallyInfeasible
        };
    }
    ctx.result.statistics.elapsed_ms = ctx.start.elapsed().as_millis();
    if let Some(observer) = &mut ctx.observer {
        observer(MasterEvent::Finished {
            status: ctx.result.status,
            statistics: ctx.result.statistics.clone(),
        });
    }
    Ok(ctx.result)
}

struct Master<'a> {
    core_index: CoreIndex,
    guidance: &'a MasterGuidance<'a>,
    observer: Option<&'a mut dyn FnMut(MasterEvent)>,
    trace_level: TraceLevel,
    problem: &'a GroundProblem,
    partitions: &'a [RootPartition],
    options: &'a MasterOptions,
    start: Instant,
    initial: usize,
    footprints: Vec<Footprint>,
    goal_atoms: State,
    predecessors: Vec<BTreeSet<usize>>,
    result: MasterResult,
    unknown: bool,
    bounded: bool,
    stopped: bool,
    uncertain_branches: usize,
    /// Exact proofs only, keyed by remaining blocks and the complete world.
    /// Cleared for each initial grounding; bounded/unknown failures never enter.
    infeasible_suffixes: BTreeSet<(BTreeSet<usize>, State)>,
}

impl Master<'_> {
    fn oracle(
        &mut self,
        partition: usize,
        request: &LocalRequest,
        purpose: OraclePurpose,
        interface_context: Option<InterfaceTrainingContext>,
    ) -> Result<Option<(usize, crate::solver::SolveResult)>, SolverError> {
        if self.result.statistics.local_calls >= self.options.max_local_calls
            || self.time_left() == 0
        {
            return Ok(None);
        }
        let mut budget = self.options.local_budget;
        budget.max_time_ms = budget.max_time_ms.min(self.time_left());
        let call = self.result.statistics.local_calls;
        if let Some(observer) = &mut self.observer {
            observer(MasterEvent::OracleStarted {
                call,
                initial_network: self.initial,
                partition,
                budget,
                request: (self.trace_level == TraceLevel::Full).then(|| request.clone()),
                purpose,
                interface_context,
            });
        }
        let local = solve_local(self.problem, request, &budget)?;
        if let Some(observer) = &mut self.observer {
            observer(MasterEvent::OracleFinished {
                call,
                status: local.status,
                statistics: local.statistics.clone(),
                certificate: if self.trace_level == TraceLevel::Full {
                    local.certificate.clone()
                } else {
                    None
                },
            });
        }
        let stats = &mut self.result.statistics;
        stats.local_calls += 1;
        stats.core_checks += usize::from(purpose == OraclePurpose::CoreMinimization);
        stats.local_expanded += local.statistics.expanded;
        stats.local_generated += local.statistics.generated;
        stats.oracle_feasible += usize::from(local.status == ProofStatus::Feasible);
        stats.oracle_infeasible += usize::from(local.status == ProofStatus::Infeasible);
        stats.oracle_bounded += usize::from(local.status == ProofStatus::InfeasibleWithinBound);
        stats.local_unknown += usize::from(local.status == ProofStatus::BudgetExhausted);
        Ok(Some((call, local)))
    }

    fn minimize_core(
        &mut self,
        partition_index: usize,
        request: &LocalRequest,
        source_call: usize,
        source_proof: ProofStatus,
        minimize_outgoing: bool,
    ) -> Result<(), SolverError> {
        if self.guidance.conflict_core_checks == 0 {
            return Ok(());
        }
        let partition = self.partitions[partition_index].id;
        let Some(mut core) = conjunction(&request.goal) else {
            return Ok(());
        };
        let literals: Vec<_> = core
            .must_be_true
            .iter()
            .map(|a| (a.clone(), true))
            .chain(core.must_be_false.iter().map(|a| (a.clone(), false)))
            .collect();
        let mut proof = source_proof;
        let mut verification_call = source_call;
        let mut removed = 0;
        let mut remaining_checks = if minimize_outgoing {
            self.guidance.conflict_core_checks
        } else {
            0
        };
        // A blocked network often has an empty core. Test it once instead of
        // spending a deletion call on every irrelevant outgoing literal.
        if literals.len() > 1 && remaining_checks > 0 {
            let trial = LocalRequest {
                goal: Condition::True,
                ..request.clone()
            };
            if let Some((call, local)) =
                self.oracle(partition, &trial, OraclePurpose::CoreMinimization, None)?
            {
                remaining_checks -= 1;
                if matches!(
                    local.status,
                    ProofStatus::Infeasible | ProofStatus::InfeasibleWithinBound
                ) {
                    core = BoundaryState::default();
                    proof = local.status;
                    verification_call = call;
                    removed = literals.len();
                }
            } else {
                remaining_checks = 0;
            }
        }
        for (atom, positive) in
            literals
                .into_iter()
                .take(if removed == 0 { remaining_checks } else { 0 })
        {
            let mut trial = core.clone();
            if positive {
                trial.must_be_true.remove(&atom);
            } else {
                trial.must_be_false.remove(&atom);
            }
            let trial_request = LocalRequest {
                goal: trial.condition()?,
                ..request.clone()
            };
            let Some((call, local)) = self.oracle(
                partition,
                &trial_request,
                OraclePurpose::CoreMinimization,
                None,
            )?
            else {
                break;
            };
            if matches!(
                local.status,
                ProofStatus::Infeasible | ProofStatus::InfeasibleWithinBound
            ) {
                core = trial;
                proof = local.status;
                verification_call = call;
                removed += 1;
            }
        }
        {
            // This projection fixes all actual local dependencies, including
            // method guards and retained goal/frame atoms. Other world facts
            // cannot affect local plan existence under this proof bound.
            let incoming = incoming_projection_from_reads(
                &self.footprints[partition_index].reads,
                &request.initial_state,
                &core,
            );
            let cut = VerifiedConflictCore {
                initial_network: self.initial,
                partition,
                initial_state: request.initial_state.clone(),
                incoming: Some(incoming),
                source_goal: request.goal.clone(),
                core,
                exactness: self.options.local_budget.exactness,
                proof,
                source_call,
                verification_call,
                removed_literals: removed,
            };
            cut.validate_scope()?;
            self.result.statistics.core_literals_removed += removed;
            self.result.statistics.core_incoming_literals += cut
                .incoming
                .as_ref()
                .map_or(0, |b| b.must_be_true.len() + b.must_be_false.len());
            if let Some(observer) = &mut self.observer {
                observer(MasterEvent::ConflictCoreVerified { core: cut.clone() });
            }
            self.result.conflict_cores.push(cut);
            let (inserted, retired) = self.core_index.insert(
                &self.result.conflict_cores,
                self.result.conflict_cores.len() - 1,
            );
            self.result.statistics.core_active = self.core_index.active;
            self.result.statistics.core_subsumed += retired;
            self.result.statistics.core_redundant += usize::from(!inserted);
        }
        Ok(())
    }

    fn time_left(&self) -> u64 {
        self.options
            .max_time_ms
            .saturating_sub(self.start.elapsed().as_millis().min(u64::MAX as u128) as u64)
    }

    fn search(
        &mut self,
        network: &GroundNetwork,
        remaining: &BTreeSet<usize>,
        state: &State,
    ) -> Result<Option<Vec<LocalSegment>>, SolverError> {
        if self.stopped {
            return Ok(None);
        }
        if self
            .infeasible_suffixes
            .contains(&(remaining.clone(), state.clone()))
        {
            self.result.statistics.suffix_cache_hits += 1;
            return Ok(None);
        }
        let uncertain_before = self.uncertain_branches;
        if self.time_left() == 0
            || self.result.statistics.master_nodes >= self.options.max_master_nodes
        {
            self.unknown = true;
            self.uncertain_branches += 1;
            self.stopped = true;
            return Ok(None);
        }
        self.result.statistics.master_nodes += 1;
        if remaining.is_empty() {
            return Ok(self.problem.goal.holds(state).then(Vec::new));
        }
        // Stable input order breaks ties; explore other available blocks when
        // guards fail, including a producer that follows a consumer in the AST.
        let choices: Vec<_> = remaining
            .iter()
            .copied()
            .filter(|&p| self.predecessors[p].is_disjoint(remaining))
            .collect();
        for index in choices {
            let mut suffix = remaining.clone();
            suffix.remove(&index);
            let mut future_reads = self.goal_atoms.clone();
            for &other in &suffix {
                future_reads.extend(self.footprints[other].reads.iter().cloned());
            }
            let boundary: Vec<_> = if suffix.is_empty() {
                Vec::new()
            } else {
                self.footprints[index]
                    .writes
                    .intersection(&future_reads)
                    .cloned()
                    .collect()
            };
            self.result.statistics.max_boundary_atoms = self
                .result
                .statistics
                .max_boundary_atoms
                .max(boundary.len());
            if boundary.len() > self.options.max_boundary_atoms
                || boundary.len() >= usize::BITS as usize
            {
                self.unknown = true;
                self.uncertain_branches += 1;
                self.result.statistics.boundary_cutoffs += 1;
                if let Some(observer) = &mut self.observer {
                    observer(MasterEvent::BoundaryLimit {
                        initial_network: self.initial,
                        partition: self.partitions[index].id,
                        atoms: boundary.len(),
                    });
                }
                continue;
            }
            let request_network = network.induced(&self.partitions[index].roots)?;
            let incoming = projection(state, &self.footprints[index].reads);
            let mut preferred_mask = 0usize;
            if !boundary.is_empty()
                && let Some(guide) = self.guidance.interface
            {
                let start = Instant::now();
                let proposal = guide.propose(&InterfaceQuery {
                    problem: self.problem,
                    network: &request_network,
                    incoming: state,
                    boundary: &boundary,
                    future_reads: &future_reads,
                })?;
                proposal.condition()?;
                let allowed: State = boundary.iter().cloned().collect();
                if !proposal.must_be_true.is_subset(&allowed)
                    || !proposal.must_be_false.is_subset(&allowed)
                {
                    return Err(SolverError(
                        "interface proposal contains a non-boundary atom".into(),
                    ));
                }
                for (bit, atom) in boundary.iter().enumerate() {
                    let desired = if proposal.must_be_true.contains(atom) {
                        true
                    } else if proposal.must_be_false.contains(atom) {
                        false
                    } else {
                        state.contains(atom)
                    };
                    if desired != state.contains(atom) {
                        preferred_mask |= 1 << bit;
                    }
                }
                let inference_time_us = start.elapsed().as_micros();
                self.result.statistics.interface_proposals += 1;
                self.result.statistics.interface_inference_us += inference_time_us;
                if let Some(observer) = &mut self.observer {
                    observer(MasterEvent::InterfaceProposed {
                        initial_network: self.initial,
                        partition: self.partitions[index].id,
                        proposal,
                        inference_time_us,
                    });
                }
            }
            let mut ordinal = 0usize;
            while ordinal < (1usize << boundary.len()) {
                let current_ordinal = ordinal;
                ordinal += 1;
                let mask = current_ordinal ^ preferred_mask;
                if self.stopped {
                    return Ok(None);
                }
                if self.time_left() == 0 {
                    self.stopped = true;
                    self.unknown = true;
                    self.uncertain_branches += 1;
                    return Ok(None);
                }
                self.result.statistics.interface_candidates += 1;
                // Current valuation first; then deterministic bit-flip order.
                let mut outgoing = BoundaryState::default();
                for (bit, atom) in boundary.iter().enumerate() {
                    if state.contains(atom) ^ (mask & (1 << bit) != 0) {
                        outgoing.must_be_true.insert(atom.clone());
                    } else {
                        outgoing.must_be_false.insert(atom.clone());
                    }
                }
                let goal = if suffix.is_empty() {
                    self.problem.goal.clone()
                } else {
                    outgoing.condition()?
                };
                if let Some(id) = self.core_index.matching(
                    &self.result.conflict_cores,
                    self.initial,
                    self.partitions[index].id,
                    state,
                    &goal,
                    self.options.local_budget.exactness,
                ) {
                    let cut = &self.result.conflict_cores[id];
                    self.bounded |= cut.proof == ProofStatus::InfeasibleWithinBound;
                    self.uncertain_branches +=
                        usize::from(cut.proof == ProofStatus::InfeasibleWithinBound);
                    self.result.statistics.core_hits += 1;
                    self.result.statistics.core_generalized_hits +=
                        usize::from(cut.initial_state != *state);
                    ordinal = after_forbidden_cube(current_ordinal, &boundary, &cut.core);
                    self.result.statistics.core_assignments_skipped = self
                        .result
                        .statistics
                        .core_assignments_skipped
                        .saturating_add(ordinal - current_ordinal);
                    if let Some(observer) = &mut self.observer {
                        let full = self.trace_level == TraceLevel::Full;
                        observer(MasterEvent::ConflictCoreApplied {
                            core_index: id,
                            initial_network: self.initial,
                            partition: self.partitions[index].id,
                            initial_state: full.then(|| state.clone()),
                            goal: full.then(|| goal.clone()),
                            boundary: full.then(|| boundary.clone()),
                            ordinal: current_ordinal,
                            next_ordinal: ordinal,
                        });
                    }
                    continue;
                }
                if let Some(cut) = self.result.nogoods.iter().find(|cut| {
                    cut.initial_network == self.initial
                        && cut.partition == self.partitions[index].id
                        && cut.initial_state == *state
                        && cut.goal == goal
                        && cut.exactness == self.options.local_budget.exactness
                }) {
                    self.bounded |= cut.proof == ProofStatus::InfeasibleWithinBound;
                    self.uncertain_branches +=
                        usize::from(cut.proof == ProofStatus::InfeasibleWithinBound);
                    self.result.statistics.nogood_hits += 1;
                    continue;
                }
                if self.result.statistics.local_calls >= self.options.max_local_calls {
                    self.unknown = true;
                    self.uncertain_branches += 1;
                    self.stopped = true;
                    return Ok(None);
                }
                let request = LocalRequest {
                    network: request_network.clone(),
                    initial_state: state.clone(),
                    goal,
                };
                let Some((call, local)) = self.oracle(
                    self.partitions[index].id,
                    &request,
                    OraclePurpose::Solve,
                    (self.trace_level == TraceLevel::Full && self.observer.is_some()).then(|| {
                        InterfaceTrainingContext {
                            boundary: boundary.clone(),
                            future_reads: future_reads.clone(),
                        }
                    }),
                )?
                else {
                    self.unknown = true;
                    self.uncertain_branches += 1;
                    self.stopped = true;
                    return Ok(None);
                };
                match local.status {
                    ProofStatus::Feasible => {
                        let certificate = local.certificate.ok_or_else(|| {
                            SolverError("feasible oracle result lacks certificate".into())
                        })?;
                        if let Some(mut rest) =
                            self.search(network, &suffix, &certificate.final_state)?
                        {
                            let started = Instant::now();
                            let causal = crate::solver::causal::causal_contract(
                                self.problem,
                                &request,
                                &certificate,
                                self.partitions[index].id,
                            )?;
                            let stats = &mut self.result.statistics;
                            stats.causal_time_us += started.elapsed().as_micros();
                            stats.causal_certificates += 1;
                            stats.causal_links += causal.causal_links.len();
                            stats.causal_imports += causal.imports.len();
                            stats.causal_threats += causal.threat_resolutions.len();
                            rest.insert(
                                0,
                                LocalSegment {
                                    partition: self.partitions[index].clone(),
                                    incoming: incoming.clone(),
                                    outgoing,
                                    causal: Some(causal),
                                    request,
                                    certificate,
                                },
                            );
                            return Ok(Some(rest));
                        }
                    }
                    ProofStatus::Infeasible | ProofStatus::InfeasibleWithinBound => {
                        self.bounded |= local.status == ProofStatus::InfeasibleWithinBound;
                        self.uncertain_branches +=
                            usize::from(local.status == ProofStatus::InfeasibleWithinBound);
                        self.result.nogoods.push(VerifiedNogood {
                            initial_network: self.initial,
                            partition: self.partitions[index].id,
                            initial_state: state.clone(),
                            goal: request.goal.clone(),
                            exactness: local.exactness,
                            proof: local.status,
                        });
                        if let Some(observer) = &mut self.observer {
                            observer(MasterEvent::CutVerified {
                                call,
                                nogood: self.result.nogoods.last().unwrap().clone(),
                            });
                        }
                        // Final blocks retain their goal but can still reuse a
                        // dependency-scoped incoming cut across other worlds.
                        self.minimize_core(
                            index,
                            &request,
                            call,
                            local.status,
                            !suffix.is_empty(),
                        )?;
                    }
                    ProofStatus::BudgetExhausted => {
                        self.unknown = true;
                        self.uncertain_branches += 1;
                    }
                }
            }
        }
        if self.uncertain_branches == uncertain_before {
            self.infeasible_suffixes
                .insert((remaining.clone(), state.clone()));
        }
        Ok(None)
    }
}

pub(crate) fn atoms(condition: &Condition, output: &mut State) {
    match condition {
        Condition::Atom(atom) => {
            output.insert(atom.clone());
        }
        Condition::Not(inner) => atoms(inner, output),
        Condition::And(terms) | Condition::Or(terms) => {
            for term in terms {
                atoms(term, output);
            }
        }
        Condition::True | Condition::False => {}
    }
}

pub(crate) fn footprint(
    problem: &GroundProblem,
    network: &GroundNetwork,
    partition: &RootPartition,
) -> Footprint {
    footprint_tasks(
        problem,
        partition
            .roots
            .iter()
            .map(|&i| network.tasks[i].clone())
            .collect(),
    )
}

pub(crate) fn network_footprint(problem: &GroundProblem, network: &GroundNetwork) -> Footprint {
    footprint_tasks(problem, network.tasks.clone())
}

fn footprint_tasks(problem: &GroundProblem, mut pending: Vec<GroundTask>) -> Footprint {
    let mut result = Footprint::default();
    let mut seen = BTreeSet::new();
    while let Some(task) = pending.pop() {
        if !seen.insert(task.clone()) {
            continue;
        }
        if let Some(action) = problem.actions.get(&task) {
            atoms(&action.precondition, &mut result.reads);
            result
                .writes
                .extend(action.add.iter().chain(&action.delete).cloned());
        }
        if let Some(methods) = problem.methods_by_task.get(&task) {
            for &index in methods {
                let method = &problem.methods[index];
                atoms(&method.precondition, &mut result.reads);
                pending.extend(method.network.tasks.iter().cloned());
            }
        }
    }
    result
}

fn projection(state: &State, relevant: &State) -> BoundaryState {
    BoundaryState {
        must_be_true: state.intersection(relevant).cloned().collect(),
        must_be_false: relevant.difference(state).cloned().collect(),
    }
}

fn partition_order(
    network: &GroundNetwork,
    partitions: &[RootPartition],
) -> Result<Vec<BTreeSet<usize>>, SolverError> {
    let mut owner = BTreeMap::new();
    let mut ids = BTreeSet::new();
    for (index, partition) in partitions.iter().enumerate() {
        if partition.roots.is_empty() || !ids.insert(partition.id) {
            return Err(SolverError("empty or duplicate master partition".into()));
        }
        for &root in &partition.roots {
            if root >= network.tasks.len() || owner.insert(root, index).is_some() {
                return Err(SolverError(
                    "master partitions must cover every root exactly once".into(),
                ));
            }
        }
    }
    if owner.len() != network.tasks.len() {
        return Err(SolverError(
            "master partitions do not cover initial roots".into(),
        ));
    }
    let closure = network_closure(network)?;
    let mut predecessors = vec![BTreeSet::new(); partitions.len()];
    for (&a, &left) in &owner {
        for (&b, &right) in &owner {
            if left != right && closure[a][b] {
                predecessors[right].insert(left);
            }
        }
    }
    Ok(predecessors)
}

fn cyclic_order(predecessors: &[BTreeSet<usize>]) -> bool {
    let mut remaining: BTreeSet<_> = (0..predecessors.len()).collect();
    while let Some(next) = remaining
        .iter()
        .copied()
        .find(|&p| predecessors[p].is_disjoint(&remaining))
    {
        remaining.remove(&next);
    }
    !remaining.is_empty()
}

fn compose(
    problem: &GroundProblem,
    network: &GroundNetwork,
    initial: usize,
    segments: &[LocalSegment],
) -> Result<PlanCertificate, SolverError> {
    let mut global_next = network.tasks.len();
    let mut decisions = Vec::new();
    for segment in segments {
        let mut map: BTreeMap<_, _> = segment
            .partition
            .roots
            .iter()
            .enumerate()
            .map(|(i, &root)| (i, root))
            .collect();
        let mut local_next = segment.partition.roots.len();
        for decision in &segment.certificate.decisions {
            let occurrence = *map
                .get(&decision.occurrence)
                .ok_or_else(|| SolverError("unknown occurrence in local certificate".into()))?;
            if let DecisionKind::Refine { method } = decision.kind {
                for offset in 0..problem.methods[method].network.tasks.len() {
                    map.insert(local_next + offset, global_next + offset);
                }
                let count = problem.methods[method].network.tasks.len();
                local_next += count;
                global_next += count;
            }
            decisions.push(Decision {
                occurrence,
                kind: decision.kind.clone(),
            });
        }
    }
    let request = LocalRequest {
        network: network.clone(),
        initial_state: problem.initial_state.clone(),
        goal: problem.goal.clone(),
    };
    let mut certificate = certify_decisions(problem, &request, &decisions)?;
    certificate.initial_network = initial;
    validate_problem_certificate(problem, &certificate)?;
    Ok(certificate)
}
