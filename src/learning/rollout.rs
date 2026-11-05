//! Counterfactual merge/STOP evaluations against the same bounded exact oracle.
use super::features::Observation;
use super::model::{Supervision, ValueTarget};
use crate::agglomeration::{self, AgglomerationOptions, MergeCandidate};
use crate::analysis::{NetworkId, analyze_program};
use crate::instrumentation::{MasterEvent, TraceLevel};
use crate::interaction::InteractionGraph;
use crate::master::{
    MasterGuidance, MasterOptions, MasterResult, MasterStatus, RootPartition, initial_partitions,
    solve_sequential_guided,
};
use crate::parsing::json::Program;
use crate::partition::{MicroPartition, SpectralOptions, micropartition_network};
use crate::solver::{GroundingMode, GroundingOptions, ground_reachable_program};
use crate::store::{DatasetStore, Entry, Metadata, RecordKind};
use crate::training::sha256;
use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RolloutOptions {
    pub max_ground_instances: usize,
    pub micro_max_size: usize,
    pub agglomeration: AgglomerationOptions,
    pub master: MasterOptions,
    pub max_levels: usize,
    pub max_candidates: usize,
    pub trace: TraceLevel,
    #[serde(default)]
    pub conflict_core_checks: usize,
}
impl Default for RolloutOptions {
    fn default() -> Self {
        Self {
            max_ground_instances: 200_000,
            micro_max_size: 1,
            agglomeration: AgglomerationOptions::default(),
            master: MasterOptions {
                max_time_ms: 500,
                max_local_calls: 64,
                local_budget: crate::solver::SearchBudget {
                    max_time_ms: 100,
                    ..Default::default()
                },
                ..Default::default()
            },
            max_levels: 4,
            max_candidates: 8,
            trace: TraceLevel::Summary,
            conflict_core_checks: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProblemRecord {
    pub program: Program,
    pub grounding_mode: GroundingMode,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionOutcome {
    pub action: usize,
    pub partitions: Vec<RootPartition>,
    pub result: Option<MasterResult>,
    pub error: Option<String>,
    pub events: Vec<MasterEvent>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RolloutRecord {
    pub schema_version: u32,
    pub problem_id: String,
    pub group_id: String,
    pub level: usize,
    pub options: RolloutOptions,
    pub clusters: Vec<MicroPartition>,
    pub observation: Observation,
    pub outcomes: Vec<ActionOutcome>,
    pub omitted_candidates: usize,
    pub grounding_ms: u128,
    pub elapsed_ms: u128,
    pub analysis_ms: u128,
    pub graph_ms: u128,
    pub spectral_ms: u128,
    pub feature_ms: u128,
    pub grounded_actions: usize,
    pub grounded_methods: usize,
}

impl RolloutRecord {
    pub fn validate(&self) -> Result<()> {
        self.observation.validate()?;
        if self.schema_version != 1
            || self.outcomes.len() != self.observation.actions.len()
            || self
                .outcomes
                .iter()
                .enumerate()
                .any(|(i, o)| o.action != i || o.result.is_some() == o.error.is_some())
        {
            bail!("invalid rollout outcomes or schema");
        }
        for outcome in &self.outcomes {
            if let Some(result) = &outcome.result {
                if result.exactness != self.options.master.local_budget.exactness {
                    bail!("rollout oracle exactness mismatch");
                }
                if (result.status == MasterStatus::Feasible) != result.certificate.is_some() {
                    bail!("rollout feasibility/certificate mismatch");
                }
                if result.nogoods.iter().any(|cut| {
                    !matches!(
                        cut.proof,
                        crate::solver::ProofStatus::Infeasible
                            | crate::solver::ProofStatus::InfeasibleWithinBound
                    )
                }) {
                    bail!("unproven rollout no-good");
                }
                if result
                    .nogoods
                    .iter()
                    .any(|cut| cut.exactness != result.exactness)
                {
                    bail!("rollout no-good scope mismatch");
                }
                for core in &result.conflict_cores {
                    core.validate_scope()?;
                    if core.exactness != result.exactness
                        || core.verification_call < core.source_call
                        || core.verification_call >= result.statistics.local_calls
                    {
                        bail!("rollout conflict-core scope mismatch");
                    }
                }
                let mut started = std::collections::BTreeMap::new();
                let mut finished = std::collections::BTreeMap::new();
                let mut end = 0;
                let mut cuts = 0;
                let mut cores = 0;
                let mut applications = 0;
                let mut skipped = 0usize;
                for event in &outcome.events {
                    if end != 0 {
                        bail!("event follows master finish");
                    }
                    match event {
                        MasterEvent::OracleStarted {
                            call,
                            initial_network,
                            partition,
                            request,
                            ..
                        } => {
                            if self.options.trace == TraceLevel::Full && request.is_none() {
                                bail!("full trace missing oracle request");
                            }
                            if started
                                .insert(*call, (*initial_network, *partition))
                                .is_some()
                            {
                                bail!("duplicate oracle call");
                            }
                        }
                        MasterEvent::OracleFinished {
                            call,
                            status,
                            certificate,
                            ..
                        } => {
                            if (certificate.is_some()
                                && *status != crate::solver::ProofStatus::Feasible)
                                || (self.options.trace == TraceLevel::Full
                                    && *status == crate::solver::ProofStatus::Feasible
                                    && certificate.is_none())
                            {
                                bail!("oracle trace feasibility/certificate mismatch");
                            }
                            if !started.contains_key(call)
                                || finished.insert(*call, *status).is_some()
                            {
                                bail!("unpaired oracle finish");
                            }
                        }
                        MasterEvent::CutVerified { call, nogood } => {
                            if finished.get(call) != Some(&nogood.proof)
                                || started.get(call)
                                    != Some(&(nogood.initial_network, nogood.partition))
                            {
                                bail!("unproven cut event");
                            }
                            cuts += 1;
                        }
                        MasterEvent::ConflictCoreVerified { core } => {
                            core.validate_scope()?;
                            if finished.get(&core.verification_call) != Some(&core.proof)
                                || started.get(&core.source_call)
                                    != Some(&(core.initial_network, core.partition))
                                || started.get(&core.verification_call)
                                    != Some(&(core.initial_network, core.partition))
                                || !matches!(
                                    finished.get(&core.source_call),
                                    Some(
                                        crate::solver::ProofStatus::Infeasible
                                            | crate::solver::ProofStatus::InfeasibleWithinBound
                                    )
                                )
                            {
                                bail!("unproven conflict-core event");
                            }
                            if !result.conflict_cores.get(cores).is_some_and(|c| {
                                serde_json::to_value(c).ok() == serde_json::to_value(core).ok()
                            }) {
                                bail!("conflict-core event differs from result");
                            }
                            cores += 1;
                        }
                        MasterEvent::ConflictCoreApplied {
                            core_index,
                            initial_network,
                            partition,
                            initial_state,
                            goal,
                            boundary,
                            ordinal,
                            next_ordinal,
                        } => {
                            if *core_index >= cores || next_ordinal <= ordinal {
                                bail!("invalid core application order");
                            }
                            let cut = result
                                .conflict_cores
                                .get(*core_index)
                                .ok_or_else(|| anyhow::anyhow!("unknown applied core"))?;
                            if cut.initial_network != *initial_network
                                || cut.partition != *partition
                            {
                                bail!("core application scope mismatch");
                            }
                            if let (Some(state), Some(goal), Some(boundary)) =
                                (initial_state, goal, boundary)
                            {
                                if boundary.len() >= usize::BITS as usize
                                    || !cut.matches(
                                        *initial_network,
                                        *partition,
                                        state,
                                        goal,
                                        result.exactness,
                                    )
                                    || *ordinal >= (1usize << boundary.len())
                                    || *next_ordinal
                                        != crate::conflicts::after_forbidden_cube(
                                            *ordinal, boundary, &cut.core,
                                        )
                                {
                                    bail!("unjustified core application");
                                }
                            } else if self.options.trace == TraceLevel::Full {
                                bail!("full core application is missing its context");
                            }
                            applications += 1;
                            skipped = skipped.saturating_add(next_ordinal - ordinal);
                        }
                        MasterEvent::Finished { status, statistics } => {
                            if *status != result.status
                                || serde_json::to_value(statistics)?
                                    != serde_json::to_value(&result.statistics)?
                            {
                                bail!("master finish event mismatch");
                            }
                            end += 1;
                        }
                        _ => {}
                    }
                }
                if end != 1
                    || started.len() != result.statistics.local_calls
                    || finished.len() != started.len()
                    || cuts != result.nogoods.len()
                    || cores != result.conflict_cores.len()
                {
                    bail!("incomplete master trace");
                }
                if result.conflict_cores.iter().any(|c| c.incoming.is_some())
                    && (applications != result.statistics.core_hits
                        || skipped != result.statistics.core_assignments_skipped
                        || result
                            .statistics
                            .core_active
                            .saturating_add(result.statistics.core_subsumed)
                            .saturating_add(result.statistics.core_redundant)
                            != cores)
                {
                    bail!("core database/application statistics mismatch");
                }
            }
        }
        Ok(())
    }

    pub fn supervision(&self, master_weight: f64) -> Result<Supervision> {
        self.validate()?;
        if !master_weight.is_finite() || master_weight < 0.0 {
            bail!("invalid effort weight");
        }
        let effort = |r: &MasterResult| {
            (r.statistics.local_expanded as f64 + master_weight * r.statistics.master_nodes as f64)
                .ln_1p()
        };
        let mut labels = Supervision::default();
        for (i, outcome) in self.outcomes.iter().enumerate() {
            let Some(result) = &outcome.result else {
                continue;
            };
            let clean_prefix = result.status == MasterStatus::BudgetExhausted
                && result.statistics.local_unknown == 0
                && result.statistics.boundary_cutoffs == 0;
            if result.status == MasterStatus::Feasible || clean_prefix {
                labels.values.push(ValueTarget {
                    action: i,
                    log_effort: effort(result),
                    lower_bound: result.status == MasterStatus::BudgetExhausted,
                });
            }
            for (j, other) in self.outcomes.iter().enumerate() {
                let Some(other) = &other.result else { continue };
                if result.status != MasterStatus::Feasible {
                    continue;
                }
                let preferred = other.status != MasterStatus::Feasible
                    || effort(result) < effort(other)
                    || (effort(result) == effort(other)
                        && result.certificate.as_ref().unwrap().actions.len()
                            < other.certificate.as_ref().unwrap().actions.len());
                if i != j && preferred {
                    labels.preferences.push((i, j));
                }
            }
        }
        Ok(labels)
    }
}

pub(super) fn limit_candidates(
    observation: &mut Observation,
    candidates: &mut Vec<MergeCandidate>,
    limit: usize,
) -> usize {
    let omitted = candidates.len().saturating_sub(limit);
    let mut order: Vec<_> = (0..candidates.len()).collect();
    order.sort_by(|&a, &b| {
        candidates[b]
            .score
            .total
            .total_cmp(&candidates[a].score.total)
            .then_with(|| a.cmp(&b))
    });
    order.truncate(limit);
    order.sort();
    let mut actions = order
        .iter()
        .map(|&i| observation.actions[i].clone())
        .collect::<Vec<_>>();
    actions.push(observation.actions.last().unwrap().clone());
    observation.actions = actions;
    *candidates = order.into_iter().map(|i| candidates[i].clone()).collect();
    omitted
}

pub fn collect_program(
    store: &DatasetStore,
    program: &Program,
    source: Option<String>,
    options: &RolloutOptions,
) -> Result<Vec<Entry>> {
    if options.max_levels == 0 || options.max_candidates == 0 || options.micro_max_size == 0 {
        bail!("rollout level/candidate/cluster limits must be positive");
    }
    let group = sha256(&serde_json::to_vec(&program.domain)?);
    let problem = store.put(
        Metadata {
            kind: RecordKind::Problem,
            group_id: Some(group.clone()),
            source_id: source,
        },
        &ProblemRecord {
            program: program.clone(),
            grounding_mode: GroundingMode::InitialReachable,
        },
    )?;
    let start = Instant::now();
    let model = match ground_reachable_program(
        program,
        &GroundingOptions {
            max_instances: options.max_ground_instances,
        },
    ) {
        Ok(model) => model,
        Err(error) => {
            store.put(Metadata{kind:RecordKind::CollectionFailure,group_id:Some(group),source_id:Some(problem.id)},
                &serde_json::json!({"phase":"grounding","error":error.to_string(),"elapsed_ms":start.elapsed().as_millis(),"options":options}))?;
            return Err(error.into());
        }
    };
    let grounding_ms = start.elapsed().as_millis();
    let start = Instant::now();
    let analysis = analyze_program(program)?;
    let analysis_ms = start.elapsed().as_millis();
    let start = Instant::now();
    let graph = InteractionGraph::from_analysis(&analysis);
    let graph_ms = start.elapsed().as_millis();
    let start = Instant::now();
    let mut clusters = micropartition_network(
        &graph,
        NetworkId::Initial,
        &SpectralOptions {
            max_cluster_size: options.micro_max_size,
            ..Default::default()
        },
    )?
    .clusters;
    let spectral_ms = start.elapsed().as_millis();
    let mut records = Vec::new();
    for level in 0..options.max_levels {
        let feature_start = Instant::now();
        let state = agglomeration::observe(
            &analysis,
            &graph,
            NetworkId::Initial,
            &clusters,
            &options.agglomeration,
        )?;
        clusters = state.clusters.clone();
        let (mut observation, mut candidates) =
            Observation::build(&analysis, &graph, &state, &options.agglomeration)?;
        observation.set_search_context(&options.master);
        let omitted_candidates =
            limit_candidates(&mut observation, &mut candidates, options.max_candidates);
        let feature_ms = feature_start.elapsed().as_millis();
        let mut outcomes = Vec::new();
        let level_start = Instant::now();
        for action in 0..observation.actions.len() {
            let proposed = if action < candidates.len() {
                agglomeration::merged_clusters(&clusters, &candidates[action])?
            } else {
                clusters.clone()
            };
            let partitions = initial_partitions(&graph, &proposed)?;
            let mut events = Vec::new();
            let result = solve_sequential_guided(
                &model,
                &partitions,
                &options.master,
                &MasterGuidance {
                    interface: None,
                    conflict_core_checks: options.conflict_core_checks,
                },
                options.trace,
                Some(&mut |event| events.push(event)),
            );
            let (result, error) = match result {
                Ok(result) => (Some(result), None),
                Err(error) => (None, Some(error.to_string())),
            };
            outcomes.push(ActionOutcome {
                action,
                partitions,
                result,
                error,
                events,
            });
        }
        let record = RolloutRecord {
            schema_version: 1,
            problem_id: problem.id.clone(),
            group_id: group.clone(),
            level,
            options: options.clone(),
            clusters: clusters.clone(),
            observation,
            outcomes,
            omitted_candidates,
            grounding_ms,
            elapsed_ms: level_start.elapsed().as_millis(),
            analysis_ms,
            graph_ms,
            spectral_ms,
            feature_ms,
            grounded_actions: model.actions.len(),
            grounded_methods: model.methods.len(),
        };
        record.validate()?;
        records.push(store.put(
            Metadata {
                kind: RecordKind::Rollout,
                group_id: Some(group.clone()),
                source_id: Some(problem.id.clone()),
            },
            &record,
        )?);
        let Some(best) = candidates.iter().max_by(|a, b| {
            a.score
                .total
                .total_cmp(&b.score.total)
                .then_with(|| b.left.cmp(&a.left))
                .then_with(|| b.right.cmp(&a.right))
        }) else {
            break;
        };
        clusters = agglomeration::merged_clusters(&clusters, best)?;
    }
    Ok(records)
}

pub use super::audit::{AuditStatistics, RolloutAuditor, audit_record};
