//! Repartition after scoped failures. Every attempt has fresh proof caches;
//! unknowns never become global infeasibility, and budgets are shared.
use crate::{
    agglomeration::{self, AgglomerationOptions},
    analysis::{HtnAnalysis, NetworkId},
    instrumentation::{MasterEvent, TraceLevel},
    interaction::{InteractionGraph, NodeId},
    master::*,
    partition::MicroPartition,
    solver::{GroundProblem, validate_problem_certificate},
};
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeSet, time::Instant};
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct EscalationOptions {
    pub max_attempts: usize,
    pub attempt_time_ms: u64,
    pub max_attempt_time_ms: u64,
    pub monolithic_fallback: bool,
    pub split_on_local_unknown: bool,
}
impl Default for EscalationOptions {
    fn default() -> Self {
        Self {
            max_attempts: 6,
            attempt_time_ms: 350,
            max_attempt_time_ms: 4000,
            monolithic_fallback: true,
            split_on_local_unknown: true,
        }
    }
}
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EscalationWork {
    pub master_nodes: usize,
    pub local_calls: usize,
    pub local_expanded: usize,
    pub local_generated: usize,
    pub elapsed_ms: u128,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Repair {
    Initial,
    BoundaryMerge,
    Coarsen,
    SplitLocal,
    IncreaseLocalBudget,
    MonolithicFallback,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Attempt {
    pub repair: Repair,
    pub partitions: Vec<RootPartition>,
    pub options: MasterOptions,
    pub status: MasterStatus,
    pub statistics: MasterStatistics,
    pub boundary_partitions: Vec<usize>,
    pub unknown_partitions: Vec<usize>,
    pub events: Vec<MasterEvent>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationResult {
    pub result: MasterResult,
    pub attempts: Vec<Attempt>,
    pub work: EscalationWork,
    pub stop_reason: String,
}
fn key(clusters: &[MicroPartition]) -> Vec<Vec<NodeId>> {
    let mut k: Vec<_> = clusters
        .iter()
        .map(|c| {
            let mut n = c.nodes.clone();
            n.sort();
            n
        })
        .collect();
    k.sort();
    k
}
#[allow(clippy::too_many_arguments)]
pub fn solve_escalating(
    problem: &GroundProblem,
    analysis: &HtnAnalysis,
    graph: &InteractionGraph,
    clusters: &[MicroPartition],
    merge: &AgglomerationOptions,
    budget: &MasterOptions,
    guidance: &MasterGuidance<'_>,
    options: &EscalationOptions,
    trace: TraceLevel,
) -> Result<EscalationResult> {
    solve_with_executor(
        problem,
        analysis,
        graph,
        clusters,
        merge,
        budget,
        options,
        trace,
        &mut |partitions, attempt, observer| {
            Ok(solve_sequential_guided(
                problem,
                partitions,
                attempt,
                guidance,
                trace,
                Some(observer),
            )?)
        },
    )
}
type AttemptExecutor<'a> = dyn FnMut(&[RootPartition], &MasterOptions, &mut dyn FnMut(MasterEvent)) -> Result<MasterResult>
    + 'a;
#[allow(clippy::too_many_arguments)]
fn solve_with_executor(
    problem: &GroundProblem,
    analysis: &HtnAnalysis,
    graph: &InteractionGraph,
    clusters: &[MicroPartition],
    merge: &AgglomerationOptions,
    budget: &MasterOptions,
    options: &EscalationOptions,
    trace: TraceLevel,
    execute: &mut AttemptExecutor<'_>,
) -> Result<EscalationResult> {
    if options.max_attempts == 0
        || options.max_attempts > 128
        || options.attempt_time_ms == 0
        || options.max_attempt_time_ms < options.attempt_time_ms
    {
        bail!("require 1..128 attempts and 0 < initial stage time <= maximum stage time");
    }
    let start = Instant::now();
    let mut output = EscalationResult {
        result: MasterResult {
            status: MasterStatus::BudgetExhausted,
            exactness: budget.local_budget.exactness,
            certificate: None,
            segments: vec![],
            nogoods: vec![],
            conflict_cores: vec![],
            statistics: MasterStatistics::default(),
        },
        attempts: vec![],
        work: EscalationWork::default(),
        stop_reason: "AttemptLimit".into(),
    };
    let mut current =
        agglomeration::observe(analysis, graph, NetworkId::Initial, clusters, merge)?.clusters;
    let mut visited = BTreeSet::new();
    let mut split = false;
    let mut repair = Repair::Initial;
    let mut stage_time = options.attempt_time_ms;
    let mut local_time = budget.local_budget.max_time_ms.min(stage_time);
    for _ in 0..options.max_attempts {
        let elapsed = start.elapsed().as_millis();
        let left = (budget.max_time_ms as u128)
            .saturating_sub(elapsed)
            .min(u64::MAX as u128) as u64;
        if left == 0
            || output.work.local_calls >= budget.max_local_calls
            || output.work.master_nodes >= budget.max_master_nodes
        {
            output.stop_reason = "SharedBudget".into();
            break;
        }
        visited.insert(key(&current));
        let mut attempt_budget = *budget;
        attempt_budget.max_time_ms = left.min(stage_time);
        attempt_budget.max_local_calls = budget.max_local_calls - output.work.local_calls;
        attempt_budget.max_master_nodes = budget.max_master_nodes - output.work.master_nodes;
        attempt_budget.local_budget.max_time_ms = local_time.min(attempt_budget.max_time_ms);
        let partitions = initial_partitions(graph, &current)?;
        let mut events = Vec::new();
        let result = execute(&partitions, &attempt_budget, &mut |e| events.push(e))?;
        let mut boundary = BTreeSet::new();
        let mut unknown = BTreeSet::new();
        let mut calls = std::collections::BTreeMap::new();
        let mut time_unknown = false;
        for event in &events {
            match event {
                MasterEvent::BoundaryLimit { partition, .. } => {
                    boundary.insert(*partition);
                }
                MasterEvent::OracleStarted {
                    call,
                    partition,
                    purpose: crate::instrumentation::OraclePurpose::Solve,
                    budget: local_budget,
                    ..
                } => {
                    calls.insert(*call, (*partition, *local_budget));
                }
                MasterEvent::OracleFinished {
                    call,
                    status: crate::solver::ProofStatus::BudgetExhausted,
                    statistics,
                    ..
                } => {
                    if let Some((p, b)) = calls.get(call) {
                        unknown.insert(*p);
                        time_unknown |= b.max_time_ms > 0
                            && statistics.expanded < b.max_expansions
                            && statistics.generated < b.max_generated;
                    }
                }
                _ => {}
            }
        }
        time_unknown |= result.statistics.elapsed_ms >= attempt_budget.max_time_ms as u128;
        output.work.local_calls += result.statistics.local_calls;
        output.work.master_nodes += result.statistics.master_nodes;
        output.work.local_expanded += result.statistics.local_expanded;
        output.work.local_generated += result.statistics.local_generated;
        output.attempts.push(Attempt {
            repair,
            partitions,
            options: attempt_budget,
            status: result.status,
            statistics: result.statistics.clone(),
            boundary_partitions: boundary.iter().copied().collect(),
            unknown_partitions: unknown.iter().copied().collect(),
            events: if trace == TraceLevel::Full {
                events
            } else {
                vec![]
            },
        });
        output.result = result;
        if output.result.status == MasterStatus::Feasible {
            validate_problem_certificate(
                problem,
                output
                    .result
                    .certificate
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("feasible escalation lacks certificate"))?,
            )?;
            output.stop_reason = "Feasible".into();
            break;
        }
        // Split only once and only blocks involved in unknown local searches.
        // Canonical node sets, rather than fresh IDs, prevent repair cycles.
        if options.split_on_local_unknown && boundary.is_empty() && !split && !unknown.is_empty() {
            let mut next = Vec::new();
            let mut id = current
                .iter()
                .map(|c| c.id)
                .max()
                .unwrap_or(0)
                .checked_add(1)
                .context("partition ID overflow")?;
            for c in &current {
                if unknown.contains(&c.id) && c.nodes.len() > 1 {
                    for &node in &c.nodes {
                        next.push(MicroPartition {
                            id,
                            nodes: vec![node],
                        });
                        id = id.checked_add(1).context("partition ID overflow")?;
                    }
                } else {
                    next.push(c.clone());
                }
            }
            split = true;
            let next =
                agglomeration::observe(analysis, graph, NetworkId::Initial, &next, merge)?.clusters;
            if !visited.contains(&key(&next)) {
                current = next;
                repair = Repair::SplitLocal;
                continue;
            }
        }
        let state = agglomeration::observe(analysis, graph, NetworkId::Initial, &current, merge)?;
        let mut candidates =
            agglomeration::legal_candidates(graph, NetworkId::Initial, &state, merge)?;
        candidates.sort_by(|a, b| {
            let touches = |c: &agglomeration::MergeCandidate| {
                usize::from(boundary.contains(&c.left) || boundary.contains(&c.right))
            };
            touches(b)
                .cmp(&touches(a))
                .then_with(|| b.score.total.total_cmp(&a.score.total))
                .then_with(|| (a.left, a.right).cmp(&(b.left, b.right)))
        });
        let mut next = None;
        for candidate in candidates {
            let merged = agglomeration::merged_clusters(&current, &candidate)?;
            if !visited.contains(&key(&merged)) {
                next = Some(merged);
                break;
            }
        }
        if let Some(next) = next {
            current = next;
            repair = if boundary.is_empty() {
                Repair::Coarsen
            } else {
                Repair::BoundaryMerge
            };
        } else {
            // This is an explicit restart of the exact whole-network solver,
            // not an additional learned action or an adjacency-bypassing merge.
            let members: Vec<_> = current
                .iter()
                .flat_map(|c| c.nodes.iter().copied())
                .collect();
            if options.monolithic_fallback
                && current.len() > 1
                && members.len() <= merge.max_cluster_size
            {
                let id = current
                    .iter()
                    .map(|c| c.id)
                    .max()
                    .unwrap_or(0)
                    .checked_add(1)
                    .context("partition ID overflow")?;
                let fallback = vec![MicroPartition { id, nodes: members }];
                if !visited.contains(&key(&fallback)) {
                    current = fallback;
                    repair = Repair::MonolithicFallback;
                    continue;
                }
            }
            let next_stage = stage_time
                .saturating_mul(2)
                .min(options.max_attempt_time_ms);
            let next_local = local_time.saturating_mul(2).min(next_stage);
            if time_unknown
                && (next_stage > stage_time || next_local > local_time)
                && next_local > 0
            {
                stage_time = next_stage;
                local_time = next_local;
                repair = Repair::IncreaseLocalBudget;
            } else {
                output.stop_reason = "NoRepair".into();
                break;
            }
        }
    }
    // Failed partition strategies are never a proof over all HTN refinements.
    if output.result.status != MasterStatus::Feasible {
        output.result.status = MasterStatus::BudgetExhausted;
    }
    output.work.elapsed_ms = start.elapsed().as_millis();
    Ok(output)
}
impl EscalationResult {
    /// Counters describe all scoped attempts; no proof cache is shared.
    pub fn total_statistics(&self) -> MasterStatistics {
        let mut total = MasterStatistics::default();
        for attempt in &self.attempts {
            let s = &attempt.statistics;
            macro_rules! sum {($($field:ident),*)=>{$(total.$field+=s.$field;)*};}
            sum!(
                master_nodes,
                local_calls,
                interface_candidates,
                nogood_hits,
                local_expanded,
                local_generated,
                local_unknown,
                boundary_cutoffs,
                suffix_cache_hits,
                oracle_feasible,
                oracle_infeasible,
                oracle_bounded,
                core_checks,
                core_hits,
                core_generalized_hits,
                core_assignments_skipped,
                core_active,
                core_subsumed,
                core_redundant,
                core_incoming_literals,
                core_literals_removed,
                interface_proposals,
                interface_inference_us,
                causal_certificates,
                causal_links,
                causal_imports,
                causal_threats,
                causal_time_us
            );
            total.max_boundary_atoms = total.max_boundary_atoms.max(s.max_boundary_atoms);
        }
        total.elapsed_ms = self.work.elapsed_ms;
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn time_retries_grow_within_stage_and_shared_call_caps() {
        let program = crate::parsing::json::decode_program(include_bytes!(
            "../tests/fixtures/interleaving.json"
        ))
        .unwrap();
        let analysis = crate::analysis::analyze_program(&program).unwrap();
        let graph = InteractionGraph::from_analysis(&analysis);
        let problem = crate::solver::ground_reachable_program(
            &program,
            &crate::solver::GroundingOptions::default(),
        )
        .unwrap();
        let clusters = vec![MicroPartition {
            id: 1,
            nodes: graph.network_nodes(NetworkId::Initial),
        }];
        let mut budget = MasterOptions {
            max_time_ms: 1000,
            max_local_calls: 3,
            ..Default::default()
        };
        budget.local_budget.max_time_ms = 10;
        let mut observed = Vec::new();
        let result = solve_with_executor(
            &problem,
            &analysis,
            &graph,
            &clusters,
            &AgglomerationOptions::default(),
            &budget,
            &EscalationOptions {
                attempt_time_ms: 10,
                max_attempt_time_ms: 40,
                ..Default::default()
            },
            TraceLevel::Summary,
            &mut |_, slice, _| {
                observed.push((
                    slice.max_time_ms,
                    slice.local_budget.max_time_ms,
                    slice.max_local_calls,
                ));
                Ok(MasterResult {
                    status: MasterStatus::BudgetExhausted,
                    exactness: slice.local_budget.exactness,
                    certificate: None,
                    segments: vec![],
                    nogoods: vec![],
                    conflict_cores: vec![],
                    statistics: MasterStatistics {
                        local_calls: 1,
                        master_nodes: 1,
                        elapsed_ms: slice.max_time_ms as u128,
                        ..Default::default()
                    },
                })
            },
        )
        .unwrap();
        assert_eq!(observed, vec![(10, 10, 3), (20, 20, 2), (40, 40, 1)]);
        assert_eq!(result.work.local_calls, 3);
        assert_eq!(result.result.status, MasterStatus::BudgetExhausted);
        assert!(
            result
                .attempts
                .iter()
                .skip(1)
                .all(|a| matches!(a.repair, Repair::IncreaseLocalBudget))
        );
    }
}
