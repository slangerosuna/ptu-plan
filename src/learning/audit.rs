//! Snapshot-scoped semantic rollout validation with reusable symbolic/proof caches.
use super::features::Observation;
use super::rollout::{ProblemRecord, RolloutRecord, limit_candidates};
use crate::{
    agglomeration,
    analysis::{HtnAnalysis, NetworkId, analyze_program},
    instrumentation::{MasterEvent, OraclePurpose},
    interaction::InteractionGraph,
    master::initial_partitions,
    solver::{
        GroundProblem, GroundingMode, GroundingOptions, State, ground_reachable_program,
        validate_problem_certificate,
    },
    store::{DatasetStore, Entry, RecordKind, Snapshot},
    training::sha256,
};
use anyhow::{Context, Result, bail};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

struct PreparedProblem {
    group: String,
    model: GroundProblem,
    analysis: HtnAnalysis,
    graph: InteractionGraph,
}
#[derive(Debug, Clone, Default, Serialize)]
pub struct AuditStatistics {
    pub records: usize,
    pub prepared_problems: usize,
    pub core_verifications: usize,
    pub core_cache_hits: usize,
}
/// Caches are confined to one immutable index snapshot and compilation strategy.
pub struct RolloutAuditor<'a> {
    store: &'a DatasetStore,
    entries: BTreeMap<&'a str, &'a Entry>,
    problems: BTreeMap<(String, usize), PreparedProblem>,
    verified_cores: BTreeSet<Vec<u8>>,
    pub statistics: AuditStatistics,
}
impl<'a> RolloutAuditor<'a> {
    pub fn new(store: &'a DatasetStore, snapshot: &'a Snapshot) -> Self {
        Self {
            store,
            entries: snapshot
                .entries
                .iter()
                .map(|e| (e.id.as_str(), e))
                .collect(),
            problems: BTreeMap::new(),
            verified_cores: BTreeSet::new(),
            statistics: AuditStatistics::default(),
        }
    }
    pub fn model(&self, record: &RolloutRecord) -> Result<&GroundProblem> {
        Ok(&self
            .problems
            .get(&(
                record.problem_id.clone(),
                record.options.max_ground_instances,
            ))
            .context("rollout source has not been audited")?
            .model)
    }
    pub(crate) fn structure(
        &self,
        record: &RolloutRecord,
    ) -> Result<(&HtnAnalysis, &InteractionGraph)> {
        let prepared = self
            .problems
            .get(&(
                record.problem_id.clone(),
                record.options.max_ground_instances,
            ))
            .context("rollout source has not been audited")?;
        Ok((&prepared.analysis, &prepared.graph))
    }
    pub fn audit(&mut self, entry: &Entry) -> Result<RolloutRecord> {
        if self.entries.get(entry.id.as_str()).copied() != Some(entry) {
            bail!("rollout entry is outside the audit snapshot");
        }
        let store = self.store;
        let record: RolloutRecord = store.load(entry)?;
        record.validate()?;
        if entry.metadata.kind != RecordKind::Rollout
            || entry.metadata.group_id.as_deref() != Some(&record.group_id)
            || entry.metadata.source_id.as_deref() != Some(&record.problem_id)
        {
            bail!("rollout metadata mismatch");
        }
        let key = (
            record.problem_id.clone(),
            record.options.max_ground_instances,
        );
        if !self.problems.contains_key(&key) {
            let source = self
                .entries
                .get(record.problem_id.as_str())
                .filter(|e| e.metadata.kind == RecordKind::Problem)
                .context("missing rollout problem")?;
            let problem: ProblemRecord = store.load(source)?;
            let group = sha256(&serde_json::to_vec(&problem.program.domain)?);
            if problem.grounding_mode != GroundingMode::InitialReachable
                || source.metadata.group_id.as_deref() != Some(&group)
            {
                bail!("rollout problem/source group or grounding mismatch");
            }
            let model = ground_reachable_program(
                &problem.program,
                &GroundingOptions {
                    max_instances: record.options.max_ground_instances,
                },
            )?;
            let analysis = analyze_program(&problem.program)?;
            let graph = InteractionGraph::from_analysis(&analysis);
            self.problems.insert(
                key.clone(),
                PreparedProblem {
                    group,
                    model,
                    analysis,
                    graph,
                },
            );
            self.statistics.prepared_problems += 1;
        }
        let prepared = &self.problems[&key];
        if prepared.group != record.group_id {
            bail!("rollout/source group mismatch");
        }
        let PreparedProblem {
            model,
            analysis,
            graph,
            ..
        } = prepared;
        let state = agglomeration::observe(
            analysis,
            graph,
            NetworkId::Initial,
            &record.clusters,
            &record.options.agglomeration,
        )?;
        let (mut observation, mut candidates) =
            Observation::build(analysis, graph, &state, &record.options.agglomeration)?;
        if let Some(old) = &record.observation.hilg {
            let graph = observation.hilg.as_mut().unwrap();
            if old.wl.is_none() {
                graph.wl = None;
            }
            if old.colors.is_empty() {
                graph.colors.clear();
            }
        } else {
            observation.hilg = None;
        }
        observation.set_search_context(&record.options.master);
        let omitted = limit_candidates(
            &mut observation,
            &mut candidates,
            record.options.max_candidates,
        );
        if omitted != record.omitted_candidates
            || !crate::training::spectral_cache_matches(
                &serde_json::to_value(&record.observation)?,
                &serde_json::to_value(&observation)?,
            )
        {
            bail!("rollout observation differs from symbolic partition state");
        }
        for outcome in &record.outcomes {
            if let Some(result) = &outcome.result {
                for core in &result.conflict_cores {
                    let partition = outcome
                        .partitions
                        .iter()
                        .find(|p| p.id == core.partition)
                        .context("conflict core has unknown partition")?;
                    core.validate_scope()?;
                    let key = serde_json::to_vec(&(
                        &record.problem_id,
                        record.options.max_ground_instances,
                        &partition.roots,
                        core.initial_network,
                        &core.initial_state,
                        &core.incoming,
                        &core.core,
                        core.exactness,
                        core.proof,
                        &record.options.master.local_budget,
                    ))?;
                    if self.verified_cores.contains(&key) {
                        self.statistics.core_cache_hits += 1;
                    } else {
                        crate::conflicts::validate_conflict_core(
                            model,
                            partition,
                            core,
                            &record.options.master.local_budget,
                        )?;
                        self.verified_cores.insert(key);
                        self.statistics.core_verifications += 1;
                    }
                }
            }
            let proposed = if let Some(candidate) = candidates.get(outcome.action) {
                agglomeration::merged_clusters(&state.clusters, candidate)?
            } else {
                state.clusters.clone()
            };
            if initial_partitions(graph, &proposed)? != outcome.partitions {
                bail!("rollout partitions differ from proposed merge");
            }
            if let Some(certificate) = outcome.result.as_ref().and_then(|r| r.certificate.as_ref())
            {
                validate_problem_certificate(model, certificate)?;
                let result = outcome.result.as_ref().unwrap();
                let mut incoming = model.initial_state.clone();
                let mut actions = Vec::new();
                let mut seen = std::collections::BTreeSet::new();
                for (index, segment) in result.segments.iter().enumerate() {
                    if !outcome.partitions.contains(&segment.partition)
                        || !seen.insert(segment.partition.id)
                        || segment.request.initial_state != incoming
                    {
                        bail!("inconsistent sequential segment provenance");
                    }
                    let expected = model.initial_networks[certificate.initial_network]
                        .network
                        .induced(&segment.partition.roots)?;
                    if serde_json::to_value(&expected)?
                        != serde_json::to_value(&segment.request.network)?
                    {
                        bail!("segment has wrong induced network");
                    }
                    let goal = if index + 1 == outcome.partitions.len() {
                        model.goal.clone()
                    } else {
                        segment.outgoing.condition()?
                    };
                    if segment.request.goal != goal
                        || !segment.incoming.condition()?.holds(&incoming)
                    {
                        bail!("segment contract mismatch");
                    }
                    crate::solver::validate_certificate(
                        model,
                        &segment.request,
                        &segment.certificate,
                    )?;
                    if let Some(contract) = &segment.causal {
                        if contract.partition != segment.partition.id {
                            bail!("causal contract partition mismatch");
                        }
                        crate::solver::causal::validate_causal_contract(
                            model,
                            &segment.request,
                            &segment.certificate,
                            contract,
                        )?;
                    }
                    incoming = segment.certificate.final_state.clone();
                    actions.extend(segment.certificate.actions.iter().cloned());
                }
                if seen.len() != outcome.partitions.len()
                    || incoming != certificate.final_state
                    || actions != certificate.actions
                {
                    bail!("segments do not compose the global witness");
                }
            }
            let mut requests = std::collections::BTreeMap::new();
            let mut footprints = BTreeMap::new();
            let mut goal_atoms = State::new();
            crate::master::atoms(&model.goal, &mut goal_atoms);
            for event in &outcome.events {
                match event {
                    MasterEvent::InitialNetwork {
                        initial_network,
                        partitions,
                    } => {
                        if model.initial_networks.get(*initial_network).is_none()
                            || *partitions != outcome.partitions
                        {
                            bail!("trace initial-network/partition provenance mismatch");
                        }
                    }
                    MasterEvent::OracleStarted {
                        call,
                        request: Some(request),
                        initial_network,
                        partition,
                        budget,
                        purpose,
                        interface_context,
                        ..
                    } => {
                        let network = &model
                            .initial_networks
                            .get(*initial_network)
                            .context("unknown oracle initial network")?
                            .network;
                        let partition = outcome
                            .partitions
                            .iter()
                            .find(|p| p.id == *partition)
                            .context("unknown oracle partition")?;
                        let expected = network.induced(&partition.roots)?;
                        if expected.tasks != request.network.tasks
                            || expected.precedence != request.network.precedence
                        {
                            bail!("oracle request has wrong induced network");
                        }
                        let limit = record.options.master.local_budget;
                        if budget.exactness != limit.exactness
                            || budget.max_expansions > limit.max_expansions
                            || budget.max_generated > limit.max_generated
                            || budget.max_time_ms > limit.max_time_ms
                            || budget.max_time_ms > record.options.master.max_time_ms
                        {
                            bail!("oracle trace exceeds recorded search limits");
                        }
                        if let Some(context) = interface_context {
                            if *purpose != OraclePurpose::Solve {
                                bail!("core probe has interface training context");
                            }
                            let all = footprints.entry(*initial_network).or_insert_with(|| {
                                outcome
                                    .partitions
                                    .iter()
                                    .map(|p| (p.id, crate::master::footprint(model, network, p)))
                                    .collect::<BTreeMap<_, _>>()
                            });
                            let mut possible = goal_atoms.clone();
                            for (&id, f) in all.iter() {
                                if id != partition.id {
                                    possible.extend(f.reads.iter().cloned());
                                }
                            }
                            if !goal_atoms.is_subset(&context.future_reads)
                                || !context.future_reads.is_subset(&possible)
                            {
                                bail!("interface context has invalid future reads");
                            }
                            let boundary: State = context.boundary.iter().cloned().collect();
                            if boundary.len() != context.boundary.len() {
                                bail!("duplicate interface boundary atom");
                            }
                            if !boundary.is_empty() || request.goal != model.goal {
                                let expected: State = all[&partition.id]
                                    .writes
                                    .intersection(&context.future_reads)
                                    .cloned()
                                    .collect();
                                let target = crate::conflicts::conjunction(&request.goal)
                                    .context("non-conjunctive interface goal")?;
                                if expected != boundary
                                    || target
                                        .must_be_true
                                        .union(&target.must_be_false)
                                        .cloned()
                                        .collect::<State>()
                                        != boundary
                                {
                                    bail!("oracle interface context/goal mismatch");
                                }
                            }
                        }
                        requests.insert(*call, (request, *purpose));
                    }
                    MasterEvent::OracleFinished {
                        call,
                        certificate: Some(certificate),
                        ..
                    } => {
                        crate::solver::validate_certificate(
                            model,
                            requests
                                .get(call)
                                .context("oracle trace missing request")?
                                .0,
                            certificate,
                        )?;
                    }
                    MasterEvent::ConflictCoreVerified { core }
                        if record.options.trace == crate::instrumentation::TraceLevel::Full =>
                    {
                        let (source, purpose) = requests
                            .get(&core.source_call)
                            .context("core trace missing source request")?;
                        let (verified, verified_purpose) = requests
                            .get(&core.verification_call)
                            .context("core trace missing verification request")?;
                        if *purpose != OraclePurpose::Solve
                            || source.goal != core.source_goal
                            || source.initial_state != core.initial_state
                            || verified.initial_state != core.initial_state
                            || crate::conflicts::conjunction(&verified.goal).as_ref()
                                != Some(&core.core)
                            || (core.verification_call != core.source_call
                                && *verified_purpose != OraclePurpose::CoreMinimization)
                        {
                            bail!("conflict-core proof request mismatch");
                        }
                    }
                    _ => {}
                }
            }
        }
        self.statistics.records += 1;
        Ok(record)
    }
}

/// Independently audit one record. Use RolloutAuditor to amortize a full snapshot.
pub fn audit_record(store: &DatasetStore, entry: &Entry) -> Result<()> {
    let snapshot = store.snapshot()?;
    RolloutAuditor::new(store, &snapshot)
        .audit(entry)
        .map(|_| ())
}
