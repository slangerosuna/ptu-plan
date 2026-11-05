//! On-policy black-box planner environment. Validation domains are evaluation
//! only. Certificates are replayed before a success reward is emitted.
use super::{
    features::Observation,
    interface::InterfaceCheckpoint,
    model::{MergeModel, Prediction, random},
    policy::{PolicyDecision, action_probabilities, agglomerate_with_selector},
    training::Checkpoint,
};
use crate::{
    agglomeration::AgglomerationOptions,
    analysis::{HtnAnalysis, NetworkId, analyze_program},
    escalation::{EscalationOptions, EscalationWork, solve_escalating},
    instrumentation::TraceLevel,
    interaction::InteractionGraph,
    master::{MasterGuidance, MasterOptions, MasterStatus},
    partition::{MicroPartition, SpectralOptions, micropartition_network},
    solver::{GroundProblem, GroundingOptions, ground_reachable_program},
    store::{DatasetStore, RecordKind, Split, group_split},
    training::sha256,
};
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, VecDeque},
    io::{BufRead, Write},
    path::{Path, PathBuf},
};
#[derive(Debug, Clone, clap::Args, Serialize)]
pub struct EnvironmentOptions {
    #[arg(long, default_value = "runs/hilg-wl/data")]
    pub store: PathBuf,
    #[arg(long, default_value = "runs/hilg-wl/models/merge-policy.json")]
    pub checkpoint: PathBuf,
    #[arg(long)]
    pub interface: Option<PathBuf>,
    #[arg(long, default_value_t = 256)]
    pub train_problems: usize,
    #[arg(long, default_value_t = 32)]
    pub validation_problems: usize,
    #[arg(long, default_value_t = 1000)]
    pub total_ms: u64,
    #[arg(long, default_value_t = 350)]
    pub stage_ms: u64,
    #[arg(long, default_value_t = 100)]
    pub local_ms: u64,
    #[arg(long, default_value_t = 64)]
    pub local_calls: usize,
    #[arg(long, default_value_t = 6)]
    pub attempts: usize,
    #[arg(long, default_value_t = 12)]
    pub boundary_atoms: usize,
    #[arg(long, default_value_t = 200_000)]
    pub ground_instances: usize,
    #[arg(long, default_value_t = 0.1)]
    pub value_weight: f64,
    #[arg(long, default_value_t = 0.02)]
    pub effort_penalty: f64,
    #[arg(long, default_value_t = 0.05)]
    pub retry_penalty: f64,
}
#[derive(Debug, Clone, Serialize)]
struct Source {
    id: String,
    group: String,
}
#[derive(Deserialize)]
struct Request {
    model: MergeModel,
    split: Split,
    indices: Vec<usize>,
    seed: u64,
    sample: bool,
    #[serde(default)]
    online: Vec<Option<super::online::OnlineInput>>,
}
#[derive(Serialize)]
pub struct Episode {
    pub problem_id: String,
    pub group: String,
    pub split: Split,
    pub sampled: bool,
    pub online: bool,
    pub seed: u64,
    pub behavior_sha256: String,
    pub decisions: Vec<PolicyDecision>,
    pub probabilities: Vec<Vec<f64>>,
    pub reward: f64,
    pub log_cost: f64,
    pub solved: bool,
    pub replay_verified: bool,
    pub work: EscalationWork,
    pub attempts: usize,
    pub statuses: Vec<MasterStatus>,
    pub certificate: Option<crate::solver::PlanCertificate>,
}
struct Prepared {
    analysis: HtnAnalysis,
    graph: InteractionGraph,
    seeds: Vec<MicroPartition>,
    problem: GroundProblem,
}
fn round_robin(groups: BTreeMap<String, Vec<Source>>, limit: usize) -> Vec<Source> {
    let mut queues: Vec<_> = groups.into_values().map(VecDeque::from).collect();
    let mut result = Vec::new();
    while result.len() < limit {
        let before = result.len();
        for q in &mut queues {
            if result.len() == limit {
                break;
            }
            if let Some(s) = q.pop_front() {
                result.push(s);
            }
        }
        if result.len() == before {
            break;
        }
    }
    result
}
pub fn serve(options: &EnvironmentOptions) -> Result<()> {
    if options.train_problems == 0
        || options.validation_problems == 0
        || options.total_ms == 0
        || options.stage_ms == 0
        || options.local_ms == 0
        || options.local_calls == 0
        || !(1..=128).contains(&options.attempts)
        || options.ground_instances == 0
        || [
            options.value_weight,
            options.effort_penalty,
            options.retry_penalty,
        ]
        .iter()
        .any(|x| !x.is_finite() || *x < 0.0)
    {
        bail!("invalid RL environment budgets or reward options");
    }
    let parent: Checkpoint = serde_json::from_slice(&std::fs::read(&options.checkpoint)?)?;
    parent.validate()?;
    let interface: Option<InterfaceCheckpoint> = options
        .interface
        .as_ref()
        .map(|p| -> Result<_> {
            let c: InterfaceCheckpoint = serde_json::from_slice(&std::fs::read(p)?)?;
            c.validate()?;
            if c.snapshot_id != parent.snapshot_id
                || c.options.split_seed != parent.options.split_seed
            {
                bail!("interface snapshot mismatch");
            }
            Ok(c)
        })
        .transpose()?;
    let store = DatasetStore::open(&options.store)?;
    let snapshot = store.snapshot()?;
    if snapshot.fingerprint()? != parent.snapshot_id {
        bail!("RL store snapshot does not match the starting checkpoint");
    }
    let mut train = BTreeMap::<String, Vec<Source>>::new();
    let mut validation = BTreeMap::<String, Vec<Source>>::new();
    let entries: BTreeMap<_, _> = snapshot
        .entries
        .iter()
        .map(|e| (e.id.as_str(), e))
        .collect();
    for e in snapshot
        .entries
        .iter()
        .filter(|e| e.metadata.kind == RecordKind::Problem)
    {
        let group = e
            .metadata
            .group_id
            .as_ref()
            .context("problem has no group")?;
        let split = group_split(group, parent.options.split_seed);
        let target = match split {
            Split::Train if parent.training_groups.contains(group) => &mut train,
            Split::Validation if parent.validation_groups.contains(group) => &mut validation,
            _ => continue,
        };
        let source: super::rollout::ProblemRecord = store.load(e)?;
        let roots = source
            .program
            .problem
            .as_ref()
            .and_then(|p| p.init_tn.as_ref())
            .map_or(0, |tn| tn.tn.subtasks.len());
        if roots < 2 {
            continue;
        }
        if group != &sha256(&serde_json::to_vec(&source.program.domain)?) {
            bail!("RL source domain mismatch");
        }
        target.entry(group.clone()).or_default().push(Source {
            id: e.id.clone(),
            group: group.clone(),
        });
    }
    let train = round_robin(train, options.train_problems);
    let validation = round_robin(validation, options.validation_problems);
    if train.is_empty() || validation.is_empty() {
        bail!("RL needs nonempty train and validation domains");
    }
    let mut output = std::io::BufWriter::new(std::io::stdout().lock());
    serde_json::to_writer(
        &mut output,
        &serde_json::json!({"schema_version":1,"snapshot_id":parent.snapshot_id,"parent_sha256":sha256(&std::fs::read(&options.checkpoint)?),
        "interface_sha256":options.interface.as_ref().map(|p|std::fs::read(p).map(|b|sha256(&b))).transpose()?,
        "engine_sha256":sha256(concat!(include_str!("rl.rs"),include_str!("online.rs"),include_str!("../escalation.rs"),include_str!("policy.rs"),include_str!("../master.rs")).as_bytes()),"train":train,"validation":validation,"options":options}),
    )?;
    output.write_all(b"\n")?;
    output.flush()?;
    let mut cache = BTreeMap::<String, Prepared>::new();
    for line in std::io::stdin().lock().lines() {
        let request: Request = serde_json::from_str(&line?)?;
        request.model.validate()?;
        if request.model.schema_version != parent.model.schema_version
            || request.model.feature_version != parent.model.feature_version
            || serde_json::to_value(request.model.config)?
                != serde_json::to_value(parent.model.config)?
        {
            bail!("RL architecture changed");
        }
        let sources = match request.split {
            Split::Train => &train,
            Split::Validation if !request.sample => &validation,
            _ => bail!(
                "sampling/training is restricted to Train domains; Test domains are unavailable"
            ),
        };
        if request.indices.is_empty() || request.indices.len() > 128 {
            bail!("RL request must contain 1..128 episodes");
        }
        if !request.online.is_empty()
            && (request.online.len() != request.indices.len() || request.split != Split::Train)
        {
            bail!("online inputs must align with Train episodes");
        }
        let mut episodes = Vec::new();
        for (offset, &index) in request.indices.iter().enumerate() {
            let input = request.online.get(offset).and_then(Option::as_ref);
            let extra = input.map(|i| i.decode(&parent)).transpose()?;
            let source = if let (Some(input), Some(record)) = (input, &extra) {
                Source {
                    id: input.id.clone(),
                    group: record.group.clone(),
                }
            } else {
                sources
                    .get(index)
                    .context("RL source index out of range")?
                    .clone()
            };
            if !cache.contains_key(&source.id) {
                if cache.len() >= 32 {
                    cache.pop_first();
                }
                let program = if let Some(record) = &extra {
                    record.program.clone()
                } else {
                    let record: super::rollout::ProblemRecord =
                        store.load(entries[source.id.as_str()])?;
                    record.program
                };
                let analysis = analyze_program(&program)?;
                let graph = InteractionGraph::from_analysis(&analysis);
                let seeds = micropartition_network(
                    &graph,
                    NetworkId::Initial,
                    &SpectralOptions {
                        max_cluster_size: 1,
                        ..Default::default()
                    },
                )?
                .clusters;
                let problem = ground_reachable_program(
                    &program,
                    &GroundingOptions {
                        max_instances: options.ground_instances,
                    },
                )?;
                if let Some(record) = &extra {
                    crate::solver::validate_problem_certificate(&problem, &record.certificate)?;
                }
                cache.insert(
                    source.id.clone(),
                    Prepared {
                        analysis,
                        graph,
                        seeds,
                        problem,
                    },
                );
            }
            let p = &cache[&source.id];
            let seed = request.seed.wrapping_add(offset as u64);
            let mut rng = seed;
            let mut choose = |probabilities: &[f64]| {
                let u = (random(&mut rng) >> 11) as f64 / (1u64 << 53) as f64;
                let mut cumulative = 0.0;
                for (i, &v) in probabilities.iter().enumerate() {
                    cumulative += v;
                    if u < cumulative {
                        return i;
                    }
                }
                probabilities.len() - 1
            };
            let mut master = MasterOptions {
                max_time_ms: options.total_ms,
                max_local_calls: options.local_calls,
                max_boundary_atoms: options.boundary_atoms,
                ..Default::default()
            };
            master.local_budget.max_time_ms = options.local_ms;
            let merge = AgglomerationOptions::default();
            let (state, trace) = agglomerate_with_selector(
                &p.analysis,
                &p.graph,
                &p.seeds,
                &merge,
                &request.model,
                options.value_weight,
                &master,
                if request.sample {
                    Some(&mut choose)
                } else {
                    None
                },
            )?;
            let result = solve_escalating(
                &p.problem,
                &p.analysis,
                &p.graph,
                &state.clusters,
                &merge,
                &master,
                &MasterGuidance {
                    interface: interface
                        .as_ref()
                        .map(|i| &i.model as &dyn crate::master::InterfaceGuide),
                    conflict_core_checks: 2,
                },
                &EscalationOptions {
                    max_attempts: options.attempts,
                    attempt_time_ms: options.stage_ms,
                    max_attempt_time_ms: options.total_ms.max(options.stage_ms),
                    ..Default::default()
                },
                TraceLevel::Summary,
            )?;
            let solved = result.result.status == MasterStatus::Feasible;
            let cost = result
                .work
                .local_expanded
                .saturating_add(result.work.master_nodes)
                .saturating_add(result.work.local_calls);
            let log_cost = (cost as f64).ln_1p();
            let reward = if solved { 1.0 } else { -1.0 }
                - options.effort_penalty * log_cost
                - options.retry_penalty * result.attempts.len().saturating_sub(1) as f64;
            let probabilities = trace
                .decisions
                .iter()
                .map(|d| action_probabilities(&d.prediction, options.value_weight))
                .collect::<Result<_>>()?;
            episodes.push(Episode {
                problem_id: source.id.clone(),
                group: source.group.clone(),
                split: request.split,
                sampled: request.sample,
                online: extra.is_some(),
                seed,
                behavior_sha256: sha256(&serde_json::to_vec(&request.model)?),
                decisions: trace.decisions,
                probabilities,
                reward,
                log_cost,
                solved,
                replay_verified: solved,
                work: result.work,
                attempts: result.attempts.len(),
                statuses: result.attempts.iter().map(|a| a.status).collect(),
                certificate: result.result.certificate,
            });
        }
        serde_json::to_writer(&mut output, &episodes)?;
        output.write_all(b"\n")?;
        output.flush()?;
    }
    Ok(())
}
#[derive(Deserialize)]
struct Parity {
    observation: Observation,
    prediction: Prediction,
}
pub fn validate_export(parent: &Path, checkpoint: &Path, parity: &Path) -> Result<()> {
    let old: Checkpoint = serde_json::from_slice(&std::fs::read(parent)?)?;
    let new: Checkpoint = serde_json::from_slice(&std::fs::read(checkpoint)?)?;
    old.validate()?;
    new.validate()?;
    if old.options.seed != new.options.seed
        || old.options.split_seed != new.options.split_seed
        || old.snapshot_id != new.snapshot_id
        || old.training_groups != new.training_groups
        || old.validation_groups != new.validation_groups
        || old.test_groups != new.test_groups
        || old.training_records != new.training_records
        || old.validation_records != new.validation_records
        || old.test_records != new.test_records
        || old.model.schema_version != new.model.schema_version
        || old.model.feature_version != new.model.feature_version
        || serde_json::to_value(old.model.config)? != serde_json::to_value(new.model.config)?
    {
        bail!("RL checkpoint source/architecture provenance changed");
    }
    let checks: Vec<Parity> = serde_json::from_slice(&std::fs::read(parity)?)?;
    if checks.is_empty() {
        bail!("RL parity references are empty");
    }
    for c in checks {
        let prediction = new.model.predict(&c.observation)?;
        for (a, b) in [
            (&prediction.logits, &c.prediction.logits),
            (&prediction.log_effort, &c.prediction.log_effort),
        ] {
            if a.len() != b.len()
                || a.iter().zip(b).any(|(x, y)| {
                    !x.is_finite() || !y.is_finite() || (x - y).abs() > 1e-4 * (1.0 + y.abs())
                })
            {
                bail!("RL native inference parity mismatch");
            }
        }
    }
    Ok(())
}
