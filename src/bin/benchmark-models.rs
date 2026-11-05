//! Paired held-out planner ablations; timings include partitioning and inference.
use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use ptu_plan::{
    agglomeration::{AgglomerationOptions, greedy_agglomerate},
    analysis::{NetworkId, analyze_program},
    instrumentation::TraceLevel,
    interaction::InteractionGraph,
    learning::{
        beam::{BeamOptions, beam_agglomerate_with_latent},
        interface::InterfaceCheckpoint,
        latent_training::LatentCheckpoint,
        policy::learned_agglomerate_with_context,
        rollout::ProblemRecord,
        training::Checkpoint,
    },
    master::{
        MasterGuidance, MasterOptions, MasterStatistics, MasterStatus, initial_partitions,
        solve_sequential_guided,
    },
    partition::{SpectralOptions, micropartition_network},
    solver::{
        GroundingMode, GroundingOptions, ground_reachable_program, validate_problem_certificate,
    },
    store::{DatasetStore, RecordKind, Split, group_split},
    training::sha256,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    fs,
    io::Write,
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
enum Variant {
    Singleton,
    Greedy,
    GreedyInterface,
    Policy,
    PolicyInterface,
    BeamExact,
    BeamExactInterface,
    BeamLatentInterface,
    PolicyEscalatingInterface,
    BeamEscalatingInterface,
}
impl Variant {
    fn interface(self) -> bool {
        matches!(
            self,
            Self::GreedyInterface
                | Self::PolicyInterface
                | Self::BeamExactInterface
                | Self::BeamLatentInterface
                | Self::PolicyEscalatingInterface
                | Self::BeamEscalatingInterface
        )
    }
}
#[derive(Debug, Clone, Parser, Serialize, Deserialize)]
#[command(
    about = "Compare trained planner components on held-out source problems, with paired budgets and replay validation"
)]
struct Options {
    #[arg(long, default_value = "runs/full/data")]
    store: PathBuf,
    #[arg(long, default_value = "runs/full/models")]
    models: PathBuf,
    #[arg(long, default_value = "runs/full/benchmarks/heldout-350ms.json")]
    output: PathBuf,
    /// Zero selects every held-out source problem. Positive limits round-robin domains.
    #[arg(long, default_value_t = 0)]
    limit: usize,
    /// Use exactly the source IDs from a completed benchmark, preserving domain holdouts.
    #[arg(long)]
    #[serde(default)]
    cohort: Option<PathBuf>,
    #[arg(long, default_value_t = 3)]
    repeats: usize,
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "singleton,greedy,greedy-interface,policy,policy-interface,beam-exact,beam-exact-interface,beam-latent-interface"
    )]
    variants: Vec<Variant>,
    #[arg(long, default_value_t = 350)]
    master_time_ms: u64,
    #[arg(long, default_value_t = 100)]
    local_time_ms: u64,
    #[arg(long, default_value_t = 64)]
    max_local_calls: usize,
    #[arg(long, default_value_t = 12)]
    max_boundary_atoms: usize,
    #[arg(long, default_value_t = 100)]
    #[serde(default = "default_escalation_stage")]
    escalation_stage_ms: u64,
    #[arg(long, default_value_t = 2)]
    core_checks: usize,
    #[arg(long, default_value_t = 200_000)]
    max_ground_instances: usize,
    #[arg(long, default_value_t = 0.1)]
    value_weight: f64,
    #[arg(long, default_value_t = 8)]
    beam_width: usize,
    #[arg(long, default_value_t = 3)]
    beam_depth: usize,
    #[arg(long, default_value_t = 8)]
    beam_branching: usize,
    #[arg(long, default_value_t = 10_000)]
    beam_max_evaluations: usize,
    #[arg(long)]
    #[serde(skip)]
    resume: bool,
}
fn default_escalation_stage() -> u64 {
    100
}

#[derive(Debug, Serialize, Deserialize)]
struct PolicyStats {
    stop_reason: String,
    decisions: usize,
    beam: Option<serde_json::Value>,
}
#[derive(Debug, Serialize, Deserialize)]
struct Run {
    variant: Variant,
    repetition: usize,
    order: usize,
    partition_us: u128,
    master_us: u128,
    planning_us: u128,
    partitions: Vec<Vec<usize>>,
    status: Option<MasterStatus>,
    statistics: MasterStatistics,
    plan_actions: Option<usize>,
    replay_verified: bool,
    policy: Option<PolicyStats>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    escalation: Option<ptu_plan::escalation::EscalationResult>,
    error: Option<String>,
}
impl Run {
    fn new(variant: Variant, repetition: usize, order: usize) -> Self {
        Self {
            variant,
            repetition,
            order,
            partition_us: 0,
            master_us: 0,
            planning_us: 0,
            partitions: vec![],
            status: None,
            statistics: MasterStatistics::default(),
            plan_actions: None,
            replay_verified: false,
            policy: None,
            escalation: None,
            error: None,
        }
    }
}
#[derive(Debug, Serialize, Deserialize)]
struct Case {
    problem_id: String,
    #[serde(default)]
    problem_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    cohort_problem_id: Option<String>,
    group_id: String,
    domain: String,
    name: String,
    roots: usize,
    preparation_us: u128,
    error: Option<String>,
    runs: Vec<Run>,
}
#[derive(Debug, Serialize, Deserialize)]
struct Report {
    schema_version: u32,
    options: Options,
    snapshot_id: String,
    #[serde(default)]
    cohort_sha256: Option<String>,
    model_sha256: BTreeMap<String, String>,
    master: MasterOptions,
    agglomeration: AgglomerationOptions,
    beam: BeamOptions,
    source_problems: usize,
    source_groups: Vec<String>,
    machine: serde_json::Value,
    warmup_cases: usize,
    complete: bool,
    cases: Vec<Case>,
}
fn read<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}
fn validate_pairing(case: &Case) -> Result<()> {
    for (a, b) in [
        (Variant::Greedy, Variant::GreedyInterface),
        (Variant::Policy, Variant::PolicyInterface),
        (Variant::BeamExact, Variant::BeamExactInterface),
    ] {
        for left in case
            .runs
            .iter()
            .filter(|r| r.variant == a && r.error.is_none())
        {
            if let Some(right) = case
                .runs
                .iter()
                .find(|r| r.variant == b && r.repetition == left.repetition && r.error.is_none())
                && left.partitions != right.partitions
            {
                bail!("interface ablation changed partition selection");
            }
        }
    }
    for (i, a) in case
        .runs
        .iter()
        .enumerate()
        .filter(|(_, r)| r.error.is_none())
    {
        for b in case.runs[..i]
            .iter()
            .filter(|r| r.error.is_none() && r.partitions == a.partitions)
        {
            if (a.status == Some(MasterStatus::Feasible)
                && b.status == Some(MasterStatus::SequentiallyInfeasible))
                || (b.status == Some(MasterStatus::Feasible)
                    && a.status == Some(MasterStatus::SequentiallyInfeasible))
            {
                bail!("contradictory feasibility for identical partitions");
            }
        }
    }
    Ok(())
}
fn save(path: &Path, report: &Report) -> Result<()> {
    let temporary = path.with_extension("json.tmp");
    let mut file = fs::File::create(&temporary)?;
    serde_json::to_writer(&mut file, report)?;
    file.flush()?;
    file.sync_all()?;
    drop(file);
    fs::rename(temporary, path)?;
    Ok(())
}
struct Lock(fs::File);
impl Drop for Lock {
    fn drop(&mut self) {
        let _ = self.0.unlock();
    }
}

fn run() -> Result<()> {
    let options = Options::parse();
    if options.repeats == 0
        || options.repeats > 32
        || options.variants.is_empty()
        || options.master_time_ms == 0
        || options.local_time_ms == 0
        || options.max_local_calls == 0
        || !options.value_weight.is_finite()
        || options.value_weight < 0.0
    {
        bail!("invalid benchmark counts, budgets or value weight");
    }
    if options
        .variants
        .iter()
        .enumerate()
        .any(|(i, v)| options.variants[..i].contains(v))
    {
        bail!("duplicate benchmark variant");
    }
    fs::create_dir_all(options.output.parent().unwrap_or(Path::new(".")))?;
    let file = fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(options.output.with_extension("lock"))?;
    file.try_lock()
        .context("another benchmark owns this output")?;
    let _lock = Lock(file);
    let merge: Checkpoint = read(&options.models.join("merge-policy.json"))?;
    merge.validate()?;
    let interface: InterfaceCheckpoint = read(&options.models.join("interface-policy.json"))?;
    interface.validate()?;
    let latent: LatentCheckpoint = read(&options.models.join("latent-policy.json"))?;
    latent.validate()?;
    latent.model.validate(&merge.model)?;
    if [interface.options.split_seed, latent.options.split_seed]
        .iter()
        .any(|s| *s != merge.options.split_seed)
        || interface.snapshot_id != merge.snapshot_id
        || latent.snapshot_id != merge.snapshot_id
    {
        bail!("benchmark model provenance mismatch");
    }
    let store = DatasetStore::open(&options.store)?;
    let snapshot = store.snapshot()?;
    let snapshot_id = snapshot.fingerprint()?;
    if snapshot_id != merge.snapshot_id {
        bail!("benchmark store differs from the models' source snapshot");
    }
    let model_sha256 = [
        "merge-policy.json",
        "interface-policy.json",
        "latent-policy.json",
    ]
    .into_iter()
    .map(|name| {
        Ok((
            name.to_string(),
            sha256(&fs::read(options.models.join(name))?),
        ))
    })
    .collect::<Result<BTreeMap<_, _>>>()?;
    let heldout: BTreeSet<_> = merge.test_groups.iter().cloned().collect();
    if heldout.is_empty() {
        bail!("checkpoint has no held-out domains");
    }
    let mut groups = BTreeMap::<String, VecDeque<_>>::new();
    for entry in snapshot
        .entries
        .iter()
        .filter(|e| e.metadata.kind == RecordKind::Problem)
    {
        let group = entry
            .metadata
            .group_id
            .as_ref()
            .context("source lacks group")?;
        if heldout.contains(group) {
            if group_split(group, merge.options.split_seed) != Split::Test {
                bail!("source is not held out");
            }
            groups
                .entry(group.clone())
                .or_default()
                .push_back(entry.clone());
        }
    }
    let mut sources = Vec::new();
    loop {
        let before = sources.len();
        for group in groups.values_mut() {
            if options.limit > 0 && sources.len() >= options.limit {
                break;
            }
            if let Some(entry) = group.pop_front() {
                sources.push(entry);
            }
        }
        if sources.len() == before || (options.limit > 0 && sources.len() >= options.limit) {
            break;
        }
    }
    let mut cohort_ids = BTreeMap::new();
    let cohort_sha256 = if let Some(path) = &options.cohort {
        if options.limit > 0 {
            bail!("--cohort cannot be combined with --limit");
        }
        let bytes = fs::read(path)?;
        let cohort: serde_json::Value = serde_json::from_slice(&bytes)?;
        if cohort["complete"] != true {
            bail!("cohort report must be complete");
        }
        let cases = cohort["cases"].as_array().context("cohort lacks cases")?;
        let available: BTreeMap<_, _> = snapshot
            .entries
            .iter()
            .filter(|e| e.metadata.kind == RecordKind::Problem)
            .map(|e| (e.id.as_str(), e))
            .collect();
        let missing = cases.iter().any(|c| {
            c["problem_id"]
                .as_str()
                .is_none_or(|id| !available.contains_key(id))
        });
        let mut by_payload = BTreeMap::<String, Vec<ptu_plan::store::Entry>>::new();
        if missing {
            for entry in available.values().filter(|e| {
                e.metadata
                    .group_id
                    .as_ref()
                    .is_some_and(|g| heldout.contains(g))
            }) {
                let source: ProblemRecord = store.load(entry)?;
                by_payload
                    .entry(sha256(&serde_json::to_vec(&source)?))
                    .or_default()
                    .push((*entry).clone());
            }
        }
        let legacy_store = if cases.iter().any(|c| {
            c["problem_id"]
                .as_str()
                .is_some_and(|id| !available.contains_key(id))
                && c["problem_sha256"].as_str().is_none_or(str::is_empty)
        }) {
            let path = Path::new(
                cohort["options"]["store"]
                    .as_str()
                    .context("legacy cohort lacks its source store path")?,
            );
            if !path.is_dir() {
                bail!(
                    "legacy cohort source store is unavailable; use a cohort report with problem_sha256 fingerprints"
                );
            }
            let source = DatasetStore::open(path)?;
            let index = source.snapshot()?;
            Some((source, index))
        } else {
            None
        };
        sources.clear();
        let mut seen = BTreeSet::new();
        let mut selected = BTreeSet::new();
        for case in cases {
            let id = case["problem_id"]
                .as_str()
                .context("cohort case lacks problem ID")?;
            if !seen.insert(id) {
                bail!("duplicate cohort problem");
            }
            let entry = if let Some(entry) = available.get(id) {
                (*entry).clone()
            } else {
                let hash =
                    if let Some(hash) = case["problem_sha256"].as_str().filter(|s| !s.is_empty()) {
                        hash.to_owned()
                    } else {
                        let (original, index) = legacy_store
                            .as_ref()
                            .context("legacy source store unavailable")?;
                        let old = index
                            .entries
                            .iter()
                            .find(|e| e.id == id && e.metadata.kind == RecordKind::Problem)
                            .context("original cohort source is missing")?;
                        let source: ProblemRecord = original.load(old)?;
                        sha256(&serde_json::to_vec(&source)?)
                    };
                let candidates = by_payload
                    .get(&hash)
                    .context("cohort program missing from the selected store")?;
                if candidates.len() != 1 {
                    bail!("ambiguous cohort program identity");
                }
                candidates[0].clone()
            };
            if !selected.insert(entry.id.clone()) {
                bail!("duplicate cohort program");
            }
            let group = entry
                .metadata
                .group_id
                .as_ref()
                .context("cohort source lacks group")?;
            if !heldout.contains(group)
                || group_split(group, merge.options.split_seed) != Split::Test
            {
                bail!("cohort source is not a held-out test domain");
            }
            cohort_ids.insert(entry.id.clone(), id.to_owned());
            sources.push(entry);
        }
        Some(sha256(&bytes))
    } else {
        None
    };
    if sources.is_empty() {
        bail!("no held-out source problems");
    }
    let source_groups = sources
        .iter()
        .filter_map(|e| e.metadata.group_id.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let mut master = MasterOptions {
        max_time_ms: options.master_time_ms,
        max_local_calls: options.max_local_calls,
        max_boundary_atoms: options.max_boundary_atoms,
        ..Default::default()
    };
    master.local_budget.max_time_ms = options.local_time_ms;
    let beam = BeamOptions {
        width: options.beam_width,
        depth: options.beam_depth,
        branching: options.beam_branching,
        max_evaluations: options.beam_max_evaluations,
    };
    let mut report = Report {
        schema_version: 1,
        options: options.clone(),
        snapshot_id,
        cohort_sha256,
        model_sha256,
        master,
        agglomeration: AgglomerationOptions::default(),
        beam,
        source_problems: sources.len(),
        source_groups,
        machine: serde_json::json!({"os":std::env::consts::OS,"arch":std::env::consts::ARCH,"available_threads":std::thread::available_parallelism()?.get(),"inference":"native Rust CPU, serial","cpu_model":fs::read_to_string("/proc/cpuinfo").ok().and_then(|s|s.lines().find(|l|l.starts_with("model name")).map(String::from))}),
        warmup_cases: 0,
        complete: false,
        cases: vec![],
    };
    if options.output.exists() {
        if !options.resume {
            bail!("output exists; use --resume or a new --output");
        }
        let old: Report = read(&options.output)?;
        if serde_json::to_value(&old.options)? != serde_json::to_value(&options)?
            || old.snapshot_id != report.snapshot_id
            || old.cohort_sha256 != report.cohort_sha256
            || old.model_sha256 != report.model_sha256
        {
            bail!("resume configuration/data/model mismatch");
        }
        report = old;
    }
    let finished = report
        .cases
        .iter()
        .map(|c| c.problem_id.clone())
        .collect::<BTreeSet<_>>();
    for (case_index, entry) in sources.iter().enumerate() {
        if finished.contains(&entry.id) {
            continue;
        }
        let source: ProblemRecord = store.load(entry)?;
        let group = entry.metadata.group_id.clone().unwrap();
        if source.grounding_mode != GroundingMode::InitialReachable
            || group != sha256(&serde_json::to_vec(&source.program.domain)?)
        {
            bail!("source grounding/group mismatch");
        }
        let mut case = Case {
            problem_id: entry.id.clone(),
            problem_sha256: sha256(&serde_json::to_vec(&source)?),
            cohort_problem_id: cohort_ids.get(&entry.id).cloned(),
            group_id: group,
            domain: source.program.domain.name.clone(),
            name: source
                .program
                .problem
                .as_ref()
                .map_or_else(String::new, |p| p.name.clone()),
            roots: source
                .program
                .problem
                .as_ref()
                .and_then(|p| p.init_tn.as_ref())
                .map_or(0, |t| t.tn.subtasks.len()),
            preparation_us: 0,
            error: None,
            runs: vec![],
        };
        let preparation = Instant::now();
        let prepared = (|| -> Result<_> {
            let analysis = analyze_program(&source.program)?;
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
                &source.program,
                &GroundingOptions {
                    max_instances: options.max_ground_instances,
                },
            )?;
            Ok((analysis, graph, seeds, problem))
        })();
        case.preparation_us = preparation.elapsed().as_micros();
        match prepared {
            Err(error) => case.error = Some(format!("{error:#}")),
            Ok((analysis, graph, seeds, problem)) => {
                // All variants execute once before recorded runs, warming their
                // code paths equally. No learned weights or caches persist.
                let warmup = report.warmup_cases == 0;
                for repetition in 0..options.repeats + usize::from(warmup) {
                    let warming = warmup && repetition == 0;
                    let repeat = repetition - usize::from(warmup && !warming);
                    for order in 0..options.variants.len() {
                        let rotation = (case_index + repeat) % options.variants.len();
                        let slot = if repeat % 2 == 0 {
                            order
                        } else {
                            options.variants.len() - 1 - order
                        };
                        let variant = options.variants[(rotation + slot) % options.variants.len()];
                        let started = Instant::now();
                        let mut sample = Run::new(variant, repeat, order);
                        let outcome = (|| -> Result<()> {
                            let (clusters, trace) = match variant {
                                Variant::Singleton => (seeds.clone(), None),
                                Variant::Greedy | Variant::GreedyInterface => (
                                    greedy_agglomerate(
                                        &analysis,
                                        &graph,
                                        NetworkId::Initial,
                                        &seeds,
                                        &report.agglomeration,
                                    )?
                                    .clusters,
                                    None,
                                ),
                                Variant::Policy
                                | Variant::PolicyInterface
                                | Variant::PolicyEscalatingInterface => {
                                    let (a, t) = learned_agglomerate_with_context(
                                        &analysis,
                                        &graph,
                                        &seeds,
                                        &report.agglomeration,
                                        &merge.model,
                                        options.value_weight,
                                        &report.master,
                                    )?;
                                    (a.clusters, Some(t))
                                }
                                _ => {
                                    let (a, t) = beam_agglomerate_with_latent(
                                        &analysis,
                                        &graph,
                                        &seeds,
                                        &report.agglomeration,
                                        &merge.model,
                                        options.value_weight,
                                        &report.master,
                                        &report.beam,
                                        if matches!(
                                            variant,
                                            Variant::BeamLatentInterface
                                                | Variant::BeamEscalatingInterface
                                        ) {
                                            Some(&latent.model)
                                        } else {
                                            None
                                        },
                                    )?;
                                    (a.clusters, Some(t))
                                }
                            };
                            let partitions = initial_partitions(&graph, &clusters)?;
                            sample.partition_us = started.elapsed().as_micros();
                            sample.partitions =
                                partitions.iter().map(|p| p.roots.clone()).collect();
                            sample.policy = trace.map(|t| PolicyStats {
                                stop_reason: format!("{:?}", t.stop_reason),
                                decisions: t.decisions.len(),
                                beam: t.beam.map(|b| serde_json::to_value(b).unwrap()),
                            });
                            let search = Instant::now();
                            let guidance = MasterGuidance {
                                interface: if variant.interface() {
                                    Some(&interface.model)
                                } else {
                                    None
                                },
                                conflict_core_checks: options.core_checks,
                            };
                            let result = if matches!(
                                variant,
                                Variant::PolicyEscalatingInterface
                                    | Variant::BeamEscalatingInterface
                            ) {
                                let outcome = ptu_plan::escalation::solve_escalating(
                                    &problem,
                                    &analysis,
                                    &graph,
                                    &clusters,
                                    &report.agglomeration,
                                    &report.master,
                                    &guidance,
                                    &ptu_plan::escalation::EscalationOptions {
                                        attempt_time_ms: options.escalation_stage_ms,
                                        max_attempt_time_ms: options.escalation_stage_ms,
                                        ..Default::default()
                                    },
                                    TraceLevel::Summary,
                                )?;
                                if let Some(last) = outcome.attempts.last() {
                                    sample.partitions =
                                        last.partitions.iter().map(|p| p.roots.clone()).collect();
                                }
                                let mut result = outcome.result.clone();
                                result.statistics = outcome.total_statistics();
                                sample.escalation = Some(outcome);
                                result
                            } else {
                                solve_sequential_guided(
                                    &problem,
                                    &partitions,
                                    &report.master,
                                    &guidance,
                                    TraceLevel::Summary,
                                    None,
                                )?
                            };
                            sample.master_us = search.elapsed().as_micros();
                            sample.planning_us = started.elapsed().as_micros();
                            sample.status = Some(result.status);
                            sample.statistics = result.statistics;
                            if let Some(certificate) = result.certificate {
                                validate_problem_certificate(&problem, &certificate)?;
                                sample.replay_verified = true;
                                sample.plan_actions = Some(certificate.actions.len());
                            }
                            if (result.status == MasterStatus::Feasible) != sample.replay_verified {
                                bail!("status/certificate mismatch");
                            }
                            Ok(())
                        })();
                        if let Err(error) = outcome {
                            sample.error = Some(format!("{error:#}"));
                            sample.planning_us = started.elapsed().as_micros();
                        }
                        if !warming {
                            case.runs.push(sample);
                        }
                    }
                }
                if warmup {
                    report.warmup_cases += 1;
                }
            }
        }
        eprintln!(
            "benchmark {}/{}: {} {} roots, {} samples{}",
            case_index + 1,
            sources.len(),
            case.domain,
            case.roots,
            case.runs.len(),
            if case.error.is_some() {
                " preparation error"
            } else {
                ""
            }
        );
        let pairing = validate_pairing(&case);
        report.cases.push(case);
        save(&options.output, &report)?;
        pairing?;
    }
    report.complete = true;
    save(&options.output, &report)?;
    println!(
        "{} held-out cases, {} variants, {} repeats -> {}",
        report.cases.len(),
        options.variants.len(),
        options.repeats,
        options.output.display()
    );
    Ok(())
}
fn main() -> Result<()> {
    run()
}
