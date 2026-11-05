//! Paired master-only pilot; all generated cut proofs are independently checked.
use anyhow::{Context, Result, bail};
use clap::Parser;
use ptu_plan::{
    analysis::{NetworkId, analyze_program},
    conflicts::validate_conflict_core,
    instrumentation::TraceLevel,
    interaction::InteractionGraph,
    learning::rollout::ProblemRecord,
    master::{
        MasterGuidance, MasterOptions, MasterResult, MasterStatus, initial_partitions,
        solve_sequential_guided,
    },
    partition::{SpectralOptions, micropartition_network},
    solver::{GroundingMode, GroundingOptions, ground_reachable_program},
    store::{DatasetStore, RecordKind},
    training::sha256,
};
use serde::Serialize;
use std::{collections::BTreeMap, path::PathBuf, time::Instant};

#[derive(Parser)]
#[command(
    about = "Compare singleton-partition master search with and without Benders cuts, verifying every retained cut"
)]
struct Options {
    #[arg(long, default_value = "target/rollouts")]
    store: PathBuf,
    #[arg(long, default_value = "target/benders-comparison.json")]
    output: PathBuf,
    #[arg(long, default_value_t = 30)]
    limit: usize,
    #[arg(long, default_value_t = 4)]
    core_checks: usize,
    #[arg(long, default_value_t = 500)]
    master_time_ms: u64,
    #[arg(long, default_value_t = 100)]
    local_time_ms: u64,
    #[arg(long, default_value_t = 64)]
    max_local_calls: usize,
    #[arg(long, default_value_t = 12)]
    max_boundary_atoms: usize,
    #[arg(long, default_value_t = 200_000)]
    max_ground_instances: usize,
    #[arg(long, default_value_t = 5000)]
    verification_time_ms: u64,
}
#[derive(Serialize)]
struct Case {
    problem_id: String,
    group_id: Option<String>,
    name: String,
    cores_first: bool,
    preparation_ms: u128,
    error: Option<String>,
    baseline: Option<MasterResult>,
    cores: Option<MasterResult>,
    verified_cuts: usize,
}
#[derive(Default, Serialize)]
struct Totals {
    statuses: BTreeMap<String, usize>,
    local_calls: usize,
    local_expanded: usize,
    interface_candidates: usize,
    core_checks: usize,
    core_hits: usize,
    generalized_hits: usize,
    assignments_skipped: usize,
}
impl Totals {
    fn add(&mut self, result: &MasterResult) {
        *self
            .statuses
            .entry(format!("{:?}", result.status))
            .or_default() += 1;
        let s = &result.statistics;
        self.local_calls += s.local_calls;
        self.local_expanded += s.local_expanded;
        self.interface_candidates += s.interface_candidates;
        self.core_checks += s.core_checks;
        self.core_hits += s.core_hits;
        self.generalized_hits += s.core_generalized_hits;
        self.assignments_skipped = self
            .assignments_skipped
            .saturating_add(s.core_assignments_skipped);
    }
}
#[derive(Serialize)]
struct Report {
    schema_version: u32,
    snapshot_id: String,
    master: MasterOptions,
    core_checks: usize,
    max_ground_instances: usize,
    verification_time_ms: u64,
    baseline: Totals,
    cores: Totals,
    cases: Vec<Case>,
}
fn run() -> Result<()> {
    let options = Options::parse();
    if options.limit == 0 || options.core_checks == 0 {
        bail!("limit and core checks must be positive");
    }
    let store = DatasetStore::open(&options.store)?;
    let snapshot = store.snapshot()?;
    let mut master = MasterOptions {
        max_time_ms: options.master_time_ms,
        max_local_calls: options.max_local_calls,
        max_boundary_atoms: options.max_boundary_atoms,
        ..Default::default()
    };
    master.local_budget.max_time_ms = options.local_time_ms;
    let mut report = Report {
        schema_version: 1,
        snapshot_id: snapshot.fingerprint()?,
        master,
        core_checks: options.core_checks,
        max_ground_instances: options.max_ground_instances,
        verification_time_ms: options.verification_time_ms,
        baseline: Totals::default(),
        cores: Totals::default(),
        cases: Vec::new(),
    };
    for (number, entry) in snapshot
        .entries
        .iter()
        .filter(|e| e.metadata.kind == RecordKind::Problem)
        .take(options.limit)
        .enumerate()
    {
        let problem: ProblemRecord = store.load(entry)?;
        if problem.grounding_mode != GroundingMode::InitialReachable
            || entry.metadata.group_id.as_deref()
                != Some(&sha256(&serde_json::to_vec(&problem.program.domain)?))
        {
            bail!("benchmark source group/grounding mismatch");
        }
        let start = Instant::now();
        let prepared = (|| -> Result<_> {
            let analysis = analyze_program(&problem.program)?;
            let graph = InteractionGraph::from_analysis(&analysis);
            let clusters = micropartition_network(
                &graph,
                NetworkId::Initial,
                &SpectralOptions {
                    max_cluster_size: 1,
                    ..Default::default()
                },
            )?
            .clusters;
            let partitions = initial_partitions(&graph, &clusters)?;
            let model = ground_reachable_program(
                &problem.program,
                &GroundingOptions {
                    max_instances: options.max_ground_instances,
                },
            )?;
            Ok((model, partitions))
        })();
        let mut case = Case {
            problem_id: entry.id.clone(),
            group_id: entry.metadata.group_id.clone(),
            name: problem
                .program
                .problem
                .as_ref()
                .map_or_else(|| problem.program.domain.name.clone(), |p| p.name.clone()),
            cores_first: number % 2 == 1,
            preparation_ms: start.elapsed().as_millis(),
            error: None,
            baseline: None,
            cores: None,
            verified_cuts: 0,
        };
        match prepared {
            Err(error) => case.error = Some(format!("{error:#}")),
            Ok((model, partitions)) => {
                let order = if case.cores_first {
                    [true, false]
                } else {
                    [false, true]
                };
                for enabled in order {
                    let result = solve_sequential_guided(
                        &model,
                        &partitions,
                        &master,
                        &MasterGuidance {
                            interface: None,
                            conflict_core_checks: if enabled { options.core_checks } else { 0 },
                        },
                        TraceLevel::Summary,
                        None,
                    )?;
                    if enabled {
                        let mut budget = master.local_budget;
                        budget.max_time_ms = options.verification_time_ms;
                        for cut in &result.conflict_cores {
                            let partition = partitions
                                .iter()
                                .find(|p| p.id == cut.partition)
                                .context("unknown cut partition")?;
                            validate_conflict_core(&model, partition, cut, &budget).with_context(
                                || format!("independent cut verification for {}", entry.id),
                            )?;
                            case.verified_cuts += 1;
                        }
                        report.cores.add(&result);
                        case.cores = Some(result);
                    } else {
                        report.baseline.add(&result);
                        case.baseline = Some(result);
                    }
                }
                let a = case.baseline.as_ref().unwrap().status;
                let b = case.cores.as_ref().unwrap().status;
                if (a == MasterStatus::Feasible && b == MasterStatus::SequentiallyInfeasible)
                    || (b == MasterStatus::Feasible && a == MasterStatus::SequentiallyInfeasible)
                {
                    bail!("contradictory partition-level results for {}", entry.id);
                }
            }
        }
        report.cases.push(case);
    }
    if report.cases.is_empty() {
        bail!("no source problems in snapshot");
    }
    if let Some(parent) = options
        .output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&options.output, serde_json::to_vec_pretty(&report)?)?;
    println!(
        "{} paired problems; {} independent cut verifications -> {}",
        report.cases.len(),
        report.cases.iter().map(|c| c.verified_cuts).sum::<usize>(),
        options.output.display()
    );
    println!(
        "{}",
        serde_json::to_string(
            &serde_json::json!({"baseline":report.baseline,"cores":report.cores})
        )?
    );
    Ok(())
}
fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("benchmark-cores: {e:#}");
            std::process::ExitCode::FAILURE
        }
    }
}
