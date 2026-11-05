use anyhow::{Context, Result, bail};
use clap::Parser;
use ptu_plan::instrumentation::TraceLevel;
use ptu_plan::learning::rollout::{RolloutOptions, collect_program};
use ptu_plan::parsing::json::Program;
use ptu_plan::store::{DatasetStore, SolutionSnapshot};
use std::collections::{BTreeMap, VecDeque};
use std::fs;
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    about = "Collect bounded counterfactual merge/STOP outcomes into an immutable experiment store"
)]
struct Options {
    /// Whole-program JSON files or existing solution record JSON files
    inputs: Vec<PathBuf>,
    #[arg(long)]
    dataset: Option<PathBuf>,
    #[arg(long, default_value = "target/rollouts")]
    store: PathBuf,
    #[arg(long, default_value_t = 20)]
    limit: usize,
    /// Skip problems with fewer initial roots (use 2 for interface supervision)
    #[arg(long, default_value_t = 1)]
    min_roots: usize,
    #[arg(long, default_value_t = 4)]
    levels: usize,
    #[arg(long, default_value_t = 8)]
    candidates: usize,
    #[arg(long, default_value_t = 1)]
    micro_max_size: usize,
    #[arg(long, default_value_t = 32)]
    merge_max_size: usize,
    #[arg(long, default_value_t = 200_000)]
    max_ground_instances: usize,
    #[arg(long, default_value_t = 500)]
    master_time_ms: u64,
    #[arg(long, default_value_t = 100)]
    local_time_ms: u64,
    #[arg(long, default_value_t = 64)]
    max_local_calls: usize,
    #[arg(long, default_value_t = 12)]
    max_boundary_atoms: usize,
    #[arg(long)]
    full_trace: bool,
    #[arg(long, default_value_t = 0)]
    conflict_core_checks: usize,
}
fn run() -> Result<()> {
    let options = Options::parse();
    if options.limit == 0 || (options.inputs.is_empty() && options.dataset.is_none()) {
        bail!("provide inputs or --dataset, and a positive --limit");
    }
    let store = DatasetStore::open(&options.store)?;
    let mut config = RolloutOptions {
        max_ground_instances: options.max_ground_instances,
        micro_max_size: options.micro_max_size,
        max_levels: options.levels,
        max_candidates: options.candidates,
        conflict_core_checks: options.conflict_core_checks,
        trace: if options.full_trace {
            TraceLevel::Full
        } else {
            TraceLevel::Summary
        },
        ..Default::default()
    };
    config.agglomeration.max_cluster_size = options.merge_max_size;
    config.master.max_time_ms = options.master_time_ms;
    config.master.local_budget.max_time_ms = options.local_time_ms;
    config.master.max_local_calls = options.max_local_calls;
    config.master.max_boundary_atoms = options.max_boundary_atoms;
    let mut programs = Vec::new();
    for path in &options.inputs {
        let value: serde_json::Value = serde_json::from_slice(&fs::read(path)?)?;
        let program: Program =
            serde_json::from_value(value.get("program").unwrap_or(&value).clone())
                .with_context(|| format!("reading {}", path.display()))?;
        if root_count(&program) >= options.min_roots {
            programs.push((Some(path.to_string_lossy().into_owned()), program));
        }
    }
    if let Some(root) = &options.dataset {
        let snapshot = SolutionSnapshot::open(root)?;
        let mut groups = BTreeMap::<String, VecDeque<_>>::new();
        // Real examples first within a group; groups are visited round-robin.
        let mut entries = snapshot.entries.clone();
        entries.sort_by(|a, b| b.optimal.cmp(&a.optimal).then_with(|| a.file.cmp(&b.file)));
        for entry in entries {
            groups
                .entry(entry.group_id.clone())
                .or_default()
                .push_back(entry);
        }
        while programs.len() < options.limit {
            let mut progress = false;
            for group in groups.values_mut() {
                if programs.len() == options.limit {
                    break;
                }
                if let Some(entry) = group.pop_front() {
                    let program = snapshot.program(&entry)?;
                    if root_count(&program) >= options.min_roots {
                        programs.push((Some(entry.file.clone()), program));
                    }
                    progress = true;
                }
            }
            if !progress {
                break;
            }
        }
    }
    let mut collected = 0;
    let mut failures = 0;
    for (source, program) in programs.into_iter().take(options.limit) {
        match collect_program(&store, &program, source, &config) {
            Ok(records) => {
                collected += records.len();
                println!(
                    "{}: {} rollout states",
                    program
                        .problem
                        .as_ref()
                        .map(|p| p.name.as_str())
                        .unwrap_or("domain"),
                    records.len()
                );
            }
            Err(error) => {
                failures += 1;
                eprintln!("{}: {error:#}", program.domain.name);
            }
        }
    }
    println!(
        "{collected} rollout states stored in {}; {failures} failed problems",
        options.store.display()
    );
    if collected == 0 {
        bail!("no rollout states collected");
    }
    Ok(())
}
fn root_count(program: &Program) -> usize {
    program
        .problem
        .as_ref()
        .and_then(|p| p.init_tn.as_ref())
        .map_or(0, |n| n.tn.subtasks.len())
}
fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("collect-rollouts: {error:#}");
            std::process::ExitCode::FAILURE
        }
    }
}
