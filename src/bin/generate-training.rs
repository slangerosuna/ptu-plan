use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::Parser;
use serde::Serialize;

use ptu_plan::parsing::discovery::discover_sibling_problems;
use ptu_plan::parsing::parser::{parse_program, read_program_json, resolve_parser_config};
use ptu_plan::solver::{GroundingMode, GroundingOptions, SearchBudget};
use ptu_plan::training::{
    DatasetWriter,
    backward::{self, BackwardOptions},
    panda::{self, PandaOptions},
};

#[derive(Debug, Parser, Serialize)]
#[command(about = "Mix replay-validated backward samples and cost-optimal PANDA IPC plans")]
struct Options {
    #[arg(long, default_value = "../train")]
    output: PathBuf,
    #[arg(long, default_value = "../hddl/test_data/ipc")]
    ipc_dir: PathBuf,
    /// Repeat to select benchmark domains; defaults to three small IPC families
    #[arg(long = "domain")]
    domains: Vec<String>,
    /// Explicit domain/problem pair, including nonstandard benchmark filenames
    #[arg(long)]
    domain_file: Option<PathBuf>,
    #[arg(long, requires = "domain_file")]
    problem: Option<PathBuf>,
    /// Whole-program JSON template(s); otherwise use each selected domain's first problem
    #[arg(long)]
    template: Vec<PathBuf>,
    #[arg(long, default_value_t = 128)]
    synthetic_count: usize,
    /// Sample the whole grounded task pool (default: initial-reachable tasks)
    #[arg(long)]
    backward_full_grounding: bool,
    /// Maximum accepted real records; unsupported/time-limited cases are skipped
    #[arg(long, default_value_t = 12)]
    real_limit: usize,
    #[arg(long, default_value_t = 6)]
    per_domain: usize,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value_t = 8)]
    max_depth: usize,
    #[arg(long, default_value_t = 32)]
    max_actions: usize,
    #[arg(long, default_value_t = 20_000)]
    max_attempts: usize,
    #[arg(long, default_value_t = 200_000)]
    max_ground_instances: usize,
    #[arg(long, default_value = "target/panda")]
    panda_root: PathBuf,
    #[arg(long, default_value = "target/panda-training")]
    work: PathBuf,
    #[arg(long, default_value_t = 15)]
    search_seconds: u64,
    #[arg(long, default_value_t = 30)]
    stage_seconds: u64,
    #[arg(long, default_value_t = 1_073_741_824)]
    memory_bytes: u64,
    #[arg(long)]
    parser: Option<PathBuf>,
    /// Re-ground and replay every existing record without collecting new data
    #[arg(long)]
    audit_only: bool,
}

#[derive(Serialize)]
struct Report<'a> {
    options: &'a Options,
    synthetic_generated: usize,
    real_accepted: usize,
    files_added: usize,
    failures: Vec<(String, String)>,
    audited_records: usize,
    counts_by_source: BTreeMap<String, usize>,
}

fn run(options: &Options) -> Result<()> {
    let writer = DatasetWriter::new(&options.output)?;
    let grounding = GroundingOptions {
        max_instances: options.max_ground_instances,
    };
    if options.audit_only {
        println!(
            "Audited {} records in {}",
            writer.audit(&grounding)?,
            options.output.display()
        );
        return Ok(());
    }
    let external =
        options.real_limit > 0 || (options.template.is_empty() && options.synthetic_count > 0);
    let parser = if external {
        Some(resolve_parser_config(options.parser.as_deref()).map_err(anyhow::Error::msg)?)
    } else {
        None
    };
    let domains = if options.domain_file.is_some() {
        vec!["explicit".into()]
    } else if options.domains.is_empty() {
        vec![
            "Blocksworld-GTOHP".into(),
            "AssemblyHierarchical".into(),
            "Towers".into(),
        ]
    } else {
        options.domains.clone()
    };
    let mut cases = Vec::new();
    let mut templates = Vec::new();
    let mut failures = Vec::new();
    for domain in &domains {
        if !external {
            break;
        }
        let domain_path = options
            .domain_file
            .clone()
            .unwrap_or_else(|| options.ipc_dir.join(domain).join("domain.hddl"));
        let problems = match &options.problem {
            Some(problem) => vec![problem.clone()],
            None => discover_sibling_problems(&domain_path)
                .with_context(|| format!("discovering {domain}"))?,
        };
        if options.template.is_empty()
            && options.synthetic_count > 0
            && let Some(problem) = problems.first()
        {
            match parse_program(
                parser.as_ref().context("missing parser configuration")?,
                &domain_path,
                Some(problem),
            ) {
                Ok(program) => templates.push(program),
                Err(error) => failures.push((format!("template:{domain}"), error)),
            }
        }
        cases.push(
            problems
                .into_iter()
                .take(options.per_domain)
                .map(|problem| (domain_path.clone(), problem))
                .collect::<Vec<_>>(),
        );
    }
    for template in &options.template {
        templates.push(read_program_json(template).map_err(anyhow::Error::msg)?);
    }
    let mut report = Report {
        options,
        synthetic_generated: 0,
        real_accepted: 0,
        files_added: 0,
        failures,
        audited_records: 0,
        counts_by_source: BTreeMap::new(),
    };
    // A failed family does not discard successful records from earlier families.
    for (index, template) in templates.iter().enumerate() {
        let remaining = options.synthetic_count - report.synthetic_generated;
        if remaining == 0 {
            break;
        }
        let count = remaining.div_ceil(templates.len() - index);
        eprintln!("Backward: {} samples from {}", count, template.domain.name);
        let generated = backward::generate_to(
            template,
            &grounding,
            &BackwardOptions {
                grounding_mode: if options.backward_full_grounding {
                    GroundingMode::Full
                } else {
                    GroundingMode::InitialReachable
                },
                count,
                seed: options.seed.wrapping_add(index as u64),
                max_depth: options.max_depth,
                max_actions: options.max_actions,
                max_attempts: options.max_attempts,
                ..BackwardOptions::default()
            },
            |sample| {
                report.files_added += usize::from(writer.write(sample)?);
                report.synthetic_generated += 1;
                writer.manifest()?;
                Ok(())
            },
        );
        match generated {
            Ok(generated) => {
                if generated.samples.len() != count {
                    report.failures.push((
                        format!("backward:{}", template.domain.name),
                        format!(
                            "retained {}/{count} samples after {} attempts; last rejection: {}",
                            generated.samples.len(),
                            generated.attempts,
                            generated.last_rejection
                        ),
                    ));
                }
            }
            Err(error) => {
                eprintln!("Skipping synthetic family: {error:#}");
                report.failures.push((
                    format!("backward:{}", template.domain.name),
                    format!("{error:#}"),
                ));
            }
        }
        writer.manifest()?;
    }
    let panda_options = PandaOptions {
        root: options.panda_root.clone(),
        work: options.work.clone(),
        stage_seconds: options.stage_seconds,
        search_seconds: options.search_seconds,
        memory_bytes: options.memory_bytes,
        grounding,
        validation_budget: SearchBudget {
            max_time_ms: 10_000,
            max_expansions: 200_000,
            max_generated: 500_000,
            ..SearchBudget::default()
        },
    };
    // Round-robin prevents the first domain from consuming the whole real quota.
    for index in 0..options.per_domain {
        for family in &cases {
            if report.real_accepted >= options.real_limit {
                break;
            }
            let Some((domain, problem)) = family.get(index) else {
                continue;
            };
            eprintln!("PANDA optimal: {}", problem.display());
            match panda::collect(
                domain,
                problem,
                parser.as_ref().context("missing parser configuration")?,
                &panda_options,
            ) {
                Ok(sample) => {
                    eprintln!("  accepted: {} actions", sample.action_cost);
                    report.files_added += usize::from(writer.write(&sample)?);
                    report.real_accepted += 1;
                }
                Err(error) => {
                    eprintln!("  skipped: {error:#}");
                    report
                        .failures
                        .push((problem.display().to_string(), format!("{error:#}")));
                }
            }
            writer.manifest()?;
        }
        if report.real_accepted >= options.real_limit {
            break;
        }
    }
    report.audited_records = writer.audit(&grounding)?;
    for entry in writer.manifest()? {
        *report.counts_by_source.entry(entry.source).or_default() += 1;
    }
    fs::write(
        options.output.join("collection-report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    fs::write(
        options.output.join("README.md"),
        "# HTN training data\n\nEach backward-*.json or panda-optimal-*.json is a self-contained training sample with its source program, replayable HTN certificate, state trace, regressed boundaries, graph, and micro-partitions. manifest.json lists SHA-256 checksums, sources, domain group IDs, costs, and optimality labels. collection-report.json records settings and rejected cases.\n\nBackward plans provide feasible upper bounds. PANDA records use uniform-cost action search, original-HDDL plan conversion and verification, plus local exact replay. Only unit-cost Boolean instances are imported. Boundary suffix costs refer to the recorded residual HTN and state, not every state satisfying a partial boundary.\n\nKeep all records with the same group_id in one train/validation/test split to avoid leaking a domain or its synthetic siblings. Do not treat sampled suffix lengths as optimal values.\n\nRe-audit with: cargo run --bin generate-training -- --output ../train --audit-only\n",
    )?;
    println!(
        "{} records audited in {}: {:?}; {} files added; {} rejected cases",
        report.audited_records,
        options.output.display(),
        report.counts_by_source,
        report.files_added,
        report.failures.len()
    );
    if report.synthetic_generated != options.synthetic_count {
        bail!("synthetic quota not reached; accepted records were retained");
    }
    if options.real_limit > 0 && report.real_accepted == 0 {
        bail!("no optimal real solutions were accepted; see collection-report.json");
    }
    Ok(())
}

fn main() -> Result<()> {
    run(&Options::parse())
}
