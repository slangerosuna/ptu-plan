use super::*;
use crate::{
    learning::rollout::{RolloutOptions, collect_program},
    parsing::json::Program,
    solver::{GroundingMode, GroundingOptions},
    store::{DatasetStore, RecordKind, SolutionSnapshot},
    training::{
        DatasetWriter,
        backward::{self, BackwardOptions},
        sha256,
    },
};
use anyhow::bail;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, VecDeque},
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrepareOptions {
    pub dataset: PathBuf,
    pub run_dir: PathBuf,
    pub problems: usize,
    pub synthetic: usize,
    pub workers: usize,
    pub seed: u64,
    pub max_roots: usize,
    pub per_domain: usize,
    pub rollout: RolloutOptions,
    pub retry_failed: bool,
}
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PrepareReport {
    pub existing_solutions: usize,
    pub selected_existing: usize,
    pub synthetic_added: usize,
    pub synthetic_available: usize,
    pub generation_failures: Vec<String>,
    pub jobs: usize,
    pub successful_jobs: usize,
    pub failed_jobs: usize,
    pub reused_jobs: usize,
    pub rollout_records: usize,
    pub groups: usize,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
struct JobReport {
    id: String,
    error: Option<String>,
    records: usize,
}
struct Job {
    id: String,
    program: Program,
}
fn roots(p: &Program) -> usize {
    p.problem
        .as_ref()
        .and_then(|p| p.init_tn.as_ref())
        .map_or(0, |n| n.tn.subtasks.len())
}

pub fn prepare_data(options: &PrepareOptions) -> Result<PrepareReport> {
    if options.workers == 0
        || options.workers > 64
        || options.max_roots < 2
        || options.per_domain == 0
    {
        bail!("require workers 1..64, max roots >=2 and positive per-domain cap");
    }
    fs::create_dir_all(&options.run_dir)?;
    let _lock = super::lock(&options.run_dir.join("prepare.lock"))?;
    let snapshot = SolutionSnapshot::open(&options.dataset)?;
    let mut report = PrepareReport {
        existing_solutions: snapshot.entries.len(),
        ..Default::default()
    };
    let mut entries = snapshot.entries.clone();
    entries.sort_by(|a, b| b.optimal.cmp(&a.optimal).then_with(|| a.file.cmp(&b.file)));
    let mut groups = BTreeMap::<String, VecDeque<Program>>::new();
    let mut templates = BTreeMap::<String, Program>::new();
    for entry in &entries {
        let p = snapshot.program(entry)?;
        let size = p.problem.as_ref().map_or(usize::MAX, |p| p.objects.len());
        if templates
            .get(&entry.group_id)
            .is_none_or(|old| old.problem.as_ref().map_or(usize::MAX, |p| p.objects.len()) > size)
        {
            templates.insert(entry.group_id.clone(), p.clone());
        }
        if (2..=options.max_roots).contains(&roots(&p)) {
            groups
                .entry(entry.group_id.clone())
                .or_default()
                .push_back(p);
        }
    }
    let mut programs = Vec::new();
    let mut taken = BTreeMap::<String, usize>::new();
    while programs.len() < options.problems {
        let before = programs.len();
        for (group, queue) in &mut groups {
            if programs.len() == options.problems {
                break;
            }
            if *taken.get(group).unwrap_or(&0) >= options.per_domain {
                continue;
            }
            if let Some(program) = queue.pop_front() {
                programs.push(program);
                *taken.entry(group.clone()).or_default() += 1;
            }
        }
        if programs.len() == before {
            break;
        }
    }
    report.selected_existing = programs.len();
    let synthetic_dir = options.run_dir.join("solutions");
    let generation_key = sha256(&serde_json::to_vec(&(
        options.seed,
        options.synthetic,
        options.max_roots,
        options.rollout.max_ground_instances,
        templates.iter().collect::<Vec<_>>(),
    ))?);
    let generation_marker = options
        .run_dir
        .join(format!("generation-{generation_key}.json"));
    if options.synthetic > 0 && !generation_marker.exists() {
        let writer = DatasetWriter::new(&synthetic_dir)?;
        let template_count = templates.len().max(1);
        let next = AtomicUsize::new(0);
        let added = AtomicUsize::new(0);
        let failures = Mutex::new(Vec::new());
        let templates: Vec<_> = templates.iter().collect();
        std::thread::scope(|scope| {
            for _ in 0..options.workers.min(template_count) {
                scope.spawn(|| {
                    loop {
                        let index = next.fetch_add(1, Ordering::Relaxed);
                        let Some(&(group, template)) = templates.get(index) else {
                            break;
                        };
                        let count = options.synthetic / template_count
                            + usize::from(index < options.synthetic % template_count);
                        if count == 0 {
                            continue;
                        }
                        let result = backward::generate_to(
                            template,
                            &GroundingOptions {
                                max_instances: options.rollout.max_ground_instances,
                            },
                            &BackwardOptions {
                                count,
                                seed: options.seed.wrapping_add(index as u64),
                                grounding_mode: GroundingMode::InitialReachable,
                                min_roots: 2,
                                max_roots: options.max_roots,
                                max_depth: 6,
                                max_actions: options.max_roots.saturating_mul(8).max(24),
                                max_decisions: options.max_roots.saturating_mul(32).max(96),
                                max_attempts: count.saturating_mul(32).max(128),
                                randomize_irrelevant: true,
                            },
                            |sample| {
                                added.fetch_add(
                                    usize::from(writer.write(sample)?),
                                    Ordering::Relaxed,
                                );
                                Ok(())
                            },
                        );
                        match result {
                            Ok(generated) if generated.samples.len() < count => {
                                failures.lock().unwrap().push(format!(
                                    "{group}: {}/{} samples; {}",
                                    generated.samples.len(),
                                    count,
                                    generated.last_rejection
                                ))
                            }
                            Err(error) => {
                                failures.lock().unwrap().push(format!("{group}: {error:#}"))
                            }
                            _ => {}
                        }
                        eprintln!(
                            "backward domain {}/{} finished: {} new samples",
                            index + 1,
                            template_count,
                            added.load(Ordering::Relaxed)
                        );
                    }
                });
            }
        });
        report.synthetic_added = added.load(Ordering::Relaxed);
        report.generation_failures = failures.into_inner().unwrap();
        report.generation_failures.sort();
        writer.manifest()?;
        save(&generation_marker, &report.generation_failures)?;
    }
    if synthetic_dir.join("manifest.json").exists() {
        let generated = SolutionSnapshot::open(&synthetic_dir)?;
        report.synthetic_available = generated.entries.len();
        for entry in &generated.entries {
            let p = generated.program(entry)?;
            if (2..=options.max_roots).contains(&roots(&p)) {
                programs.push(p);
            }
        }
    }
    // Job identity excludes timings and output paths. Finished jobs are reused;
    // incomplete shards are quarantined rather than silently counted twice.
    let mut jobs = BTreeMap::new();
    for program in programs {
        let id = sha256(&serde_json::to_vec(&(&program, &options.rollout))?);
        jobs.entry(id.clone()).or_insert(Job { id, program });
    }
    report.jobs = jobs.len();
    let total_jobs = report.jobs;
    if report.jobs == 0 {
        bail!("no multi-root problems selected or generated");
    }
    let queue = Arc::new(Mutex::new(jobs.into_values().collect::<VecDeque<_>>()));
    let results = Arc::new(Mutex::new(Vec::<(JobReport, bool)>::new()));
    let done = AtomicUsize::new(0);
    let shards = options.run_dir.join("shards");
    fs::create_dir_all(&shards)?;
    std::thread::scope(|scope| -> Result<()> {
        let mut handles = Vec::new();
        for _ in 0..options.workers {
            let queue = queue.clone();
            let results = results.clone();
            let done = &done;
            let shards = &shards;
            handles.push(scope.spawn(move || -> Result<()> {
                loop {
                    let Some(job) = queue.lock().unwrap().pop_front() else {
                        break;
                    };
                    let directory = shards.join(&job.id);
                    let marker = directory.join("job.json");
                    let previous = if marker.exists() {
                        Some(read::<JobReport>(&marker)?)
                    } else {
                        None
                    };
                    let reused = previous
                        .as_ref()
                        .is_some_and(|r| r.error.is_none() || !options.retry_failed);
                    let result = if reused {
                        previous.unwrap()
                    } else {
                        if directory.exists() {
                            let suffix = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)?
                                .as_nanos();
                            fs::rename(
                                &directory,
                                shards.join(format!("{}.incomplete-{suffix}", job.id)),
                            )?;
                        }
                        let store = DatasetStore::open(&directory)?;
                        let outcome = collect_program(
                            &store,
                            &job.program,
                            Some(format!("prepared-job:{}", job.id)),
                            &options.rollout,
                        );
                        let records = store
                            .snapshot()?
                            .entries
                            .iter()
                            .filter(|e| e.metadata.kind == RecordKind::Rollout)
                            .count();
                        let result = JobReport {
                            id: job.id,
                            error: outcome.err().map(|e| format!("{e:#}")),
                            records,
                        };
                        save(&marker, &result)?;
                        result
                    };
                    eprintln!(
                        "rollout {}/{}: {} records{}{}",
                        done.fetch_add(1, Ordering::Relaxed) + 1,
                        total_jobs,
                        result.records,
                        if reused { " reused" } else { "" },
                        if result.error.is_some() {
                            " (partial/failure)"
                        } else {
                            ""
                        }
                    );
                    results.lock().unwrap().push((result, reused));
                }
                Ok(())
            }));
        }
        for handle in handles {
            handle
                .join()
                .map_err(|_| anyhow::anyhow!("rollout worker panicked"))??;
        }
        Ok(())
    })?;
    let store = DatasetStore::open(&options.run_dir.join("data"))?;
    let mut groups = std::collections::BTreeSet::new();
    for (job, reused) in results.lock().unwrap().iter() {
        report.reused_jobs += usize::from(*reused);
        report.successful_jobs += usize::from(job.error.is_none());
        report.failed_jobs += usize::from(job.error.is_some());
        let shard = DatasetStore::open(&shards.join(&job.id))?;
        for entry in &shard.snapshot()?.entries {
            let value: serde_json::Value = shard.load(entry)?;
            let copied = store.put(entry.metadata.clone(), &value)?;
            if copied.id != entry.id {
                bail!("immutable shard import changed content identity");
            }
        }
    }
    for entry in &store.snapshot()?.entries {
        if entry.metadata.kind == RecordKind::Rollout {
            report.rollout_records += 1;
            if let Some(group) = &entry.metadata.group_id {
                groups.insert(group.clone());
            }
        }
    }
    report.groups = groups.len();
    save(&options.run_dir.join("preparation.json"), &report)?;
    if report.rollout_records == 0 {
        bail!("preparation produced no usable rollout records");
    }
    Ok(report)
}
