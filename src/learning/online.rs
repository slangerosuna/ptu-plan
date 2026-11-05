//! Immutable, replay-certified training problems for concurrent RL collection.
use super::{rollout::ProblemRecord, training::Checkpoint};
use crate::{
    parsing::json::Program,
    solver::{GroundingMode, GroundingOptions, PlanCertificate},
    store::{DatasetStore, RecordKind, Split, atomic_write, group_split},
    training::{
        DatasetWriter,
        backward::{BackwardOptions, generate_to},
        sha256,
    },
};
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::{fs, path::PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertifiedProblem {
    pub schema_version: u32,
    pub group: String,
    pub program: Program,
    pub certificate: PlanCertificate,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnlineInput {
    pub id: String,
    /// Preserve the exact content-addressed bytes across the Python boundary.
    pub json: String,
}

impl OnlineInput {
    pub fn decode(&self, parent: &Checkpoint) -> Result<CertifiedProblem> {
        if sha256(self.json.as_bytes()) != self.id {
            bail!("online problem checksum mismatch");
        }
        let record: CertifiedProblem = serde_json::from_str(&self.json)?;
        if record.schema_version != 1
            || record
                .program
                .problem
                .as_ref()
                .and_then(|p| p.init_tn.as_ref())
                .is_none_or(|tn| tn.tn.subtasks.len() < 2)
            || record.group != sha256(&serde_json::to_vec(&record.program.domain)?)
            || !parent.training_groups.contains(&record.group)
            || group_split(&record.group, parent.options.split_seed) != Split::Train
        {
            bail!("online problems must belong to the parent's Train domains");
        }
        Ok(record)
    }
}

#[derive(Debug, Clone, clap::Args, Serialize)]
pub struct GenerationOptions {
    #[arg(long)]
    pub store: PathBuf,
    #[arg(long)]
    pub checkpoint: PathBuf,
    #[arg(long)]
    pub output: PathBuf,
    #[arg(long)]
    pub template_id: String,
    #[arg(long, default_value_t = 64)]
    pub count: usize,
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
    #[arg(long, default_value_t = 8)]
    pub max_roots: usize,
    #[arg(long, default_value_t = 8)]
    pub max_depth: usize,
    #[arg(long, default_value_t = 200_000)]
    pub max_ground_instances: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SourceRef {
    pub id: String,
    pub group: String,
    pub file: String,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct Batch {
    pub schema_version: u32,
    pub job: String,
    pub snapshot_id: String,
    pub parent_sha256: String,
    pub template_id: String,
    pub requested: usize,
    pub attempts: usize,
    pub last_rejection: String,
    pub samples: Vec<SourceRef>,
}

pub fn generate(options: &GenerationOptions) -> Result<()> {
    if options.count == 0
        || options.count > 1024
        || !(2..=32).contains(&options.max_roots)
        || options.max_depth == 0
        || options.max_ground_instances == 0
    {
        bail!("require count 1..1024, roots 2..32 and positive depth/grounding limits");
    }
    let parent_bytes = fs::read(&options.checkpoint)?;
    let parent: Checkpoint = serde_json::from_slice(&parent_bytes)?;
    parent.validate()?;
    let store = DatasetStore::open(&options.store)?;
    let snapshot = store.snapshot()?;
    if snapshot.fingerprint()? != parent.snapshot_id {
        bail!("generation source snapshot differs from the parent checkpoint");
    }
    let entry = snapshot
        .entries
        .iter()
        .find(|e| e.id == options.template_id && e.metadata.kind == RecordKind::Problem)
        .context("template problem not found")?;
    let template: ProblemRecord = store.load(entry)?;
    let group = sha256(&serde_json::to_vec(&template.program.domain)?);
    if entry.metadata.group_id.as_ref() != Some(&group)
        || !parent.training_groups.contains(&group)
        || group_split(&group, parent.options.split_seed) != Split::Train
    {
        bail!("generation is restricted to the parent's Train domains");
    }
    let parent_sha256 = sha256(&parent_bytes);
    let job = sha256(&serde_json::to_vec(&(
        "online-backward-v1",
        &parent_sha256,
        &options.template_id,
        options.count,
        options.seed,
        options.max_roots,
        options.max_depth,
        options.max_ground_instances,
    ))?);
    let ready = options.output.join("ready").join(format!("{job}.json"));
    let work = options.output.join("jobs").join(&job);
    fs::create_dir_all(&work)?;
    fs::create_dir_all(options.output.join("ready"))?;
    let lock = fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(work.join("job.lock"))?;
    lock.try_lock().context("another collector owns this job")?;
    if ready.exists() {
        let batch: Batch = serde_json::from_slice(&fs::read(&ready)?)?;
        if batch.schema_version != 1
            || batch.job != job
            || batch.parent_sha256 != parent_sha256
            || batch.snapshot_id != parent.snapshot_id
        {
            bail!("published online batch provenance mismatch");
        }
        for sample in batch.samples {
            if sample.id.len() != 64
                || !sample.id.bytes().all(|b| b.is_ascii_hexdigit())
                || sample.file != format!("objects/{}/{}.json", &sample.id[..2], sample.id)
                || sample.group != group
            {
                bail!("invalid published sample reference");
            }
            if sha256(&fs::read(options.output.join(sample.file))?) != sample.id {
                bail!("published online sample checksum mismatch");
            }
        }
        println!("{}", ready.display());
        return Ok(());
    }
    let solutions = work.join("solutions");
    fs::create_dir_all(&solutions)?;
    // The job lock proves no live writer owns these uncommitted temporaries.
    // A killed worker may leave them behind, including after PID reuse.
    for entry in fs::read_dir(&solutions)? {
        let path = entry?.path();
        if path
            .file_name()
            .is_some_and(|n| n.to_string_lossy().starts_with('.'))
            && path.extension().is_some_and(|e| e == "tmp")
        {
            fs::remove_file(path)?;
        }
    }
    let writer = DatasetWriter::new(&solutions)?;
    let mut samples = Vec::new();
    let report = generate_to(
        &template.program,
        &GroundingOptions {
            max_instances: options.max_ground_instances,
        },
        &BackwardOptions {
            grounding_mode: GroundingMode::InitialReachable,
            count: options.count,
            seed: options.seed,
            max_roots: options.max_roots,
            min_roots: 2,
            max_depth: options.max_depth,
            max_actions: options.max_roots.saturating_mul(8),
            max_decisions: options.max_roots.saturating_mul(32),
            max_attempts: options.count.saturating_mul(48).max(128),
            randomize_irrelevant: true,
        },
        |sample| {
            // make_sample has independently replayed this certificate. Preserve the
            // rich witness for future supervised collection as well as the RL input.
            writer.write(sample)?;
            let mut program = sample.program.clone();
            program
                .problem
                .as_mut()
                .context("generated problem missing")?
                .name = "online".into();
            let record = CertifiedProblem {
                schema_version: 1,
                group: group.clone(),
                program,
                certificate: sample.certificate.clone(),
            };
            let bytes = serde_json::to_vec(&record)?;
            let id = sha256(&bytes);
            let file = format!("objects/{}/{id}.json", &id[..2]);
            let path = options.output.join(&file);
            fs::create_dir_all(path.parent().unwrap())?;
            if path.exists() {
                if fs::read(&path)? != bytes {
                    bail!("online content identity collision");
                }
            } else {
                atomic_write(&path, &bytes)?;
            }
            samples.push(SourceRef {
                id,
                group: group.clone(),
                file,
            });
            Ok(())
        },
    )?;
    writer.manifest()?;
    let batch = Batch {
        schema_version: 1,
        job,
        snapshot_id: parent.snapshot_id,
        parent_sha256,
        template_id: options.template_id.clone(),
        requested: options.count,
        attempts: report.attempts,
        last_rejection: report.last_rejection,
        samples,
    };
    // Publishing this marker is the transaction boundary seen by the trainer.
    atomic_write(&ready, &serde_json::to_vec_pretty(&batch)?)?;
    println!("{}", ready.display());
    Ok(())
}
