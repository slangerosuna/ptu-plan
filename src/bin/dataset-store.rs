use anyhow::Result;
use clap::{Parser, Subcommand};
use ptu_plan::learning::{rollout::RolloutAuditor, training::Checkpoint};
use ptu_plan::store::{DatasetStore, RecordKind};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

#[derive(Parser)]
#[command(about = "Inspect, verify, or recover an immutable experiment dataset store")]
struct Options {
    #[arg(long, default_value = "target/rollouts")]
    store: PathBuf,
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    Stats,
    Audit,
    Repair,
}
fn run() -> Result<()> {
    let options = Options::parse();
    let store = DatasetStore::open(&options.store)?;
    match options.command {
        Command::Stats => {
            let snapshot = store.snapshot()?;
            let mut counts = BTreeMap::new();
            let mut groups = BTreeSet::new();
            let mut bytes = 0;
            for entry in &snapshot.entries {
                *counts
                    .entry(format!("{:?}", entry.metadata.kind))
                    .or_insert(0) += 1;
                bytes += entry.bytes;
                if let Some(group) = &entry.metadata.group_id {
                    groups.insert(group);
                }
            }
            println!(
                "{}",
                serde_json::to_string_pretty(
                    &serde_json::json!({"snapshot_id":snapshot.fingerprint()?,"records":counts,"groups":groups.len(),"bytes":bytes})
                )?
            );
        }
        Command::Repair => {
            let snapshot = store.repair()?;
            println!("{} records recovered and verified", snapshot.entries.len());
        }
        Command::Audit => {
            let snapshot = store.snapshot()?;
            let mut auditor = RolloutAuditor::new(&store, &snapshot);
            for entry in &snapshot.entries {
                let _: serde_json::Value = store.load(entry)?;
                match entry.metadata.kind {
                    RecordKind::Rollout => {
                        auditor.audit(entry)?;
                    }
                    RecordKind::Model => store.load::<Checkpoint>(entry)?.validate()?,
                    RecordKind::InterfaceModel => store
                        .load::<ptu_plan::learning::interface::InterfaceCheckpoint>(entry)?
                        .validate()?,
                    RecordKind::LatentModel => store
                        .load::<ptu_plan::learning::latent_training::LatentCheckpoint>(entry)?
                        .validate()?,
                    _ => {}
                }
            }
            println!("{} experiment records audited", snapshot.entries.len());
            println!(
                "{} source preparations, {} core verifications, {} cached core proofs",
                auditor.statistics.prepared_problems,
                auditor.statistics.core_verifications,
                auditor.statistics.core_cache_hits
            );
        }
    }
    Ok(())
}
fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("dataset-store: {error:#}");
            std::process::ExitCode::FAILURE
        }
    }
}
