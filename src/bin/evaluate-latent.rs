use anyhow::Result;
use clap::Parser;
use ptu_plan::{
    learning::{
        latent_evaluation::{DriftOptions, evaluate_latent_drift},
        latent_training::LatentCheckpoint,
    },
    store::DatasetStore,
};
use std::{fs, path::PathBuf};
#[derive(Parser)]
#[command(
    about = "Measure bounded multi-step latent drift against true encodings and a no-dynamics baseline on audited source rollouts"
)]
struct Options {
    #[arg(long, default_value = "target/rollouts")]
    store: PathBuf,
    #[arg(long)]
    latent_model: PathBuf,
    #[arg(long, default_value = "target/latent-drift.json")]
    output: PathBuf,
    #[arg(long, default_value_t = 4)]
    depth: usize,
    #[arg(long, default_value_t = 1000)]
    max_records: usize,
    #[arg(long, default_value_t = 8)]
    paths_per_record: usize,
    #[arg(long, default_value_t = 1024)]
    max_paths: usize,
}
fn run() -> Result<()> {
    let args = Options::parse();
    let checkpoint: LatentCheckpoint = serde_json::from_slice(&fs::read(&args.latent_model)?)?;
    let options = DriftOptions {
        depth: args.depth,
        max_records: args.max_records,
        paths_per_record: args.paths_per_record,
        max_paths: args.max_paths,
    };
    let store = DatasetStore::open(&args.store)?;
    let report = evaluate_latent_drift(&store, &checkpoint, &options)?;
    report.save(&args.output)?;
    println!(
        "{} paths from {} audited records -> {}",
        report.total_paths,
        report.audited_records,
        args.output.display()
    );
    for split in &report.splits {
        println!("{}", serde_json::to_string(split)?);
    }
    Ok(())
}
fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("evaluate-latent: {e:#}");
            std::process::ExitCode::FAILURE
        }
    }
}
