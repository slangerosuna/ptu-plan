use anyhow::Result;
use clap::Parser;
use ptu_plan::{
    learning::{
        latent_training::{LatentCheckpoint, train_latent_store},
        model::MergeModel,
        training::{Checkpoint, TrainOptions},
    },
    store::DatasetStore,
};
use std::{fs, path::PathBuf};
#[derive(Parser)]
#[command(
    about = "Train frozen-encoder latent merge transitions from independently audited symbolic rollouts"
)]
struct Options {
    #[arg(long, default_value = "target/rollouts")]
    store: PathBuf,
    #[arg(long)]
    merge_model: PathBuf,
    #[arg(long, default_value = "target/latent-policy.json")]
    output: PathBuf,
    #[arg(long)]
    resume: Option<PathBuf>,
    #[arg(long, default_value_t = 50)]
    epochs: usize,
    #[arg(long, default_value_t = 8)]
    batch_size: usize,
    #[arg(long, default_value_t = 0.003)]
    learning_rate: f64,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value_t = 42)]
    split_seed: u64,
    #[arg(long, default_value_t = 10_000)]
    max_records: usize,
}
fn run() -> Result<()> {
    let args = Options::parse();
    let value: serde_json::Value = serde_json::from_slice(&fs::read(&args.merge_model)?)?;
    let encoder: MergeModel = if value.get("model").is_some() {
        let checkpoint: Checkpoint = serde_json::from_value(value)?;
        checkpoint.validate()?;
        if checkpoint.options.split_seed != args.split_seed {
            anyhow::bail!(
                "latent split seed must match the pretrained encoder to avoid held-out domain leakage"
            );
        }
        checkpoint.model
    } else {
        serde_json::from_value(value)?
    };
    encoder.validate()?;
    let resume = args
        .resume
        .map(|p| -> Result<LatentCheckpoint> { Ok(serde_json::from_slice(&fs::read(p)?)?) })
        .transpose()?;
    let options = TrainOptions {
        model: encoder.config,
        epochs: args.epochs,
        batch_size: args.batch_size,
        learning_rate: args.learning_rate,
        seed: args.seed,
        split_seed: args.split_seed,
        max_records: args.max_records,
        ..Default::default()
    };
    let store = DatasetStore::open(&args.store)?;
    let (entry, _, report) = train_latent_store(&store, &encoder, &options, resume)?;
    store.load::<LatentCheckpoint>(&entry)?.save(&args.output)?;
    println!("latent model {} -> {}", entry.id, args.output.display());
    println!("{}", serde_json::to_string(&report)?);
    Ok(())
}
fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("train-latent-model: {e:#}");
            std::process::ExitCode::FAILURE
        }
    }
}
