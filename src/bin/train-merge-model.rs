use anyhow::Result;
use clap::Parser;
use ptu_plan::learning::model::ModelConfig;
use ptu_plan::learning::training::{Checkpoint, TrainOptions, train_store};
use ptu_plan::store::DatasetStore;
use std::fs;
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    about = "Train a CPU relational merge policy and log-search-effort value head with domain-group splits"
)]
struct Options {
    #[arg(long, default_value = "target/rollouts")]
    store: PathBuf,
    #[arg(long, default_value = "target/merge-model.json")]
    output: PathBuf,
    #[arg(long)]
    resume: Option<PathBuf>,
    #[arg(long, default_value_t = 50)]
    epochs: usize,
    #[arg(long, default_value_t = 8)]
    batch_size: usize,
    #[arg(long, default_value_t = 8)]
    hidden: usize,
    #[arg(long, default_value_t = 2)]
    layers: usize,
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
    let options = Options::parse();
    let store = DatasetStore::open(&options.store)?;
    let resume: Option<Checkpoint> = options
        .resume
        .as_ref()
        .map(|path| -> Result<_> { Ok(serde_json::from_slice(&fs::read(path)?)?) })
        .transpose()?;
    let config = TrainOptions {
        model: ModelConfig {
            hidden: options.hidden,
            layers: options.layers,
        },
        epochs: options.epochs,
        batch_size: options.batch_size,
        learning_rate: options.learning_rate,
        seed: options.seed,
        split_seed: options.split_seed,
        max_records: options.max_records,
        ..Default::default()
    };
    let (entry, _, report) = train_store(&store, &config, resume)?;
    let checkpoint: Checkpoint = store.load(&entry)?;
    if let Some(parent) = options
        .output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    checkpoint.save(&options.output)?;
    println!("model {} -> {}", entry.id, options.output.display());
    println!(
        "train loss {:.5} -> {:.5}; pair accuracy {:?}; records train/validation/test {}/{}/{}",
        report.initial_training.mean_loss,
        report.training.mean_loss,
        report.training.pair_accuracy,
        report.training.records,
        report.validation.records,
        report.test.records
    );
    Ok(())
}
fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("train-merge-model: {error:#}");
            std::process::ExitCode::FAILURE
        }
    }
}
