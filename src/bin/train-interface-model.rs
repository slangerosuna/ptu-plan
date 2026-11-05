use anyhow::Result;
use clap::Parser;
use ptu_plan::learning::interface::{
    InterfaceCheckpoint, InterfaceTrainOptions, train_interface_store,
};
use ptu_plan::store::DatasetStore;
use std::path::PathBuf;

#[derive(Parser)]
#[command(about = "Train a grounded fluent true/false/unknown interface proposal model")]
struct Options {
    #[arg(long, default_value = "target/rollouts")]
    store: PathBuf,
    #[arg(long, default_value = "target/interface-model.json")]
    output: PathBuf,
    #[arg(long)]
    resume: Option<PathBuf>,
    #[arg(long, default_value_t = 50)]
    epochs: usize,
    #[arg(long, default_value_t = 8)]
    hidden: usize,
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
    #[arg(long, default_value_t = 0.25)]
    unknown_weight: f64,
    /// Train only complete-plan boundaries, excluding locally feasible prefixes
    #[arg(long)]
    global_only: bool,
}
fn run() -> Result<()> {
    let options = Options::parse();
    let store = DatasetStore::open(&options.store)?;
    let resume = options
        .resume
        .map(|p| -> Result<_> { Ok(serde_json::from_slice(&std::fs::read(p)?)?) })
        .transpose()?;
    let config = InterfaceTrainOptions {
        hidden: options.hidden,
        epochs: options.epochs,
        batch_size: options.batch_size,
        learning_rate: options.learning_rate,
        seed: options.seed,
        split_seed: options.split_seed,
        max_records: options.max_records,
        class_weights: [options.unknown_weight, 1.0, 1.0],
        include_local: !options.global_only,
    };
    let (entry, _, report) = train_interface_store(&store, &config, resume)?;
    let checkpoint: InterfaceCheckpoint = store.load(&entry)?;
    checkpoint.save(&options.output)?;
    println!("model {} -> {}", entry.id, options.output.display());
    println!(
        "loss {:.5} -> {:.5}; atom accuracy train/validation/test {:?}/{:?}/{:?}; interfaces {}/{}/{}",
        report.initial_training.loss,
        report.training.loss,
        report.training.accuracy,
        report.validation.accuracy,
        report.test.accuracy,
        report.training.interfaces,
        report.validation.interfaces,
        report.test.interfaces
    );
    Ok(())
}
fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("train-interface-model: {e:#}");
            std::process::ExitCode::FAILURE
        }
    }
}
