//! Prepare audited training packages locally and fit every learned component.
use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use ptu_plan::{
    instrumentation::TraceLevel,
    learning::{
        pipeline::{ExportOptions, PrepareOptions, export_data, finalize_bundle, prepare_data},
        rollout::RolloutOptions,
    },
};
use std::{path::PathBuf, process::Command};

#[derive(Parser)]
#[command(
    about = "Prepare data and train merge/value, interface, and frozen-encoder latent models"
)]
struct Cli {
    #[command(subcommand)]
    command: Action,
}
#[derive(Subcommand)]
enum Action {
    /// Generate backward examples, collect oracle labels, audit, and export.
    Prepare(Preparation),
    /// On-policy planner-return fine-tuning of the complete merge/value encoder.
    Rl(RlTraining),
    /// Publish one immutable batch of certified Train-only backward problems.
    GenerateOnline(ptu_plan::learning::online::GenerationOptions),
    /// Persistent JSONL planner environment used by the RL trainer.
    RlEnvironment(ptu_plan::learning::rl::EnvironmentOptions),
    /// Validate an RL checkpoint against its parent and Torch predictions.
    RlCheck {
        #[arg(long)]
        parent: PathBuf,
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long)]
        parity: PathBuf,
    },
    /// Audit and export an existing run store without generating more data.
    Export {
        #[arg(long, default_value = "runs/hilg-wl")]
        run_dir: PathBuf,
        #[arg(long, default_value_t = 32)]
        hidden: usize,
        #[arg(long)]
        legacy_encoder: bool,
        #[arg(long, default_value_t = 3)]
        layers: usize,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long, default_value_t = 42)]
        split_seed: u64,
    },
    /// Train all learned components from an existing portable prepared package.
    Fit(Fit),
    /// Prepare data then train all learned components in one job.
    All {
        #[command(flatten)]
        preparation: Preparation,
        #[command(flatten)]
        training: Training,
    },
    /// Validate and publish the native model bundle (called by the GPU trainer).
    Finalize {
        #[arg(long)]
        run_dir: PathBuf,
        #[arg(long)]
        model_dir: PathBuf,
    },
}
#[derive(Args)]
struct RlTraining {
    #[arg(long, default_value = "runs/hilg-wl")]
    run_dir: PathBuf,
    /// Import newly committed certified problems between RL epochs.
    #[arg(long)]
    online_dir: Option<PathBuf>,
    #[arg(long, default_value_t = 0.75)]
    online_fraction: f64,
    #[arg(long)]
    parent_models: Option<PathBuf>,
    #[arg(long)]
    model_dir: Option<PathBuf>,
    #[arg(long, default_value = "python3")]
    python: PathBuf,
    #[arg(long,default_value="auto",value_parser=["auto","cpu","cuda"])]
    device: String,
    #[arg(long, default_value_t = 20)]
    epochs: usize,
    #[arg(long, default_value_t = 32)]
    episodes_per_epoch: usize,
    #[arg(long, default_value_t = 8)]
    batch_size: usize,
    #[arg(long, default_value_t = 2)]
    samples: usize,
    #[arg(long, default_value_t = 256)]
    train_problems: usize,
    #[arg(long, default_value_t = 32)]
    validation_problems: usize,
    #[arg(long, default_value_t = 0.0001)]
    learning_rate: f64,
    #[arg(long, default_value_t = 0.01)]
    entropy: f64,
    #[arg(long, default_value_t = 0.02)]
    kl: f64,
    #[arg(long, default_value_t = 5)]
    patience: usize,
    #[arg(long, default_value_t = 43)]
    seed: u64,
    #[arg(long, default_value_t = 1000)]
    total_ms: u64,
    #[arg(long, default_value_t = 350)]
    stage_ms: u64,
    #[arg(long, default_value_t = 100)]
    local_ms: u64,
    #[arg(long, default_value_t = 64)]
    local_calls: usize,
    /// Produce merge-only output; latent lookahead must then remain disabled.
    #[arg(long)]
    no_refresh_latent: bool,
    #[arg(long, default_value_t = 50)]
    latent_epochs: usize,
}
fn finetune(options: &RlTraining) -> Result<()> {
    let parent = options
        .parent_models
        .clone()
        .unwrap_or_else(|| options.run_dir.join("models"));
    let output = options
        .model_dir
        .clone()
        .unwrap_or_else(|| options.run_dir.join("rl-models"));
    let mut command = Command::new(&options.python);
    let local_script = PathBuf::from("scripts/finetune_rl.py");
    command.arg(if local_script.is_file() {
        local_script
    } else {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("scripts/finetune_rl.py")
    });
    for (key, value) in [
        ("--run-dir", options.run_dir.display().to_string()),
        ("--parent-models", parent.display().to_string()),
        ("--model-dir", output.display().to_string()),
        ("--native", std::env::current_exe()?.display().to_string()),
        ("--device", options.device.clone()),
        ("--epochs", options.epochs.to_string()),
        (
            "--episodes-per-epoch",
            options.episodes_per_epoch.to_string(),
        ),
        ("--batch-size", options.batch_size.to_string()),
        ("--samples", options.samples.to_string()),
        ("--train-problems", options.train_problems.to_string()),
        (
            "--validation-problems",
            options.validation_problems.to_string(),
        ),
        ("--learning-rate", options.learning_rate.to_string()),
        ("--entropy", options.entropy.to_string()),
        ("--kl", options.kl.to_string()),
        ("--patience", options.patience.to_string()),
        ("--seed", options.seed.to_string()),
        ("--total-ms", options.total_ms.to_string()),
        ("--stage-ms", options.stage_ms.to_string()),
        ("--local-ms", options.local_ms.to_string()),
        ("--local-calls", options.local_calls.to_string()),
        ("--latent-epochs", options.latent_epochs.to_string()),
    ] {
        command.arg(key).arg(value);
    }
    if options.no_refresh_latent {
        command.arg("--no-refresh-latent");
    }
    if let Some(directory) = &options.online_dir {
        command
            .arg("--online-dir")
            .arg(directory)
            .arg("--online-fraction")
            .arg(options.online_fraction.to_string());
    }
    if !command
        .status()
        .context("start RL PyTorch trainer")?
        .success()
    {
        bail!("RL fine-tuning failed");
    }
    Ok(())
}
#[derive(Args)]
struct Preparation {
    #[arg(long, default_value = "../train")]
    dataset: PathBuf,
    #[arg(long, default_value = "runs/hilg-wl")]
    run_dir: PathBuf,
    #[arg(long, default_value_t = 1024)]
    problems: usize,
    #[arg(long, default_value_t = 4096)]
    synthetic: usize,
    #[arg(long, default_value_t = 4)]
    workers: usize,
    #[arg(long, default_value_t = 20261106)]
    generation_seed: u64,
    #[arg(long, default_value_t = 8)]
    max_roots: usize,
    #[arg(long, default_value_t = 128)]
    per_domain: usize,
    #[arg(long, default_value_t = 200_000)]
    max_ground_instances: usize,
    #[arg(long, default_value_t = 6)]
    levels: usize,
    #[arg(long, default_value_t = 12)]
    candidates: usize,
    #[arg(long, default_value_t = 350)]
    oracle_ms: u64,
    #[arg(long, default_value_t = 100)]
    local_ms: u64,
    #[arg(long, default_value_t = 2)]
    conflict_core_checks: usize,
    #[arg(long, default_value_t = 32)]
    hidden: usize,
    #[arg(long)]
    legacy_encoder: bool,
    #[arg(long, default_value_t = 3)]
    layers: usize,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value_t = 42)]
    split_seed: u64,
    #[arg(long)]
    retry_failed: bool,
}
#[derive(Args)]
struct Training {
    /// Pin a prepared package while another process prepares a later dataset.
    #[arg(long)]
    package: Option<String>,
    #[arg(long, default_value = "python3")]
    python: PathBuf,
    #[arg(long,default_value="auto",value_parser=["auto","cpu","cuda"])]
    device: String,
    #[arg(long, default_value_t = 100)]
    epochs: usize,
    #[arg(long, default_value_t = 64)]
    batch_size: usize,
    #[arg(long, default_value_t = 0.001)]
    learning_rate: f64,
    #[arg(long, default_value_t = 12)]
    patience: usize,
    #[arg(long, default_value_t = 4)]
    cpu_threads: usize,
    /// Separate output directory; defaults to RUN_DIR/models.
    #[arg(long)]
    model_dir: Option<PathBuf>,
}
#[derive(Args)]
struct Fit {
    #[arg(long, default_value = "runs/hilg-wl")]
    run_dir: PathBuf,
    #[command(flatten)]
    training: Training,
}
fn prepare(options: &Preparation) -> Result<()> {
    let mut rollout = RolloutOptions {
        max_ground_instances: options.max_ground_instances,
        max_levels: options.levels,
        max_candidates: options.candidates,
        trace: TraceLevel::Full,
        conflict_core_checks: options.conflict_core_checks,
        ..Default::default()
    };
    rollout.master.max_time_ms = options.oracle_ms;
    rollout.master.local_budget.max_time_ms = options.local_ms;
    let report = prepare_data(&PrepareOptions {
        dataset: options.dataset.clone(),
        run_dir: options.run_dir.clone(),
        problems: options.problems,
        synthetic: options.synthetic,
        workers: options.workers,
        seed: options.generation_seed,
        max_roots: options.max_roots,
        per_domain: options.per_domain,
        rollout,
        retry_failed: options.retry_failed,
    })?;
    eprintln!(
        "prepared {} rollouts from {} successful jobs ({} failed/partial), {} domain groups; {} new solutions",
        report.rollout_records,
        report.successful_jobs,
        report.failed_jobs,
        report.groups,
        report.synthetic_added
    );
    let report = export_data(
        &options.run_dir,
        &ExportOptions {
            wl: !options.legacy_encoder,
            hidden: options.hidden,
            layers: options.layers,
            seed: options.seed,
            split_seed: options.split_seed,
            ..Default::default()
        },
    )?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
fn fit(run_dir: &std::path::Path, options: &Training) -> Result<()> {
    if options.epochs == 0
        || options.batch_size == 0
        || options.patience == 0
        || options.cpu_threads == 0
        || !options.learning_rate.is_finite()
        || options.learning_rate <= 0.0
    {
        bail!("training counts and learning rate must be positive");
    }
    let local_script = PathBuf::from("scripts/train_all.py");
    let script = if local_script.is_file() {
        local_script
    } else {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("scripts/train_all.py")
    };
    let output = options
        .model_dir
        .clone()
        .unwrap_or_else(|| run_dir.join("models"));
    let mut command = Command::new(&options.python);
    command
        .arg(script)
        .arg("--run-dir")
        .arg(run_dir)
        .arg("--model-dir")
        .arg(output)
        .arg("--device")
        .arg(&options.device)
        .arg("--epochs")
        .arg(options.epochs.to_string())
        .arg("--batch-size")
        .arg(options.batch_size.to_string())
        .arg("--learning-rate")
        .arg(options.learning_rate.to_string())
        .arg("--patience")
        .arg(options.patience.to_string())
        .arg("--cpu-threads")
        .arg(options.cpu_threads.to_string())
        .arg("--native")
        .arg(std::env::current_exe()?);
    if let Some(package) = &options.package {
        command.arg("--package").arg(package);
    }
    let status = command.status().context(
        "start PyTorch trainer; install requirements-training.txt in the chosen Python environment",
    )?;
    if !status.success() {
        bail!("training exited with {status}");
    }
    Ok(())
}
fn main() -> Result<()> {
    match Cli::parse().command {
        Action::Prepare(p) => prepare(&p),
        Action::Rl(options) => finetune(&options),
        Action::GenerateOnline(options) => ptu_plan::learning::online::generate(&options),
        Action::RlEnvironment(options) => ptu_plan::learning::rl::serve(&options),
        Action::RlCheck {
            parent,
            checkpoint,
            parity,
        } => ptu_plan::learning::rl::validate_export(&parent, &checkpoint, &parity),
        Action::Export {
            run_dir,
            hidden,
            legacy_encoder,
            layers,
            seed,
            split_seed,
        } => {
            println!(
                "{}",
                serde_json::to_string_pretty(&export_data(
                    &run_dir,
                    &ExportOptions {
                        wl: !legacy_encoder,
                        hidden,
                        layers,
                        seed,
                        split_seed,
                        ..Default::default()
                    }
                )?)?
            );
            Ok(())
        }
        Action::Fit(f) => fit(&f.run_dir, &f.training),
        Action::All {
            preparation,
            training,
        } => {
            prepare(&preparation)?;
            fit(&preparation.run_dir, &training)
        }
        Action::Finalize { run_dir, model_dir } => finalize_bundle(&run_dir, &model_dir),
    }
}
