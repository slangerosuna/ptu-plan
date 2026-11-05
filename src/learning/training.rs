use super::features::Observation;
use super::model::{MergeModel, ModelConfig, Supervision, random};
use super::rollout::RolloutAuditor;
use crate::store::{DatasetStore, Entry, Metadata, RecordKind, Split, group_split};
use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainOptions {
    pub model: ModelConfig,
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub gradient_clip: f64,
    pub value_loss_weight: f64,
    pub master_effort_weight: f64,
    pub seed: u64,
    pub split_seed: u64,
    pub max_records: usize,
}
impl Default for TrainOptions {
    fn default() -> Self {
        Self {
            model: ModelConfig::default(),
            epochs: 50,
            batch_size: 8,
            learning_rate: 0.003,
            weight_decay: 0.0001,
            gradient_clip: 5.0,
            value_loss_weight: 1.0,
            master_effort_weight: 1.0,
            seed: 42,
            split_seed: 42,
            max_records: 10_000,
        }
    }
}
impl TrainOptions {
    pub fn validate(&self) -> Result<()> {
        if !(1..=32).contains(&self.model.hidden)
            || !(1..=3).contains(&self.model.layers)
            || self.epochs == 0
            || self.batch_size == 0
            || self.max_records == 0
            || !self.learning_rate.is_finite()
            || self.learning_rate <= 0.0
            || !self.gradient_clip.is_finite()
            || self.gradient_clip <= 0.0
            || [
                self.weight_decay,
                self.value_loss_weight,
                self.master_effort_weight,
            ]
            .iter()
            .any(|v| !v.is_finite() || *v < 0.0)
        {
            bail!("invalid training or optimizer options");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Adam {
    pub step: usize,
    pub first: Vec<f64>,
    pub second: Vec<f64>,
}
impl Adam {
    pub fn new(parameters: usize) -> Self {
        Self {
            step: 0,
            first: vec![0.0; parameters],
            second: vec![0.0; parameters],
        }
    }
    pub fn update(
        &mut self,
        model: &mut MergeModel,
        gradient: &[f64],
        options: &TrainOptions,
    ) -> Result<()> {
        self.update_parameters(&mut model.parameters, gradient, options)?;
        model.validate()
    }
    pub fn update_parameters(
        &mut self,
        parameters: &mut [f64],
        gradient: &[f64],
        options: &TrainOptions,
    ) -> Result<()> {
        if self.first.len() != parameters.len()
            || self.second.len() != parameters.len()
            || gradient.len() != parameters.len()
            || parameters.iter().any(|v| !v.is_finite())
            || gradient
                .iter()
                .chain(&self.first)
                .chain(&self.second)
                .any(|v| !v.is_finite())
            || self.second.iter().any(|v| *v < 0.0)
        {
            bail!("invalid optimizer state or gradient");
        }
        let norm = gradient.iter().fold(0.0f64, |sum, g| sum.hypot(*g));
        let scale = if norm > options.gradient_clip {
            options.gradient_clip / norm
        } else {
            1.0
        };
        self.step = self
            .step
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("optimizer step overflow"))?;
        let correction1 = 1.0 - 0.9f64.powf(self.step as f64);
        let correction2 = 1.0 - 0.999f64.powf(self.step as f64);
        for (i, parameter) in parameters.iter_mut().enumerate() {
            let g = gradient[i] * scale;
            self.first[i] = 0.9 * self.first[i] + 0.1 * g;
            self.second[i] = 0.999 * self.second[i] + 0.001 * g * g;
            *parameter -= options.learning_rate
                * (self.first[i] / correction1 / (self.second[i] / correction2).sqrt().max(1e-8)
                    + options.weight_decay * *parameter);
        }
        if parameters.iter().any(|v| !v.is_finite()) {
            bail!("nonfinite optimizer update");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    pub schema_version: u32,
    pub model: MergeModel,
    pub optimizer: Adam,
    pub completed_epochs: usize,
    pub options: TrainOptions,
    pub snapshot_id: String,
    pub training_records: Vec<String>,
    pub validation_records: Vec<String>,
    pub test_records: Vec<String>,
    pub training_groups: Vec<String>,
    pub validation_groups: Vec<String>,
    pub test_groups: Vec<String>,
}
impl Checkpoint {
    pub fn save(&self, path: &std::path::Path) -> Result<()> {
        self.validate()?;
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(std::path::Path::new("."));
        crate::store::atomic_write(
            &parent.join(
                path.file_name()
                    .ok_or_else(|| anyhow::anyhow!("missing checkpoint filename"))?,
            ),
            &serde_json::to_vec_pretty(self)?,
        )
    }
    pub fn validate(&self) -> Result<()> {
        self.model.validate()?;
        self.options.validate()?;
        if self.schema_version != 1
            || self.model.config.hidden != self.options.model.hidden
            || self.model.config.layers != self.options.model.layers
            || self.optimizer.first.len() != self.model.parameters.len()
            || self.optimizer.second.len() != self.model.parameters.len()
            || self
                .optimizer
                .first
                .iter()
                .chain(&self.optimizer.second)
                .any(|v| !v.is_finite())
            || self.optimizer.second.iter().any(|v| *v < 0.0)
        {
            bail!("invalid checkpoint optimizer");
        }
        let mut records = BTreeSet::new();
        if self
            .training_records
            .iter()
            .chain(&self.validation_records)
            .chain(&self.test_records)
            .any(|id| !records.insert(id))
        {
            bail!("checkpoint record split leakage");
        }
        let groups = [
            &self.training_groups,
            &self.validation_groups,
            &self.test_groups,
        ];
        let mut seen = BTreeSet::new();
        for (i, group) in groups.iter().enumerate() {
            for id in *group {
                if !seen.insert(id)
                    || group_split(id, self.options.split_seed)
                        != [Split::Train, Split::Validation, Split::Test][i]
                {
                    bail!("checkpoint split leakage");
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Metrics {
    pub records: usize,
    pub pairs: usize,
    pub mean_loss: f64,
    pub pair_accuracy: Option<f64>,
    pub exact_value_mae: Option<f64>,
    pub censored_values: usize,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingReport {
    pub schema_version: u32,
    pub options: TrainOptions,
    pub snapshot_id: String,
    pub skipped_without_labels: usize,
    pub initial_training: Metrics,
    pub training: Metrics,
    pub validation: Metrics,
    pub test: Metrics,
    pub epoch_losses: Vec<f64>,
    pub checkpoint_id: Option<String>,
}

pub struct Example {
    pub id: String,
    pub group: String,
    pub observation: Observation,
    pub labels: Supervision,
}

fn metrics(model: &MergeModel, examples: &[Example], weight: f64) -> Result<Metrics> {
    let mut result = Metrics {
        records: examples.len(),
        ..Default::default()
    };
    let mut correct = 0;
    let mut error = 0.0;
    let mut exact = 0;
    for example in examples {
        result.mean_loss += model.loss(&example.observation, &example.labels, weight)?;
        let prediction = model.predict(&example.observation)?;
        for &(better, worse) in &example.labels.preferences {
            result.pairs += 1;
            correct += usize::from(prediction.logits[better] > prediction.logits[worse]);
        }
        for target in &example.labels.values {
            if target.lower_bound {
                result.censored_values += 1;
            } else {
                error += (prediction.log_effort[target.action] - target.log_effort).abs();
                exact += 1;
            }
        }
    }
    if !examples.is_empty() {
        result.mean_loss /= examples.len() as f64;
    }
    if result.pairs > 0 {
        result.pair_accuracy = Some(correct as f64 / result.pairs as f64);
    }
    if exact > 0 {
        result.exact_value_mae = Some(error / exact as f64);
    }
    Ok(result)
}

pub fn fit(
    model: &mut MergeModel,
    optimizer: &mut Adam,
    examples: &[Example],
    options: &TrainOptions,
    start_epoch: usize,
) -> Result<Vec<f64>> {
    options.validate()?;
    if examples.is_empty() {
        bail!("training examples are empty");
    }
    let end = start_epoch
        .checked_add(options.epochs)
        .ok_or_else(|| anyhow::anyhow!("epoch overflow"))?;
    let mut losses = Vec::new();
    for epoch in start_epoch..end {
        let mut order: Vec<_> = (0..examples.len()).collect();
        let mut seed = options.seed.wrapping_add(epoch as u64);
        for i in (1..order.len()).rev() {
            let j = (random(&mut seed) % (i as u64 + 1)) as usize;
            order.swap(i, j);
        }
        let mut loss = 0.0;
        for batch in order.chunks(options.batch_size) {
            let mut gradient = vec![0.0; model.parameters.len()];
            for &index in batch {
                let example = &examples[index];
                let (value, delta) = model.loss_gradient(
                    &example.observation,
                    &example.labels,
                    options.value_loss_weight,
                )?;
                loss += value;
                for (g, d) in gradient.iter_mut().zip(delta) {
                    *g += d / batch.len() as f64;
                }
            }
            optimizer.update(model, &gradient, options)?;
        }
        losses.push(loss / examples.len() as f64);
    }
    Ok(losses)
}

pub fn train_store(
    store: &DatasetStore,
    options: &TrainOptions,
    resume: Option<Checkpoint>,
) -> Result<(Entry, Entry, TrainingReport)> {
    options.validate()?;
    let snapshot = store.snapshot()?;
    let snapshot_id = snapshot.fingerprint()?;
    let mut partitions = [Vec::<Example>::new(), Vec::new(), Vec::new()];
    let mut skipped = 0;
    let mut auditor = RolloutAuditor::new(store, &snapshot);
    for entry in snapshot
        .entries
        .iter()
        .filter(|e| e.metadata.kind == RecordKind::Rollout)
        .take(options.max_records)
    {
        let record = auditor.audit(entry)?;
        let labels = record.supervision(options.master_effort_weight)?;
        if labels.preferences.is_empty() && labels.values.is_empty() {
            skipped += 1;
            continue;
        }
        let split = match group_split(&record.group_id, options.split_seed) {
            Split::Train => 0,
            Split::Validation => 1,
            Split::Test => 2,
        };
        partitions[split].push(Example {
            id: entry.id.clone(),
            group: record.group_id,
            observation: record.observation,
            labels,
        });
    }
    drop(auditor);
    let [training, validation, test] = partitions;
    if training.is_empty() {
        bail!(
            "no labeled training groups; collect more domains or select another explicit split seed"
        );
    }
    let (mut model, mut optimizer, start_epoch) = if let Some(checkpoint) = resume {
        checkpoint.validate()?;
        if checkpoint.options.split_seed != options.split_seed
            || checkpoint.options.seed != options.seed
            || checkpoint.model.config.hidden != options.model.hidden
            || checkpoint.model.config.layers != options.model.layers
        {
            bail!("resume configuration changes model shape or split/sampler seed");
        }
        (
            checkpoint.model,
            checkpoint.optimizer,
            checkpoint.completed_epochs,
        )
    } else {
        let model = MergeModel::new(options.model, options.seed)?;
        let optimizer = Adam::new(model.parameters.len());
        (model, optimizer, 0)
    };
    let initial_training = metrics(&model, &training, options.value_loss_weight)?;
    let epoch_losses = fit(&mut model, &mut optimizer, &training, options, start_epoch)?;
    let mut report = TrainingReport {
        schema_version: 1,
        options: options.clone(),
        snapshot_id: snapshot_id.clone(),
        skipped_without_labels: skipped,
        initial_training,
        training: metrics(&model, &training, options.value_loss_weight)?,
        validation: metrics(&model, &validation, options.value_loss_weight)?,
        test: metrics(&model, &test, options.value_loss_weight)?,
        epoch_losses,
        checkpoint_id: None,
    };
    let ids = |examples: &[Example]| examples.iter().map(|e| e.id.clone()).collect();
    let groups = |examples: &[Example]| {
        examples
            .iter()
            .map(|e| e.group.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    };
    let checkpoint = Checkpoint {
        schema_version: 1,
        model,
        optimizer,
        completed_epochs: start_epoch + options.epochs,
        options: options.clone(),
        snapshot_id,
        training_records: ids(&training),
        validation_records: ids(&validation),
        test_records: ids(&test),
        training_groups: groups(&training),
        validation_groups: groups(&validation),
        test_groups: groups(&test),
    };
    checkpoint.validate()?;
    let model_entry = store.put(
        Metadata {
            kind: RecordKind::Model,
            group_id: None,
            source_id: None,
        },
        &checkpoint,
    )?;
    report.checkpoint_id = Some(model_entry.id.clone());
    let report_entry = store.put(
        Metadata {
            kind: RecordKind::TrainingReport,
            group_id: None,
            source_id: Some(model_entry.id.clone()),
        },
        &report,
    )?;
    Ok((model_entry, report_entry, report))
}

pub fn group_counts(store: &DatasetStore, seed: u64) -> Result<BTreeMap<String, usize>> {
    let mut result = BTreeMap::new();
    for entry in store
        .snapshot()?
        .entries
        .iter()
        .filter(|e| e.metadata.kind == RecordKind::Rollout)
    {
        let group = entry
            .metadata
            .group_id
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("rollout missing group"))?;
        *result
            .entry(format!("{:?}", group_split(group, seed)))
            .or_default() += 1;
    }
    Ok(result)
}
