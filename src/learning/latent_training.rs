//! Audited symbolic before/after merges with frozen-encoder regression targets.
use super::{
    audit::RolloutAuditor,
    features::Observation,
    latent::{LatentModel, TransitionExample},
    model::{MergeModel, random},
    training::{Adam, TrainOptions},
};
use crate::{
    agglomeration,
    analysis::NetworkId,
    store::{DatasetStore, Entry, Metadata, RecordKind, Split, group_split},
};
use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatentCheckpoint {
    pub schema_version: u32,
    pub encoder: MergeModel,
    pub model: LatentModel,
    pub optimizer: Adam,
    pub options: TrainOptions,
    pub completed_epochs: usize,
    pub snapshot_id: String,
    pub records: [Vec<String>; 3],
    pub groups: [Vec<String>; 3],
}
impl LatentCheckpoint {
    pub fn validate(&self) -> Result<()> {
        self.model.validate(&self.encoder)?;
        self.options.validate()?;
        if self.schema_version != 1
            || self.optimizer.first.len() != self.model.parameters.len()
            || self.optimizer.second.len() != self.model.parameters.len()
            || self
                .optimizer
                .first
                .iter()
                .chain(&self.optimizer.second)
                .any(|v| !v.is_finite())
            || self.optimizer.second.iter().any(|v| *v < 0.0)
            || self.encoder.config.hidden != self.options.model.hidden
            || self.encoder.config.layers != self.options.model.layers
        {
            bail!("invalid latent checkpoint metadata/optimizer");
        }
        let mut records = BTreeSet::new();
        let mut groups = BTreeSet::new();
        for i in 0..3 {
            if self.records[i].iter().any(|id| !records.insert(id))
                || self.groups[i].iter().any(|id| {
                    !groups.insert(id)
                        || group_split(id, self.options.split_seed)
                            != [Split::Train, Split::Validation, Split::Test][i]
                })
            {
                bail!("latent checkpoint split leakage");
            }
        }
        Ok(())
    }
    pub fn save(&self, path: &std::path::Path) -> Result<()> {
        self.validate()?;
        crate::store::atomic_write(path, &serde_json::to_vec_pretty(self)?)
    }
}
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LatentMetrics {
    pub transitions: usize,
    pub node_mse: Option<f64>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatentReport {
    pub schema_version: u32,
    pub snapshot_id: String,
    pub encoder_sha256: String,
    pub initial_training: LatentMetrics,
    pub training: LatentMetrics,
    pub validation: LatentMetrics,
    pub test: LatentMetrics,
    pub epoch_losses: Vec<f64>,
}
fn metrics(model: &LatentModel, examples: &[TransitionExample]) -> Result<LatentMetrics> {
    let mut loss = 0.0;
    for example in examples {
        loss += model.loss(example)?;
    }
    Ok(LatentMetrics {
        transitions: examples.len(),
        node_mse: (!examples.is_empty()).then(|| loss / examples.len() as f64),
    })
}

pub fn train_latent_store(
    store: &DatasetStore,
    encoder: &MergeModel,
    options: &TrainOptions,
    resume: Option<LatentCheckpoint>,
) -> Result<(Entry, Entry, LatentReport)> {
    encoder.validate()?;
    options.validate()?;
    if encoder.config.hidden != options.model.hidden
        || encoder.config.layers != options.model.layers
    {
        bail!("latent training encoder/config mismatch");
    }
    let snapshot = store.snapshot()?;
    let snapshot_id = snapshot.fingerprint()?;
    let (mut model, mut optimizer, start_epoch) = if let Some(checkpoint) = resume {
        checkpoint.validate()?;
        if checkpoint.model.encoder_sha256 != crate::training::sha256(&serde_json::to_vec(encoder)?)
            || checkpoint.options.seed != options.seed
            || checkpoint.options.split_seed != options.split_seed
        {
            bail!("latent resume changes encoder or split/sampler seed");
        }
        (
            checkpoint.model,
            checkpoint.optimizer,
            checkpoint.completed_epochs,
        )
    } else {
        let model = LatentModel::new(encoder, options.seed)?;
        let optimizer = Adam::new(model.parameters.len());
        (model, optimizer, 0)
    };
    let mut examples: [Vec<TransitionExample>; 3] = Default::default();
    let mut records: [Vec<String>; 3] = Default::default();
    let mut groups: [BTreeSet<String>; 3] = Default::default();
    let mut auditor = RolloutAuditor::new(store, &snapshot);
    for entry in snapshot
        .entries
        .iter()
        .filter(|e| e.metadata.kind == RecordKind::Rollout)
        .take(options.max_records)
    {
        let record = auditor.audit(entry)?;
        let (analysis, graph) = auditor.structure(&record)?;
        let state = agglomeration::observe(
            analysis,
            graph,
            NetworkId::Initial,
            &record.clusters,
            &record.options.agglomeration,
        )?;
        let candidates = agglomeration::legal_candidates(
            graph,
            NetworkId::Initial,
            &state,
            &record.options.agglomeration,
        )?;
        let before = encoder.encode(&record.observation)?;
        let split = match group_split(&record.group_id, options.split_seed) {
            Split::Train => 0,
            Split::Validation => 1,
            Split::Test => 2,
        };
        let start = examples[split].len();
        for (action, input) in record.observation.actions.iter().enumerate() {
            let Some((a, b)) = input.pair else {
                continue;
            };
            let candidate = candidates
                .iter()
                .find(|c| {
                    c.left == record.observation.nodes[a].id
                        && c.right == record.observation.nodes[b].id
                })
                .ok_or_else(|| anyhow::anyhow!("audited transition has no legal candidate"))?;
            let clusters = agglomeration::merged_clusters(&state.clusters, candidate)?;
            let child = agglomeration::observe(
                analysis,
                graph,
                NetworkId::Initial,
                &clusters,
                &record.options.agglomeration,
            )?;
            let (mut after, _) =
                Observation::build(analysis, graph, &child, &record.options.agglomeration)?;
            after.set_search_context(&record.options.master);
            // Preserve the audited serialized context exactly; JSON float
            // round trips can differ from recomputed logarithms by one ULP.
            after.context.clone_from(&record.observation.context);
            examples[split].push(TransitionExample {
                inputs: model.inputs(&record.observation, &before, action, &after)?,
                target: encoder.encode(&after)?,
            });
        }
        if examples[split].len() > start {
            records[split].push(entry.id.clone());
            groups[split].insert(record.group_id);
        }
    }
    drop(auditor);
    if examples[0].is_empty() {
        bail!(
            "no audited training merge transitions; collect multi-root rollouts or choose an explicit split seed"
        );
    }
    let initial_training = metrics(&model, &examples[0])?;
    let end = start_epoch
        .checked_add(options.epochs)
        .ok_or_else(|| anyhow::anyhow!("latent epoch overflow"))?;
    let mut epoch_losses = Vec::new();
    for epoch in start_epoch..end {
        let mut order: Vec<_> = (0..examples[0].len()).collect();
        let mut seed = options.seed.wrapping_add(epoch as u64);
        for i in (1..order.len()).rev() {
            let j = (random(&mut seed) % (i as u64 + 1)) as usize;
            order.swap(i, j);
        }
        let mut loss = 0.0;
        for batch in order.chunks(options.batch_size) {
            let mut gradient = vec![0.0; model.parameters.len()];
            for &i in batch {
                let (l, g) = model.loss_gradient(&examples[0][i])?;
                loss += l;
                for (total, delta) in gradient.iter_mut().zip(g) {
                    *total += delta / batch.len() as f64;
                }
            }
            optimizer.update_parameters(&mut model.parameters, &gradient, options)?;
        }
        epoch_losses.push(loss / examples[0].len() as f64);
    }
    let report = LatentReport {
        schema_version: 1,
        snapshot_id: snapshot_id.clone(),
        encoder_sha256: model.encoder_sha256.clone(),
        initial_training,
        training: metrics(&model, &examples[0])?,
        validation: metrics(&model, &examples[1])?,
        test: metrics(&model, &examples[2])?,
        epoch_losses,
    };
    let checkpoint = LatentCheckpoint {
        schema_version: 1,
        encoder: encoder.clone(),
        model,
        optimizer,
        options: options.clone(),
        completed_epochs: end,
        snapshot_id,
        records,
        groups: groups.map(|g| g.into_iter().collect()),
    };
    checkpoint.validate()?;
    let model_entry = store.put(
        Metadata {
            kind: RecordKind::LatentModel,
            group_id: None,
            source_id: None,
        },
        &checkpoint,
    )?;
    let report_entry = store.put(
        Metadata {
            kind: RecordKind::LatentTrainingReport,
            group_id: None,
            source_id: Some(model_entry.id.clone()),
        },
        &report,
    )?;
    Ok((model_entry, report_entry, report))
}
