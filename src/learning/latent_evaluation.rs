//! Bounded open-loop drift evaluation on audited symbolic merge trajectories.
use super::{
    audit::RolloutAuditor, features::Observation, latent_training::LatentCheckpoint,
    model::Prediction,
};
use crate::{
    agglomeration,
    analysis::NetworkId,
    store::{DatasetStore, RecordKind, Split, group_split},
    training::sha256,
};
use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DriftOptions {
    pub depth: usize,
    pub max_records: usize,
    pub paths_per_record: usize,
    pub max_paths: usize,
}
impl Default for DriftOptions {
    fn default() -> Self {
        Self {
            depth: 4,
            max_records: 1000,
            paths_per_record: 8,
            max_paths: 1024,
        }
    }
}
impl DriftOptions {
    pub fn validate(&self) -> Result<()> {
        if !(1..=4).contains(&self.depth)
            || self.max_records == 0
            || self.paths_per_record == 0
            || self.max_paths == 0
        {
            bail!("invalid latent drift evaluation limits");
        }
        Ok(())
    }
}
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PredictionQuality {
    pub node_mse: f64,
    pub value_mae: f64,
    pub policy_decisions: usize,
    pub policy_top1_agreement: Option<f64>,
}
impl PredictionQuality {
    fn add(
        &mut self,
        hidden: &[Vec<f64>],
        target: &[Vec<f64>],
        prediction: &Prediction,
        truth: &Prediction,
    ) -> Result<()> {
        let count: usize = target.iter().map(Vec::len).sum();
        if count == 0
            || hidden.len() != target.len()
            || hidden.iter().zip(target).any(|(a, b)| a.len() != b.len())
        {
            bail!("unaligned drift comparison");
        }
        self.node_mse += hidden
            .iter()
            .zip(target)
            .flat_map(|(a, b)| a.iter().zip(b))
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / count as f64;
        self.value_mae += prediction
            .log_effort
            .iter()
            .zip(&truth.log_effort)
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>()
            / truth.log_effort.len() as f64;
        let best = |p: &Prediction| {
            (0..p.logits.len())
                .max_by(|&a, &b| p.logits[a].total_cmp(&p.logits[b]).then_with(|| b.cmp(&a)))
                .unwrap()
        };
        if truth.logits.len() > 1 {
            self.policy_decisions += 1;
            *self.policy_top1_agreement.get_or_insert(0.0) +=
                f64::from(best(prediction) == best(truth));
        }
        if !self.node_mse.is_finite() || !self.value_mae.is_finite() {
            bail!("nonfinite latent drift metrics");
        }
        Ok(())
    }
    fn average(&mut self, samples: usize) {
        if samples > 0 {
            self.node_mse /= samples as f64;
            self.value_mae /= samples as f64;
            if let Some(agreement) = &mut self.policy_top1_agreement {
                *agreement /= self.policy_decisions as f64;
            }
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepthQuality {
    pub depth: usize,
    pub samples: usize,
    pub learned: Option<PredictionQuality>,
    /// No learned dynamics: retain surviving embeddings and mean the merge pair.
    pub identity: Option<PredictionQuality>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftSplit {
    pub split: Split,
    pub records: Vec<String>,
    pub groups: Vec<String>,
    pub paths: usize,
    pub depths: Vec<DepthQuality>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftReport {
    pub schema_version: u32,
    pub snapshot_id: String,
    pub checkpoint_sha256: String,
    pub options: DriftOptions,
    pub audited_records: usize,
    pub total_paths: usize,
    pub truncated: bool,
    pub splits: [DriftSplit; 3],
}

/// The first actions are bounded audited counterfactuals in canonical order;
/// continuation follows the highest exact structural merge score. Neither the
/// learned model nor held-out targets choose these trajectories. No updates occur.
pub fn evaluate_latent_drift(
    store: &DatasetStore,
    checkpoint: &LatentCheckpoint,
    options: &DriftOptions,
) -> Result<DriftReport> {
    checkpoint.validate()?;
    options.validate()?;
    let snapshot = store.snapshot()?;
    let mut report = DriftReport {
        schema_version: 1,
        snapshot_id: snapshot.fingerprint()?,
        checkpoint_sha256: sha256(&serde_json::to_vec(checkpoint)?),
        options: *options,
        audited_records: 0,
        total_paths: 0,
        truncated: false,
        splits: [Split::Train, Split::Validation, Split::Test].map(|split| DriftSplit {
            split,
            records: Vec::new(),
            groups: Vec::new(),
            paths: 0,
            depths: (1..=options.depth)
                .map(|depth| DepthQuality {
                    depth,
                    samples: 0,
                    learned: None,
                    identity: None,
                })
                .collect(),
        }),
    };
    let mut groups: [BTreeSet<String>; 3] = Default::default();
    let mut auditor = RolloutAuditor::new(store, &snapshot);
    for entry in snapshot
        .entries
        .iter()
        .filter(|e| e.metadata.kind == RecordKind::Rollout)
    {
        if report.audited_records >= options.max_records || report.total_paths >= options.max_paths
        {
            report.truncated = true;
            break;
        }
        let record = auditor.audit(entry)?;
        report.audited_records += 1;
        let roots: Vec<_> = record
            .observation
            .actions
            .iter()
            .enumerate()
            .filter_map(|(i, a)| a.pair.map(|_| i))
            .take(options.paths_per_record)
            .collect();
        if roots.is_empty() {
            continue;
        }
        let split = match group_split(&record.group_id, checkpoint.options.split_seed) {
            Split::Train => 0,
            Split::Validation => 1,
            Split::Test => 2,
        };
        let (analysis, graph) = auditor.structure(&record)?;
        let initial = agglomeration::observe(
            analysis,
            graph,
            NetworkId::Initial,
            &record.clusters,
            &record.options.agglomeration,
        )?;
        let initial_hidden = checkpoint.encoder.encode(&record.observation)?;
        report.splits[split].records.push(entry.id.clone());
        groups[split].insert(record.group_id.clone());
        for root_action in roots {
            if report.total_paths >= options.max_paths {
                report.truncated = true;
                break;
            }
            report.total_paths += 1;
            report.splits[split].paths += 1;
            let mut state = initial.clone();
            let mut before = record.observation.clone();
            let mut hidden = initial_hidden.clone();
            let mut identity = initial_hidden.clone();
            let mut chosen = root_action;
            for metrics in &mut report.splits[split].depths {
                let candidates = agglomeration::legal_candidates(
                    graph,
                    NetworkId::Initial,
                    &state,
                    &record.options.agglomeration,
                )?;
                let (a, b) = before.actions[chosen]
                    .pair
                    .ok_or_else(|| anyhow::anyhow!("drift path selected STOP"))?;
                let candidate = candidates
                    .iter()
                    .find(|c| c.left == before.nodes[a].id && c.right == before.nodes[b].id)
                    .ok_or_else(|| anyhow::anyhow!("drift path has no legal symbolic merge"))?;
                let clusters = agglomeration::merged_clusters(&state.clusters, candidate)?;
                let after_state = agglomeration::observe(
                    analysis,
                    graph,
                    NetworkId::Initial,
                    &clusters,
                    &record.options.agglomeration,
                )?;
                let (mut after, next_candidates) = Observation::build(
                    analysis,
                    graph,
                    &after_state,
                    &record.options.agglomeration,
                )?;
                after.context.clone_from(&before.context);
                hidden = checkpoint
                    .model
                    .transition(&before, &hidden, chosen, &after)?;
                identity = checkpoint
                    .model
                    .inputs(&before, &identity, chosen, &after)?
                    .into_iter()
                    .map(|row| row[..checkpoint.model.hidden].to_vec())
                    .collect();
                let target = checkpoint.encoder.encode(&after)?;
                let truth = checkpoint.encoder.predict_encoded(&after, &target)?;
                let prediction = checkpoint.encoder.predict_encoded(&after, &hidden)?;
                let identity_prediction = checkpoint.encoder.predict_encoded(&after, &identity)?;
                metrics.samples += 1;
                metrics
                    .learned
                    .get_or_insert_with(PredictionQuality::default)
                    .add(&hidden, &target, &prediction, &truth)?;
                metrics
                    .identity
                    .get_or_insert_with(PredictionQuality::default)
                    .add(&identity, &target, &identity_prediction, &truth)?;
                if after_state.clusters.len() <= record.options.agglomeration.target_clusters {
                    break;
                }
                // Continuation ranking is structural and independent of the target encoder.
                let Some(next) = next_candidates
                    .iter()
                    .enumerate()
                    .max_by(|(ia, a), (ib, b)| {
                        a.score
                            .total
                            .total_cmp(&b.score.total)
                            .then_with(|| ib.cmp(ia))
                    })
                    .map(|(i, _)| i)
                else {
                    break;
                };
                state = after_state;
                before = after;
                chosen = next;
            }
        }
    }
    if report.total_paths == 0 {
        bail!("no audited merge paths available for drift evaluation");
    }
    for (split, groups) in report.splits.iter_mut().zip(groups) {
        split.groups = groups.into_iter().collect();
        for depth in &mut split.depths {
            if let Some(quality) = &mut depth.learned {
                quality.average(depth.samples);
            }
            if let Some(quality) = &mut depth.identity {
                quality.average(depth.samples);
            }
        }
    }
    Ok(report)
}

impl DriftReport {
    pub fn save(&self, path: &std::path::Path) -> Result<()> {
        crate::store::atomic_write(path, &serde_json::to_vec_pretty(self)?)
    }
}
