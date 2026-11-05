use super::features::Observation;
use super::model::{MergeModel, Prediction};
use crate::agglomeration::{self, Agglomeration, AgglomerationOptions, MergeReason, MergeStep};
use crate::analysis::{HtnAnalysis, NetworkId};
use crate::interaction::InteractionGraph;
use crate::partition::MicroPartition;
use crate::training::sha256;
use anyhow::Result;
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub enum StopReason {
    PolicyStop,
    TargetReached,
    NoLegalMerges,
    SearchLimit,
}
#[derive(Debug, Clone, Serialize)]
pub struct PolicyDecision {
    pub observation: Observation,
    pub prediction: Prediction,
    pub action: usize,
    pub feature_time_us: u128,
    pub inference_time_us: u128,
}
#[derive(Debug, Clone, Serialize)]
pub struct PolicyTrace {
    pub feature_version: u32,
    pub model_sha256: String,
    pub value_weight: f64,
    pub decisions: Vec<PolicyDecision>,
    pub stop_reason: StopReason,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub beam: Option<super::beam::BeamStatistics>,
}

/// Predictions only select from freshly recomputed legal contractions or STOP.
/// Solver feasibility still requires symbolic replay of the complete plan.
pub fn learned_agglomerate(
    analysis: &HtnAnalysis,
    graph: &InteractionGraph,
    seeds: &[MicroPartition],
    options: &AgglomerationOptions,
    model: &MergeModel,
    value_weight: f64,
) -> Result<(Agglomeration, PolicyTrace)> {
    learned_agglomerate_with_context(
        analysis,
        graph,
        seeds,
        options,
        model,
        value_weight,
        &crate::master::MasterOptions::default(),
    )
}

pub fn learned_agglomerate_with_context(
    analysis: &HtnAnalysis,
    graph: &InteractionGraph,
    seeds: &[MicroPartition],
    options: &AgglomerationOptions,
    model: &MergeModel,
    value_weight: f64,
    context: &crate::master::MasterOptions,
) -> Result<(Agglomeration, PolicyTrace)> {
    agglomerate_with_selector(
        analysis,
        graph,
        seeds,
        options,
        model,
        value_weight,
        context,
        None,
    )
}
pub type ActionSampler<'a> = dyn FnMut(&[f64]) -> usize + 'a;

/// An optional seeded sampler receives normalized action probabilities. Scores
/// include the value head, exactly as in deterministic planner inference.
#[allow(clippy::too_many_arguments)]
pub fn agglomerate_with_selector(
    analysis: &HtnAnalysis,
    graph: &InteractionGraph,
    seeds: &[MicroPartition],
    options: &AgglomerationOptions,
    model: &MergeModel,
    value_weight: f64,
    context: &crate::master::MasterOptions,
    mut sampler: Option<&mut ActionSampler<'_>>,
) -> Result<(Agglomeration, PolicyTrace)> {
    model.validate()?;
    if !value_weight.is_finite() || value_weight < 0.0 {
        anyhow::bail!("invalid policy value weight");
    }
    let mut state = agglomeration::observe(analysis, graph, NetworkId::Initial, seeds, options)?;
    let mut history = state.merges.clone();
    let mut decisions = Vec::new();
    let stop_reason = loop {
        if state.clusters.len() <= options.target_clusters {
            break StopReason::TargetReached;
        }
        let start = std::time::Instant::now();
        let (mut observation, candidates) = Observation::build(analysis, graph, &state, options)?;
        observation.set_search_context(context);
        let feature_time_us = start.elapsed().as_micros();
        if candidates.is_empty() {
            break StopReason::NoLegalMerges;
        }
        let start = std::time::Instant::now();
        let (mut action, prediction) = model.choose(&observation, value_weight)?;
        if let Some(sampler) = &mut sampler {
            action = sampler(&action_probabilities(&prediction, value_weight)?);
            if action >= observation.actions.len() {
                anyhow::bail!("sampler returned an illegal action");
            }
        }
        let inference_time_us = start.elapsed().as_micros();
        let stopped = observation.actions[action].pair.is_none();
        decisions.push(PolicyDecision {
            observation,
            prediction,
            action,
            feature_time_us,
            inference_time_us,
        });
        if stopped {
            break StopReason::PolicyStop;
        }
        let candidate = &candidates[action];
        let clusters = agglomeration::merged_clusters(&state.clusters, candidate)?;
        let parent = clusters.iter().max_by_key(|c| c.id).unwrap().clone();
        history.push(MergeStep {
            children: vec![candidate.left, candidate.right],
            parent,
            reason: MergeReason::Policy,
            score: candidate.score.clone(),
        });
        state = agglomeration::observe(analysis, graph, NetworkId::Initial, &clusters, options)?;
    };
    state.seeds = seeds.to_vec();
    state.merges = history;
    let trace = PolicyTrace {
        feature_version: super::features::FEATURE_VERSION,
        model_sha256: sha256(&serde_json::to_vec(model)?),
        value_weight,
        decisions,
        stop_reason,
        beam: None,
    };
    Ok((state, trace))
}

/// Stable categorical distribution used by both RL collection and training.
pub fn action_probabilities(prediction: &Prediction, value_weight: f64) -> Result<Vec<f64>> {
    if !value_weight.is_finite()
        || value_weight < 0.0
        || prediction.logits.is_empty()
        || prediction.logits.len() != prediction.log_effort.len()
    {
        anyhow::bail!("invalid action distribution");
    }
    let scores: Vec<_> = prediction
        .logits
        .iter()
        .zip(&prediction.log_effort)
        .map(|(p, v)| p - value_weight * v)
        .collect();
    if scores.iter().any(|s| !s.is_finite()) {
        anyhow::bail!("nonfinite action scores");
    }
    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<_> = scores.iter().map(|s| (s - max).exp()).collect();
    let total: f64 = weights.iter().sum();
    Ok(weights.iter().map(|w| w / total).collect())
}
