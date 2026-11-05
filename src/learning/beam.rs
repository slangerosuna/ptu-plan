//! Receding-horizon learned beam search with exact quotient re-encoding.
use super::{
    features::Observation,
    model::{MergeModel, Prediction},
    policy::{PolicyDecision, PolicyTrace, StopReason},
};
use crate::{
    agglomeration::{
        self, Agglomeration, AgglomerationOptions, MergeCandidate, MergeReason, MergeStep,
    },
    analysis::{HtnAnalysis, NetworkId},
    interaction::InteractionGraph,
    master::MasterOptions,
    partition::MicroPartition,
    training::sha256,
};
use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BeamOptions {
    pub width: usize,
    pub depth: usize,
    pub branching: usize,
    pub max_evaluations: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    fn branch(log_probability: f64, score: f64, terminal: bool) -> Branch {
        Branch {
            state: Agglomeration {
                seeds: vec![],
                merges: vec![],
                clusters: vec![],
                features: BTreeMap::new(),
                quotient_edges: vec![],
            },
            first: None,
            first_merge: None,
            log_probability,
            score,
            terminal,
            imagined: None,
        }
    }
    #[test]
    fn expandable_transpositions_keep_the_best_probability_for_any_shared_suffix() {
        let best_prefix = branch(-1.0, -20.0, false);
        let misleading_value = branch(-2.0, -3.0, false);
        assert!(dominates(&best_prefix, &misleading_value));
        assert!(!dominates(&misleading_value, &best_prefix));
        // Once these paths share a suffix, its critic replaces the stale value.
        for (suffix_log_probability, leaf_value) in [(-0.1, 2.0), (-4.0, 100.0)] {
            assert!(
                best_prefix.log_probability + suffix_log_probability - leaf_value
                    > misleading_value.log_probability + suffix_log_probability - leaf_value
            );
        }
        // Completed paths have no future critic to replace their final score.
        assert!(dominates(
            &branch(-2.0, -3.0, true),
            &branch(-1.0, -20.0, true)
        ));
    }
}
impl Default for BeamOptions {
    fn default() -> Self {
        Self {
            width: 8,
            depth: 3,
            branching: 8,
            max_evaluations: 10_000,
        }
    }
}
#[derive(Debug, Clone, Default, Serialize)]
pub struct BeamStatistics {
    pub width: usize,
    pub depth: usize,
    pub branching: usize,
    pub evaluations: usize,
    pub prediction_cache_hits: usize,
    pub generated: usize,
    pub deduplicated: usize,
    pub rounds: usize,
    pub truncated: bool,
    pub feature_time_us: u128,
    pub quotient_time_us: u128,
    pub inference_time_us: u128,
    pub elapsed_us: u128,
    pub exact_encodings: usize,
    pub latent_transitions: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latent_model_sha256: Option<String>,
}
#[derive(Clone)]
struct ImaginedTransition {
    observation: Arc<Observation>,
    hidden: Arc<Vec<Vec<f64>>>,
    action: usize,
}
#[derive(Clone)]
struct Branch {
    state: Agglomeration,
    first: Option<Arc<PolicyDecision>>,
    first_merge: Option<MergeCandidate>,
    log_probability: f64,
    score: f64,
    terminal: bool,
    imagined: Option<ImaginedTransition>,
}
fn key(state: &Agglomeration) -> Vec<Vec<crate::interaction::NodeId>> {
    let mut key: Vec<_> = state
        .clusters
        .iter()
        .map(|c| {
            let mut nodes = c.nodes.clone();
            nodes.sort();
            nodes
        })
        .collect();
    key.sort();
    key
}

type PredictionKey = Vec<(usize, Vec<crate::interaction::NodeId>)>;
struct CachedPrediction {
    prediction: Prediction,
    bytes: usize,
}
#[derive(Default)]
struct PredictionCache {
    entries: BTreeMap<PredictionKey, CachedPrediction>,
    bytes: usize,
}
impl PredictionCache {
    fn get(&self, key: &PredictionKey) -> Option<Prediction> {
        self.entries.get(key).map(|p| p.prediction.clone())
    }
    fn insert(&mut self, key: PredictionKey, prediction: &Prediction) {
        let bytes = (prediction.logits.len() + prediction.log_effort.len())
            * std::mem::size_of::<f64>()
            + key
                .iter()
                .map(|(_, nodes)| {
                    std::mem::size_of::<(usize, Vec<crate::interaction::NodeId>)>()
                        + nodes.len() * std::mem::size_of::<crate::interaction::NodeId>()
                })
                .sum::<usize>();
        const LIMIT: usize = 16 * 1024 * 1024;
        if bytes > LIMIT {
            return;
        }
        while self.bytes + bytes > LIMIT || self.entries.len() >= 128 {
            if let Some((_, old)) = self.entries.pop_first() {
                self.bytes -= old.bytes;
            } else {
                break;
            }
        }
        self.bytes += bytes;
        self.entries.insert(
            key,
            CachedPrediction {
                prediction: prediction.clone(),
                bytes,
            },
        );
    }
}

fn dominates(candidate: &Branch, old: &Branch) -> bool {
    // Any shared suffix adds the same log probability and replaces the latest
    // value estimate. An incoming action's value cannot dominate future paths.
    let rank = |b: &Branch| {
        if b.terminal {
            b.score
        } else {
            b.log_probability
        }
    };
    rank(candidate)
        .total_cmp(&rank(old))
        .then_with(|| {
            old.first
                .as_ref()
                .map(|d| d.action)
                .cmp(&candidate.first.as_ref().map(|d| d.action))
        })
        .is_gt()
}

/// Search a shallow beam, commit its first choice, and replan from that exact
/// symbolic quotient. Values rank nodes; they never establish feasibility.
#[allow(clippy::too_many_arguments)]
pub fn beam_agglomerate(
    analysis: &HtnAnalysis,
    graph: &InteractionGraph,
    seeds: &[MicroPartition],
    options: &AgglomerationOptions,
    model: &MergeModel,
    value_weight: f64,
    context: &MasterOptions,
    beam: &BeamOptions,
) -> Result<(Agglomeration, PolicyTrace)> {
    beam_agglomerate_with_latent(
        analysis,
        graph,
        seeds,
        options,
        model,
        value_weight,
        context,
        beam,
        None,
    )
}

/// Imagined embeddings replace GNN re-encoding inside the beam. Symbolic
/// contractions and legal action features remain exact; every committed move
/// starts a fresh round from the true encoder. No learned output is a hard cut.
#[allow(clippy::too_many_arguments)]
pub fn beam_agglomerate_with_latent(
    analysis: &HtnAnalysis,
    graph: &InteractionGraph,
    seeds: &[MicroPartition],
    options: &AgglomerationOptions,
    model: &MergeModel,
    value_weight: f64,
    context: &MasterOptions,
    beam: &BeamOptions,
    latent: Option<&super::latent::LatentModel>,
) -> Result<(Agglomeration, PolicyTrace)> {
    let search_start = std::time::Instant::now();
    model.validate()?;
    if let Some(latent) = latent {
        latent.validate(model)?;
        if beam.depth > 4 {
            bail!("latent lookahead depth must be 1..4");
        }
    }
    if !value_weight.is_finite()
        || value_weight < 0.0
        || !(1..=128).contains(&beam.width)
        || !(1..=16).contains(&beam.depth)
        || !(1..=128).contains(&beam.branching)
        || beam.max_evaluations == 0
    {
        bail!("invalid beam search options");
    }
    let mut stats = BeamStatistics {
        width: beam.width,
        depth: beam.depth,
        branching: beam.branching,
        latent_model_sha256: latent
            .map(|l| serde_json::to_vec(l).map(|b| sha256(&b)))
            .transpose()?,
        ..Default::default()
    };
    let start = std::time::Instant::now();
    let mut state = agglomeration::observe(analysis, graph, NetworkId::Initial, seeds, options)?;
    stats.quotient_time_us += start.elapsed().as_micros();
    let mut history = state.merges.clone();
    let mut decisions = Vec::new();
    // Graph, weights and budgets are immutable for this call. Ordered IDs and
    // memberships determine the full observation, including candidate positions.
    let mut predictions = PredictionCache::default();
    let stop_reason = loop {
        if state.clusters.len() <= options.target_clusters {
            break StopReason::TargetReached;
        }
        if stats.evaluations >= beam.max_evaluations {
            stats.truncated = true;
            break StopReason::SearchLimit;
        }
        stats.rounds += 1;
        let mut frontier = vec![Branch {
            state: state.clone(),
            first: None,
            first_merge: None,
            log_probability: 0.0,
            score: 0.0,
            terminal: false,
            imagined: None,
        }];
        for _ in 0..beam.depth {
            let mut next = Vec::new();
            for branch in frontier {
                if branch.terminal {
                    next.push(branch);
                    continue;
                }
                if stats.evaluations >= beam.max_evaluations {
                    stats.truncated = true;
                    // A branch with a recorded first move remains a valid leaf.
                    if branch.first.is_some() {
                        next.push(branch);
                    }
                    continue;
                }
                let start = std::time::Instant::now();
                let (mut observation, candidates) =
                    Observation::build(analysis, graph, &branch.state, options)?;
                observation.set_search_context(context);
                let feature_time_us = start.elapsed().as_micros();
                let start = std::time::Instant::now();
                let signature: PredictionKey = branch
                    .state
                    .clusters
                    .iter()
                    .map(|c| {
                        let mut nodes = c.nodes.clone();
                        nodes.sort();
                        (c.id, nodes)
                    })
                    .collect();
                let mut hidden = None;
                let prediction = if let Some(latent) = latent {
                    let encoding = if let Some(transition) = &branch.imagined {
                        stats.latent_transitions += 1;
                        latent.transition(
                            &transition.observation,
                            &transition.hidden,
                            transition.action,
                            &observation,
                        )?
                    } else {
                        stats.exact_encodings += 1;
                        model.encode(&observation)?
                    };
                    let prediction = model.predict_encoded(&observation, &encoding)?;
                    hidden = Some(Arc::new(encoding));
                    stats.evaluations += 1;
                    prediction
                } else if let Some(prediction) = predictions.get(&signature) {
                    stats.prediction_cache_hits += 1;
                    prediction
                } else {
                    let prediction = model.predict(&observation)?;
                    predictions.insert(signature, &prediction);
                    stats.evaluations += 1;
                    prediction
                };
                let inference_time_us = start.elapsed().as_micros();
                stats.feature_time_us += feature_time_us;
                stats.inference_time_us += inference_time_us;
                let max = prediction
                    .logits
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max);
                let log_z = max
                    + prediction
                        .logits
                        .iter()
                        .map(|l| (l - max).exp())
                        .sum::<f64>()
                        .ln();
                let stop = observation.actions.len() - 1;
                let mut order: Vec<_> = (0..stop).collect();
                order.sort_by(|&a, &b| {
                    let score =
                        |i: usize| prediction.logits[i] - value_weight * prediction.log_effort[i];
                    score(b).total_cmp(&score(a)).then(a.cmp(&b))
                });
                order.truncate(beam.branching);
                // Keep STOP even when it falls outside the merge shortlist.
                order.push(stop);
                let shared_observation = latent.map(|_| Arc::new(observation.clone()));
                for action in order {
                    let terminal = action == stop;
                    let start = std::time::Instant::now();
                    let proposed = if terminal {
                        branch.state.clone()
                    } else {
                        let clusters = agglomeration::merged_clusters(
                            &branch.state.clusters,
                            &candidates[action],
                        )?;
                        agglomeration::observe(
                            analysis,
                            graph,
                            NetworkId::Initial,
                            &clusters,
                            options,
                        )?
                    };
                    stats.quotient_time_us += start.elapsed().as_micros();
                    let log_probability =
                        branch.log_probability + prediction.logits[action] - log_z;
                    let first = branch.first.clone().unwrap_or_else(|| {
                        Arc::new(PolicyDecision {
                            observation: observation.clone(),
                            prediction: prediction.clone(),
                            action,
                            feature_time_us,
                            inference_time_us,
                        })
                    });
                    let first_merge = if branch.first.is_none() {
                        candidates.get(action).cloned()
                    } else {
                        branch.first_merge.clone()
                    };
                    let reached_target = proposed.clusters.len() <= options.target_clusters;
                    next.push(Branch {
                        state: proposed,
                        first: Some(first),
                        first_merge,
                        log_probability,
                        score: log_probability - value_weight * prediction.log_effort[action],
                        terminal: terminal || reached_target,
                        imagined: if terminal {
                            None
                        } else {
                            shared_observation
                                .as_ref()
                                .map(|before| ImaginedTransition {
                                    observation: before.clone(),
                                    hidden: hidden.as_ref().unwrap().clone(),
                                    action,
                                })
                        },
                    });
                    stats.generated += 1;
                }
            }
            let mut unique = BTreeMap::new();
            let mut imagined_paths = Vec::new();
            for branch in next {
                // Different imagined histories can give the same symbolic
                // quotient different future embeddings; prefix dominance no
                // longer applies, so retain them as separate bounded paths.
                if latent.is_some() {
                    imagined_paths.push(branch);
                    continue;
                }
                // Terminal and unexpanded states have distinct future semantics.
                let k = (key(&branch.state), branch.terminal);
                if let Some(old) = unique.get_mut(&k) {
                    stats.deduplicated += 1;
                    let old: &mut Branch = old;
                    if dominates(&branch, old) {
                        *old = branch;
                    }
                } else {
                    unique.insert(k, branch);
                }
            }
            frontier = if latent.is_some() {
                imagined_paths
            } else {
                unique.into_values().collect()
            };
            frontier.sort_by(|a, b| {
                b.score
                    .total_cmp(&a.score)
                    .then_with(|| {
                        a.first
                            .as_ref()
                            .map(|d| d.action)
                            .cmp(&b.first.as_ref().map(|d| d.action))
                    })
                    .then_with(|| key(&a.state).cmp(&key(&b.state)))
            });
            frontier.truncate(beam.width);
            if frontier.iter().all(|b| b.terminal) {
                break;
            }
        }
        let Some(best) = frontier.into_iter().next() else {
            break StopReason::SearchLimit;
        };
        let decision = best
            .first
            .ok_or_else(|| anyhow::anyhow!("beam has no evaluated first action"))?;
        let stop = decision.observation.actions[decision.action].pair.is_none();
        decisions.push(Arc::unwrap_or_clone(decision));
        if stop {
            break StopReason::PolicyStop;
        }
        let candidate = best.first_merge.unwrap();
        let start = std::time::Instant::now();
        let clusters = agglomeration::merged_clusters(&state.clusters, &candidate)?;
        history.push(MergeStep {
            children: vec![candidate.left, candidate.right],
            parent: clusters.iter().max_by_key(|c| c.id).unwrap().clone(),
            reason: MergeReason::Policy,
            score: candidate.score,
        });
        state = agglomeration::observe(analysis, graph, NetworkId::Initial, &clusters, options)?;
        stats.quotient_time_us += start.elapsed().as_micros();
    };
    state.seeds = seeds.to_vec();
    state.merges = history;
    stats.elapsed_us = search_start.elapsed().as_micros();
    Ok((
        state,
        PolicyTrace {
            feature_version: super::features::FEATURE_VERSION,
            model_sha256: sha256(&serde_json::to_vec(model)?),
            value_weight,
            decisions,
            stop_reason,
            beam: Some(stats),
        },
    ))
}
