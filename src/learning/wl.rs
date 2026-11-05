//! Fixed, directed, edge-labelled 1-WL subtree counts over extended HILGs.
//! Two views retain schema colors and structural roles respectively. Hashing is
//! applied only to the final count vocabulary; refinement uses full SHA-256 IDs.
use super::{
    autodiff::{Tape, Var},
    features::*,
    hilg::InstanceGraph,
    model::{MergeModel, ModelConfig},
};
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
pub const ITERATIONS: usize = 3;
pub const BINS: usize = 128;
pub const HISTOGRAM: usize = (ITERATIONS + 1) * 2 * BINS;
pub const INPUT: usize = 2 * HISTOGRAM + NODE_FEATURES;
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WlFeatures {
    pub schema_version: u32,
    /// Local member-rooted counts followed by whole-instance context counts.
    pub clusters: Vec<Vec<f64>>,
}
impl WlFeatures {
    pub fn validate(&self, clusters: usize) -> Result<()> {
        if self.schema_version != 1
            || self.clusters.len() != clusters
            || self
                .clusters
                .iter()
                .any(|r| r.len() != 2 * HISTOGRAM || r.iter().any(|x| !x.is_finite() || *x < 0.0))
        {
            bail!("invalid Weisfeiler–Leman features");
        }
        Ok(())
    }
    pub fn build(graph: &InstanceGraph, clusters: usize) -> Result<Self> {
        if graph.colors.len() != graph.nodes.len() {
            bail!("WL extraction requires exact HILG colors");
        }
        let n = graph.nodes.len();
        let mut members = vec![Vec::new(); n];
        for (id, owner) in members.iter_mut().take(clusters).enumerate() {
            owner.push(id);
        }
        for e in &graph.edges {
            if e.channel == 2 {
                members[e.source].push(e.target);
            }
        }
        for owner in &mut members {
            owner.sort_unstable();
            owner.dedup();
        }
        let mut adjacency = vec![Vec::new(); n];
        for e in &graph.edges {
            // Reverse directions already have distinct labels. Multiplicity is
            // retained, including repeated arguments; weight is a count feature
            // in the existing numeric quotient features, not a WL color.
            adjacency[e.target].push((e.channel, e.position, e.source));
        }
        let mut output = vec![vec![0.0; 2 * HISTOGRAM]; clusters];
        for view in 0..2 {
            let mut colors: Vec<[u8; 32]> = graph
                .nodes
                .iter()
                .enumerate()
                .map(|(i, row)| {
                    let mut h = Sha256::new();
                    h.update(b"hilg-wl-v1");
                    h.update([view as u8]);
                    if view == 0 {
                        h.update(graph.colors[i].as_bytes());
                    }
                    // Numeric cluster summaries stay continuous inputs, rather than
                    // unstable colors. Object names are absent from both views.
                    for x in &row[12..24] {
                        h.update(x.to_bits().to_le_bytes());
                    }
                    h.finalize().into()
                })
                .collect();
            for iteration in 0..=ITERATIONS {
                let base = (iteration * 2 + view) * BINS;
                let mut global = vec![0.0; BINS];
                for (node, color) in colors.iter().enumerate() {
                    let bin = u64::from_le_bytes(color[..8].try_into().unwrap()) as usize % BINS;
                    if node >= clusters {
                        global[bin] += 1.0;
                    }
                    for &owner in &members[node] {
                        output[owner][base + bin] += 1.0;
                    }
                }
                for row in &mut output {
                    row[HISTOGRAM + base..HISTOGRAM + base + BINS].copy_from_slice(&global);
                }
                if iteration == ITERATIONS {
                    break;
                }
                colors = colors
                    .iter()
                    .enumerate()
                    .map(|(node, color)| {
                        let mut messages: Vec<_> = adjacency[node]
                            .iter()
                            .map(|&(channel, position, source)| (channel, position, colors[source]))
                            .collect();
                        messages.sort_unstable();
                        let mut h = Sha256::new();
                        h.update(b"hilg-wl-refine-v1");
                        h.update(color);
                        h.update((messages.len() as u64).to_le_bytes());
                        for (channel, position, neighbor) in messages {
                            h.update((channel as u64).to_le_bytes());
                            h.update((position as u64).to_le_bytes());
                            h.update(neighbor);
                        }
                        h.finalize().into()
                    })
                    .collect();
            }
        }
        for row in &mut output {
            for x in row {
                *x = f64::ln_1p(*x);
            }
        }
        let features = Self {
            schema_version: 1,
            clusters: output,
        };
        features.validate(clusters)?;
        Ok(features)
    }
}
pub struct WlLayout {
    pub layers: Vec<(usize, usize, usize)>,
    pub decoder: usize,
    pub policy: usize,
    pub value: usize,
    pub width: usize,
    pub total: usize,
}
impl WlLayout {
    pub fn new(config: ModelConfig) -> Result<Self> {
        let h = config.hidden;
        if !(1..=32).contains(&h) || !(1..=3).contains(&config.layers) {
            bail!("WL encoder requires hidden=1..32, layers=1..3");
        }
        let mut offset = 0;
        let mut layers = Vec::new();
        for layer in 0..config.layers {
            let width = if layer == 0 { INPUT } else { h };
            let matrix = offset;
            offset += h * width;
            let bias = offset;
            offset += h;
            layers.push((matrix, bias, width));
        }
        let width = 4 * h + PAIR_FEATURES + GLOBAL_FEATURES + CONTEXT_FEATURES + 1;
        let decoder = offset;
        offset += h * (width + 1);
        let policy = offset;
        offset += h + 1;
        let value = offset;
        offset += h + 1;
        Ok(Self {
            layers,
            decoder,
            policy,
            value,
            width,
            total: offset,
        })
    }
}
pub fn encode(
    model: &MergeModel,
    observation: &Observation,
    training: bool,
) -> Result<(Tape, Vec<Vec<Var>>)> {
    let graph = observation
        .hilg
        .as_ref()
        .context("WL checkpoint requires HILG features; re-export legacy training data")?;
    let features = graph
        .wl
        .as_ref()
        .context("WL checkpoint requires subtree counts")?;
    features.validate(observation.nodes.len())?;
    let layout = WlLayout::new(model.config)?;
    let h = model.config.hidden;
    let mut tape = if training {
        Tape::new()
    } else {
        Tape::inference()
    };
    for &parameter in &model.parameters {
        tape.constant(parameter);
    }
    let mut hidden = Vec::new();
    for (node, wl) in observation.nodes.iter().zip(&features.clusters) {
        let input: Vec<_> = wl.iter().chain(&node.features).copied().collect();
        let (matrix, bias, width) = layout.layers[0];
        let mut row = Vec::new();
        for d in 0..h {
            let mut sum = bias + d;
            for (k, &value) in input.iter().enumerate().filter(|(_, v)| **v != 0.0) {
                let term = tape.scale(matrix + d * width + k, value);
                sum = tape.add(sum, term);
            }
            row.push(tape.tanh(sum));
        }
        for &(matrix, bias, width) in layout.layers.iter().skip(1) {
            let mut output = Vec::new();
            for d in 0..h {
                let mut sum = bias + d;
                for (k, &v) in row.iter().enumerate() {
                    let term = tape.mul(matrix + d * width + k, v);
                    sum = tape.add(sum, term);
                }
                output.push(tape.tanh(sum));
            }
            row = output;
        }
        hidden.push(row);
    }
    Ok((tape, hidden))
}
pub fn decode(layout: &WlLayout, h: usize, tape: &mut Tape, encoded: &[Var]) -> (Var, Var) {
    let mut features = Vec::new();
    for d in 0..h {
        let offset = layout.decoder + d * (layout.width + 1);
        let mut sum = offset + layout.width;
        for (k, &feature) in encoded.iter().enumerate() {
            let term = tape.mul(offset + k, feature);
            sum = tape.add(sum, term);
        }
        features.push(tape.tanh(sum));
    }
    let mut policy = layout.policy + h;
    let mut value = layout.value + h;
    for (k, &feature) in features.iter().enumerate() {
        let term = tape.mul(layout.policy + k, feature);
        policy = tape.add(policy, term);
        let term = tape.mul(layout.value + k, feature);
        value = tape.add(value, term);
    }
    (policy, tape.softplus(value))
}

/// Allocation-light native inference; the autodiff path remains the gradient
/// reference. No parameter-sized tape is necessary for a fixed WL projection.
pub fn encode_values(model: &MergeModel, observation: &Observation) -> Result<Vec<Vec<f64>>> {
    let features = observation
        .hilg
        .as_ref()
        .and_then(|g| g.wl.as_ref())
        .context("WL checkpoint requires HILG subtree counts")?;
    features.validate(observation.nodes.len())?;
    let layout = WlLayout::new(model.config)?;
    let h = model.config.hidden;
    let p = &model.parameters;
    let mut hidden = Vec::new();
    for (node, features) in observation.nodes.iter().zip(&features.clusters) {
        let input: Vec<_> = features
            .iter()
            .chain(&node.features)
            .enumerate()
            .filter_map(|(i, &x)| (x != 0.0).then_some((i, x)))
            .collect();
        let (matrix, bias, width) = layout.layers[0];
        let mut row: Vec<_> = (0..h)
            .map(|d| {
                input
                    .iter()
                    .fold(p[bias + d], |sum, &(k, x)| {
                        sum + p[matrix + d * width + k] * x
                    })
                    .tanh()
            })
            .collect();
        for &(matrix, bias, width) in layout.layers.iter().skip(1) {
            row = (0..h)
                .map(|d| {
                    row.iter()
                        .enumerate()
                        .fold(p[bias + d], |sum, (k, &x)| {
                            sum + p[matrix + d * width + k] * x
                        })
                        .tanh()
                })
                .collect();
        }
        hidden.push(row);
    }
    Ok(hidden)
}
pub fn predict_values(
    model: &MergeModel,
    observation: &Observation,
    hidden: &[Vec<f64>],
) -> Result<super::model::Prediction> {
    let layout = WlLayout::new(model.config)?;
    let h = model.config.hidden;
    let p = &model.parameters;
    let pool: Vec<_> = (0..h)
        .map(|d| hidden.iter().map(|r| r[d]).sum::<f64>() / hidden.len().max(1) as f64)
        .collect();
    let mut logits = Vec::new();
    let mut values = Vec::new();
    for action in &observation.actions {
        let mut encoded = Vec::with_capacity(layout.width);
        if let Some((a, b)) = action.pair {
            encoded.extend((0..h).map(|d| (hidden[a][d] + hidden[b][d]) * 0.5));
            encoded.extend((0..h).map(|d| (hidden[a][d] - hidden[b][d]).abs()));
            encoded.extend((0..h).map(|d| hidden[a][d] * hidden[b][d]));
        } else {
            encoded.extend(&pool);
            encoded.extend(std::iter::repeat_n(0.0, h));
            encoded.extend(pool.iter().map(|x| x * x));
        }
        encoded.extend(&pool);
        encoded.extend(&action.features);
        encoded.extend(&observation.global);
        encoded.extend(&observation.context);
        encoded.push(f64::from(action.pair.is_none()));
        let features: Vec<_> = (0..h)
            .map(|d| {
                let offset = layout.decoder + d * (layout.width + 1);
                encoded
                    .iter()
                    .enumerate()
                    .fold(p[offset + layout.width], |sum, (k, &x)| {
                        sum + p[offset + k] * x
                    })
                    .tanh()
            })
            .collect();
        logits.push(
            features
                .iter()
                .enumerate()
                .fold(p[layout.policy + h], |sum, (k, &x)| {
                    sum + p[layout.policy + k] * x
                }),
        );
        let value = features
            .iter()
            .enumerate()
            .fold(p[layout.value + h], |sum, (k, &x)| {
                sum + p[layout.value + k] * x
            });
        values.push(value.max(0.0) + (-value.abs()).exp().ln_1p());
    }
    if logits.iter().chain(&values).any(|x| !x.is_finite()) {
        bail!("nonfinite WL prediction");
    }
    Ok(super::model::Prediction {
        logits,
        log_effort: values,
    })
}
