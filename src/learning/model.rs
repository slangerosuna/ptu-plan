//! Versioned relational and Weisfeiler–Leman encoders with policy/value heads.
//! Trainable projections and heads use the same reverse-mode path as inference.
use super::autodiff::{Tape, Var};
use super::features::*;
use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ModelConfig {
    pub hidden: usize,
    pub layers: usize,
}
impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            hidden: 8,
            layers: 2,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeModel {
    pub schema_version: u32,
    pub feature_version: u32,
    pub config: ModelConfig,
    pub parameters: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prediction {
    pub logits: Vec<f64>,
    pub log_effort: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValueTarget {
    pub action: usize,
    pub log_effort: f64,
    pub lower_bound: bool,
}
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Supervision {
    /// Pairs are (preferred, other); unobserved/unsolved ties have no ranking.
    pub preferences: Vec<(usize, usize)>,
    pub values: Vec<ValueTarget>,
}

struct Layout {
    input: usize,
    bias: usize,
    layers: Vec<(usize, usize, usize)>,
    policy: usize,
    value: usize,
    width: usize,
    total: usize,
}
impl Layout {
    fn new(config: ModelConfig) -> Result<Self> {
        let h = config.hidden;
        if !(1..=32).contains(&h) || !(1..=3).contains(&config.layers) {
            bail!("require hidden=1..32 and layers=1..3");
        }
        let input = 0;
        let bias = h * NODE_FEATURES;
        let mut offset = bias + h;
        let mut layers = Vec::new();
        for _ in 0..config.layers {
            let own = offset;
            offset += h * h;
            let relation = offset;
            offset += RELATIONS * h * h;
            let b = offset;
            offset += h;
            layers.push((own, relation, b));
        }
        let width = 4 * h + PAIR_FEATURES + GLOBAL_FEATURES + CONTEXT_FEATURES + 1;
        let policy = offset;
        offset += width + 1;
        let value = offset;
        offset += width + 1;
        Ok(Self {
            input,
            bias,
            layers,
            policy,
            value,
            width,
            total: offset,
        })
    }
}

pub(crate) fn random(seed: &mut u64) -> u64 {
    *seed = seed.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *seed;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

impl MergeModel {
    pub fn new(config: ModelConfig, mut seed: u64) -> Result<Self> {
        let layout = Layout::new(config)?;
        let scale = 1.0 / (config.hidden as f64).sqrt();
        let parameters = (0..layout.total)
            .map(|_| ((random(&mut seed) >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) * scale)
            .collect();
        Ok(Self {
            schema_version: 1,
            feature_version: FEATURE_VERSION,
            config,
            parameters,
        })
    }
    /// Version two: Weisfeiler–Leman instance features, nonlinear shared decoder.
    pub fn new_wl(config: ModelConfig, mut seed: u64) -> Result<Self> {
        let layout = super::wl::WlLayout::new(config)?;
        let scale = 1.0 / (config.hidden as f64).sqrt();
        Ok(Self {
            schema_version: 2,
            feature_version: FEATURE_VERSION,
            config,
            parameters: (0..layout.total)
                .map(|index| {
                    let scale = if index < config.hidden * super::wl::INPUT {
                        1.0 / (super::wl::INPUT as f64).sqrt()
                    } else {
                        scale
                    };
                    ((random(&mut seed) >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) * scale
                })
                .collect(),
        })
    }
    pub fn validate(&self) -> Result<()> {
        let total = match self.schema_version {
            1 => Layout::new(self.config)?.total,
            2 => super::wl::WlLayout::new(self.config)?.total,
            _ => bail!("unsupported merge model architecture"),
        };
        if self.feature_version != FEATURE_VERSION
            || self.parameters.len() != total
            || self.parameters.iter().any(|p| !p.is_finite())
        {
            bail!("incompatible or nonfinite model checkpoint");
        }
        Ok(())
    }
    fn encode_tape(
        &self,
        observation: &Observation,
        training: bool,
    ) -> Result<(Tape, Vec<Vec<Var>>)> {
        self.validate()?;
        observation.validate()?;
        if self.schema_version == 2 {
            return super::wl::encode(self, observation, training);
        }
        let layout = Layout::new(self.config)?;
        let h = self.config.hidden;
        let mut tape = if training {
            Tape::new()
        } else {
            Tape::inference()
        };
        for &parameter in &self.parameters {
            tape.constant(parameter);
        }
        let mut hidden = Vec::new();
        for node in &observation.nodes {
            let mut row = Vec::new();
            for d in 0..h {
                let mut sum = layout.bias + d;
                for (k, &feature) in node.features.iter().enumerate() {
                    let term = tape.scale(layout.input + d * NODE_FEATURES + k, feature);
                    sum = tape.add(sum, term);
                }
                row.push(tape.tanh(sum));
            }
            hidden.push(row);
        }
        let mut degree = vec![vec![0.0; RELATIONS]; hidden.len()];
        for edge in &observation.relations {
            degree[edge.target][edge.channel] += edge.weight;
        }
        for &(own, relations, bias) in &layout.layers {
            let mut messages = Vec::new();
            for row in &hidden {
                let mut next = Vec::new();
                for d in 0..h {
                    let mut sum = bias + d;
                    for (k, &value) in row.iter().enumerate() {
                        let term = tape.mul(own + d * h + k, value);
                        sum = tape.add(sum, term);
                    }
                    next.push(sum);
                }
                messages.push(next);
            }
            for edge in &observation.relations {
                let normalization = edge.weight / degree[edge.target][edge.channel];
                for (d, message) in messages[edge.target].iter_mut().enumerate() {
                    let mut sum = tape.constant(0.0);
                    for (k, &source) in hidden[edge.source].iter().enumerate() {
                        let term = tape.mul(relations + edge.channel * h * h + d * h + k, source);
                        sum = tape.add(sum, term);
                    }
                    let term = tape.scale(sum, normalization);
                    *message = tape.add(*message, term);
                }
            }
            hidden = messages
                .into_iter()
                .map(|row| row.into_iter().map(|v| tape.tanh(v)).collect())
                .collect();
        }
        Ok((tape, hidden))
    }

    /// Frozen-encoder targets and exact re-encoding for latent lookahead.
    pub fn encode(&self, observation: &Observation) -> Result<Vec<Vec<f64>>> {
        if self.schema_version == 2 {
            self.validate()?;
            observation.validate()?;
            let hidden = super::wl::encode_values(self, observation)?;
            if hidden.iter().flatten().any(|x| !x.is_finite()) {
                bail!("nonfinite WL encoding");
            }
            return Ok(hidden);
        }
        let (tape, hidden) = self.encode_tape(observation, false)?;
        let encoded: Vec<Vec<f64>> = hidden
            .iter()
            .map(|row| row.iter().map(|&v| tape.value(v)).collect())
            .collect();
        if encoded.iter().flatten().any(|v| !v.is_finite()) {
            bail!("nonfinite graph encoding");
        }
        Ok(encoded)
    }

    fn decode_tape(
        &self,
        observation: &Observation,
        mut tape: Tape,
        hidden: Vec<Vec<Var>>,
    ) -> Result<(Tape, Vec<Var>, Vec<Var>)> {
        let layout = Layout::new(self.config)?;
        let wl = if self.schema_version == 2 {
            Some(super::wl::WlLayout::new(self.config)?)
        } else {
            None
        };
        let h = self.config.hidden;
        let mut pool = vec![tape.constant(0.0); h];
        for row in &hidden {
            for d in 0..h {
                pool[d] = tape.add(pool[d], row[d]);
            }
        }
        for p in &mut pool {
            *p = tape.scale(*p, 1.0 / (hidden.len().max(1) as f64));
        }
        let mut logits = Vec::new();
        let mut values = Vec::new();
        for action in &observation.actions {
            let mut encoded = Vec::new();
            if let Some((a, b)) = action.pair {
                for d in 0..h {
                    let sum = tape.add(hidden[a][d], hidden[b][d]);
                    encoded.push(tape.scale(sum, 0.5));
                }
                for d in 0..h {
                    let negative = tape.scale(hidden[b][d], -1.0);
                    let delta = tape.add(hidden[a][d], negative);
                    encoded.push(tape.abs(delta));
                }
                for d in 0..h {
                    encoded.push(tape.mul(hidden[a][d], hidden[b][d]));
                }
            } else {
                encoded.extend(&pool);
                for _ in 0..h {
                    encoded.push(tape.constant(0.0));
                }
                for &p in &pool {
                    encoded.push(tape.mul(p, p));
                }
            }
            encoded.extend(&pool);
            for &feature in action
                .features
                .iter()
                .chain(&observation.global)
                .chain(&observation.context)
            {
                encoded.push(tape.constant(feature));
            }
            encoded.push(tape.constant(f64::from(action.pair.is_none())));
            if let Some(layout) = &wl {
                let (policy, value) = super::wl::decode(layout, h, &mut tape, &encoded);
                logits.push(policy);
                values.push(value);
                continue;
            }
            let mut policy = layout.policy + layout.width;
            let mut value = layout.value + layout.width;
            for (k, &feature) in encoded.iter().enumerate() {
                let term = tape.mul(layout.policy + k, feature);
                policy = tape.add(policy, term);
                let term = tape.mul(layout.value + k, feature);
                value = tape.add(value, term);
            }
            logits.push(policy);
            values.push(tape.softplus(value));
        }
        Ok((tape, logits, values))
    }
    fn forward(
        &self,
        observation: &Observation,
        training: bool,
    ) -> Result<(Tape, Vec<Var>, Vec<Var>)> {
        let (tape, hidden) = self.encode_tape(observation, training)?;
        self.decode_tape(observation, tape, hidden)
    }

    /// Decode imagined node embeddings using exact legal actions and features.
    pub fn predict_encoded(
        &self,
        observation: &Observation,
        hidden: &[Vec<f64>],
    ) -> Result<Prediction> {
        self.validate()?;
        observation.validate()?;
        if hidden.len() != observation.nodes.len()
            || hidden.iter().any(|row| {
                row.len() != self.config.hidden
                    || row.iter().any(|x| !x.is_finite() || x.abs() > 1.0)
            })
        {
            bail!("invalid latent node embeddings");
        }
        if self.schema_version == 2 {
            return super::wl::predict_values(self, observation, hidden);
        }
        let mut tape = Tape::inference();
        for &p in &self.parameters {
            tape.constant(p);
        }
        let hidden = hidden
            .iter()
            .map(|row| row.iter().map(|&v| tape.constant(v)).collect())
            .collect();
        let (tape, logits, values) = self.decode_tape(observation, tape, hidden)?;
        Self::prediction(&tape, &logits, &values)
    }

    pub fn predict(&self, observation: &Observation) -> Result<Prediction> {
        if self.schema_version == 2 {
            let hidden = self.encode(observation)?;
            return super::wl::predict_values(self, observation, &hidden);
        }
        let (tape, logits, values) = self.forward(observation, false)?;
        Self::prediction(&tape, &logits, &values)
    }
    fn prediction(tape: &Tape, logits: &[Var], values: &[Var]) -> Result<Prediction> {
        let prediction = Prediction {
            logits: logits.iter().map(|&v| tape.value(v)).collect(),
            log_effort: values.iter().map(|&v| tape.value(v)).collect(),
        };
        if prediction
            .logits
            .iter()
            .chain(&prediction.log_effort)
            .any(|v| !v.is_finite())
        {
            bail!("nonfinite model prediction");
        }
        Ok(prediction)
    }
    pub fn choose(
        &self,
        observation: &Observation,
        value_weight: f64,
    ) -> Result<(usize, Prediction)> {
        if !value_weight.is_finite() || value_weight < 0.0 {
            bail!("invalid value weight");
        }
        let prediction = self.predict(observation)?;
        let index = (0..prediction.logits.len())
            .max_by(|&a, &b| {
                let left = prediction.logits[a] - value_weight * prediction.log_effort[a];
                let right = prediction.logits[b] - value_weight * prediction.log_effort[b];
                left.total_cmp(&right).then_with(|| b.cmp(&a))
            })
            .unwrap();
        Ok((index, prediction))
    }
    pub fn loss_gradient(
        &self,
        observation: &Observation,
        labels: &Supervision,
        value_weight: f64,
    ) -> Result<(f64, Vec<f64>)> {
        let (mut tape, loss) = self.loss_tape(observation, labels, value_weight, true)?;
        let value = tape.value(loss);
        let gradients = tape.backward(loss, self.parameters.len());
        if gradients.iter().any(|v| !v.is_finite()) {
            bail!("nonfinite model gradient");
        }
        Ok((value, gradients))
    }
    pub fn loss(
        &self,
        observation: &Observation,
        labels: &Supervision,
        value_weight: f64,
    ) -> Result<f64> {
        let (tape, loss) = self.loss_tape(observation, labels, value_weight, false)?;
        Ok(tape.value(loss))
    }
    fn loss_tape(
        &self,
        observation: &Observation,
        labels: &Supervision,
        value_weight: f64,
        training: bool,
    ) -> Result<(Tape, Var)> {
        if !value_weight.is_finite() || value_weight < 0.0 {
            bail!("invalid value loss weight");
        }
        let (mut tape, logits, values) = self.forward(observation, training)?;
        let mut loss = tape.constant(0.0);
        for &(better, worse) in &labels.preferences {
            if better >= logits.len() || worse >= logits.len() || better == worse {
                bail!("invalid ranking target");
            }
            let negative = tape.scale(logits[better], -1.0);
            let delta = tape.add(logits[worse], negative);
            let term = tape.softplus(delta);
            let term = tape.scale(term, 1.0 / labels.preferences.len() as f64);
            loss = tape.add(loss, term);
        }
        for target in &labels.values {
            if target.action >= values.len()
                || !target.log_effort.is_finite()
                || target.log_effort < 0.0
            {
                bail!("invalid value target");
            }
            let constant = tape.constant(-target.log_effort);
            let residual = tape.add(values[target.action], constant);
            if target.lower_bound && tape.value(residual) >= 0.0 {
                continue;
            }
            let term = tape.huber(residual);
            let term = tape.scale(term, value_weight / labels.values.len() as f64);
            loss = tape.add(loss, term);
        }
        let value = tape.value(loss);
        if !value.is_finite() {
            bail!("nonfinite model loss");
        }
        Ok((tape, loss))
    }
}
