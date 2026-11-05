//! Learned node transitions with a frozen merge encoder; predictions are heuristic.
use super::{
    autodiff::Tape,
    features::*,
    model::{MergeModel, random},
};
use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatentModel {
    pub schema_version: u32,
    pub hidden: usize,
    pub encoder_sha256: String,
    pub parameters: Vec<f64>,
}
#[derive(Debug, Clone)]
pub struct TransitionExample {
    pub inputs: Vec<Vec<f64>>,
    pub target: Vec<Vec<f64>>,
}
impl LatentModel {
    fn width(&self) -> usize {
        3 * self.hidden + PAIR_FEATURES + NODE_FEATURES + CONTEXT_FEATURES + 1
    }
    fn count(&self) -> usize {
        self.hidden * (self.width() + 1) + self.hidden * (self.hidden + 1)
    }
    pub fn new(encoder: &MergeModel, mut seed: u64) -> Result<Self> {
        encoder.validate()?;
        let mut model = Self {
            schema_version: 1,
            hidden: encoder.config.hidden,
            encoder_sha256: crate::training::sha256(&serde_json::to_vec(encoder)?),
            parameters: Vec::new(),
        };
        model.parameters = (0..model.count())
            .map(|_| {
                ((random(&mut seed) >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0)
                    / (model.width() as f64).sqrt()
            })
            .collect();
        Ok(model)
    }
    pub fn validate(&self, encoder: &MergeModel) -> Result<()> {
        encoder.validate()?;
        if self.schema_version != 1
            || self.hidden != encoder.config.hidden
            || self.encoder_sha256 != crate::training::sha256(&serde_json::to_vec(encoder)?)
            || self.parameters.len() != self.count()
            || self.parameters.iter().any(|v| !v.is_finite())
        {
            bail!("latent transition/encoder checkpoint mismatch");
        }
        Ok(())
    }
    /// IDs align surviving nodes; the single fresh ID denotes the true merged node.
    pub fn inputs(
        &self,
        before: &Observation,
        hidden: &[Vec<f64>],
        action: usize,
        after: &Observation,
    ) -> Result<Vec<Vec<f64>>> {
        before.validate()?;
        after.validate()?;
        if hidden.len() != before.nodes.len()
            || hidden.iter().any(|row| {
                row.len() != self.hidden || row.iter().any(|v| !v.is_finite() || v.abs() > 1.0)
            })
        {
            bail!("invalid transition embeddings");
        }
        let selected = before
            .actions
            .get(action)
            .ok_or_else(|| anyhow::anyhow!("missing transition action"))?;
        let (a, b) = selected
            .pair
            .ok_or_else(|| anyhow::anyhow!("STOP has no latent merge transition"))?;
        let survivors: std::collections::BTreeSet<_> = before
            .nodes
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != a && *i != b)
            .map(|(_, n)| n.id)
            .collect();
        let actual: std::collections::BTreeSet<_> = after.nodes.iter().map(|n| n.id).collect();
        if after.nodes.len() + 1 != before.nodes.len()
            || !survivors.is_subset(&actual)
            || actual
                .difference(&survivors)
                .any(|id| before.nodes.iter().any(|n| n.id == *id))
            || before.context != after.context
        {
            bail!("transition is not a single legal merge shape");
        }
        let pair: Vec<_> = hidden[a]
            .iter()
            .zip(&hidden[b])
            .map(|(a, b)| (a + b) * 0.5)
            .collect();
        let pool: Vec<_> = (0..self.hidden)
            .map(|d| hidden.iter().map(|r| r[d]).sum::<f64>() / hidden.len() as f64)
            .collect();
        Ok(after
            .nodes
            .iter()
            .map(|node| {
                let source = before.nodes.iter().position(|n| n.id == node.id);
                let mut row = source.map_or_else(|| pair.clone(), |i| hidden[i].clone());
                row.extend(&pair);
                row.extend(&pool);
                row.extend(&selected.features);
                row.extend(&node.features);
                row.extend(&before.context);
                row.push(f64::from(source.is_none()));
                row
            })
            .collect())
    }
    fn forward_row(&self, row: &[f64], training: bool) -> Result<(Tape, Vec<usize>)> {
        let h = self.hidden;
        if !(1..=32).contains(&h)
            || self.parameters.len() != self.count()
            || self.parameters.iter().any(|v| !v.is_finite())
            || row.len() != self.width()
            || row.iter().any(|v| !v.is_finite())
        {
            bail!("invalid latent transition input/model");
        }
        let mut tape = if training {
            Tape::new()
        } else {
            Tape::inference()
        };
        for &p in &self.parameters {
            tape.constant(p);
        }
        let mut intermediate = Vec::new();
        for d in 0..h {
            let offset = d * (self.width() + 1);
            let mut sum = offset + self.width();
            for (k, &x) in row.iter().enumerate() {
                let term = tape.scale(offset + k, x);
                sum = tape.add(sum, term);
            }
            intermediate.push(tape.tanh(sum));
        }
        let output_start = h * (self.width() + 1);
        let mut output = Vec::new();
        for (d, &residual) in row.iter().take(h).enumerate() {
            let offset = output_start + d * (h + 1);
            let base = tape.constant(residual);
            let mut sum = tape.add(offset + h, base);
            for (k, &x) in intermediate.iter().enumerate() {
                let term = tape.mul(offset + k, x);
                sum = tape.add(sum, term);
            }
            output.push(tape.tanh(sum));
        }
        Ok((tape, output))
    }
    pub fn transition(
        &self,
        before: &Observation,
        hidden: &[Vec<f64>],
        action: usize,
        after: &Observation,
    ) -> Result<Vec<Vec<f64>>> {
        self.inputs(before, hidden, action, after)?
            .iter()
            .map(|row| {
                let (tape, out) = self.forward_row(row, false)?;
                let values: Vec<_> = out.iter().map(|&v| tape.value(v)).collect();
                if values.iter().any(|v| !v.is_finite()) {
                    bail!("nonfinite latent prediction");
                }
                Ok(values)
            })
            .collect()
    }
    pub fn loss_gradient(&self, example: &TransitionExample) -> Result<(f64, Vec<f64>)> {
        if example.inputs.is_empty() || example.inputs.len() != example.target.len() {
            bail!("invalid transition target rows");
        }
        let mut loss = 0.0;
        let mut gradient = vec![0.0; self.parameters.len()];
        // One tape per node bounds training memory even for wide quotients.
        for (input, target) in example.inputs.iter().zip(&example.target) {
            if target.len() != self.hidden || target.iter().any(|v| !v.is_finite() || v.abs() > 1.0)
            {
                bail!("invalid transition target");
            }
            let (mut tape, out) = self.forward_row(input, true)?;
            let mut total = tape.constant(0.0);
            for (&v, &target) in out.iter().zip(target) {
                let constant = tape.constant(-target);
                let delta = tape.add(v, constant);
                let square = tape.mul(delta, delta);
                let term = tape.scale(square, 1.0 / (example.inputs.len() * self.hidden) as f64);
                total = tape.add(total, term);
            }
            loss += tape.value(total);
            for (g, d) in gradient
                .iter_mut()
                .zip(tape.backward(total, self.parameters.len()))
            {
                *g += d;
            }
        }
        if !loss.is_finite() || gradient.iter().any(|v| !v.is_finite()) {
            bail!("nonfinite transition loss/gradient");
        }
        Ok((loss, gradient))
    }
    pub fn loss(&self, example: &TransitionExample) -> Result<f64> {
        if example.inputs.is_empty() || example.inputs.len() != example.target.len() {
            bail!("invalid transition target rows");
        }
        let mut error = 0.0;
        for (row, target) in example.inputs.iter().zip(&example.target) {
            if target.len() != self.hidden || target.iter().any(|v| !v.is_finite() || v.abs() > 1.0)
            {
                bail!("invalid transition target");
            }
            let (tape, out) = self.forward_row(row, false)?;
            for (&v, &t) in out.iter().zip(target) {
                error += (tape.value(v) - t).powi(2);
            }
        }
        let loss = error / (example.inputs.len() * self.hidden) as f64;
        if !loss.is_finite() {
            bail!("nonfinite transition loss");
        }
        Ok(loss)
    }
}
