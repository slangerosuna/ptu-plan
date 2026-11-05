//! Shared atom classifier for true/false/unknown outgoing interface proposals.
use super::{
    autodiff::Tape,
    model::random,
    rollout::{RolloutAuditor, RolloutRecord},
    training::{Adam, TrainOptions},
};
use crate::{
    master::{InterfaceGuide, InterfaceQuery, MasterStatus},
    solver::{
        BoundaryState, Condition, GroundAtom, GroundNetwork, GroundProblem, SolverError, State,
    },
    store::{DatasetStore, Entry, Metadata, RecordKind, Split, group_split},
};
use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const INTERFACE_FEATURE_VERSION: u32 = 1;
pub const ATOM_FEATURES: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FluentStatus {
    Unknown,
    False,
    True,
}
impl FluentStatus {
    fn index(self) -> usize {
        match self {
            Self::Unknown => 0,
            Self::False => 1,
            Self::True => 2,
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtomInput {
    pub atom: GroundAtom,
    pub features: Vec<f64>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterfaceInput {
    pub feature_version: u32,
    pub atoms: Vec<AtomInput>,
}

fn condition_atoms(c: &Condition, atoms: &mut State) {
    match c {
        Condition::Atom(a) => {
            atoms.insert(a.clone());
        }
        Condition::Not(c) => condition_atoms(c, atoms),
        Condition::And(cs) | Condition::Or(cs) => {
            for c in cs {
                condition_atoms(c, atoms);
            }
        }
        _ => {}
    }
}
#[derive(Default)]
struct Footprint {
    reads: State,
    add: State,
    delete: State,
    methods: usize,
}
fn footprint(problem: &GroundProblem, network: &GroundNetwork) -> Footprint {
    let mut result = Footprint::default();
    let mut pending = network.tasks.clone();
    let mut seen = BTreeSet::new();
    while let Some(task) = pending.pop() {
        if !seen.insert(task.clone()) {
            continue;
        }
        if let Some(action) = problem.actions.get(&task) {
            condition_atoms(&action.precondition, &mut result.reads);
            result.add.extend(action.add.iter().cloned());
            result.delete.extend(action.delete.iter().cloned());
        }
        if let Some(methods) = problem.methods_by_task.get(&task) {
            result.methods += methods.len();
            for &id in methods {
                condition_atoms(&problem.methods[id].precondition, &mut result.reads);
                pending.extend(problem.methods[id].network.tasks.iter().cloned());
            }
        }
    }
    result
}
impl InterfaceInput {
    pub fn from_query(query: &InterfaceQuery<'_>) -> Self {
        Self::build(query, false)
    }
    /// Retain the same structural feature counts, evaluating only output atoms.
    pub fn for_boundary(query: &InterfaceQuery<'_>) -> Self {
        Self::build(query, true)
    }
    fn build(query: &InterfaceQuery<'_>, boundary_only: bool) -> Self {
        let f = footprint(query.problem, query.network);
        let mut goal_atoms = State::new();
        condition_atoms(&query.problem.goal, &mut goal_atoms);
        let goal_literals = crate::conflicts::conjunction(&query.problem.goal).unwrap_or_default();
        let writes: State = f.add.union(&f.delete).cloned().collect();
        let mut candidates: State = if boundary_only {
            State::new()
        } else {
            f.reads.union(&writes).chain(&goal_atoms).cloned().collect()
        };
        candidates.extend(query.boundary.iter().cloned());
        let log = |n: usize| (n as f64).ln_1p();
        let bit = |b| if b { 1.0 } else { 0.0 };
        let atoms = candidates
            .into_iter()
            .map(|atom| {
                let features = vec![
                    bit(query.incoming.contains(&atom)),
                    bit(f.reads.contains(&atom)),
                    bit(f.add.contains(&atom)),
                    bit(f.delete.contains(&atom)),
                    bit(query.future_reads.contains(&atom)),
                    bit(goal_literals.must_be_true.contains(&atom)),
                    bit(goal_literals.must_be_false.contains(&atom)),
                    bit(goal_atoms.contains(&atom)),
                    log(atom.arguments.len()),
                    log(query.network.tasks.len()),
                    log(f.reads.len()),
                    log(writes.len()),
                    log(query.incoming.len()),
                    log(query.future_reads.len()),
                    log(f.methods),
                    1.0,
                ];
                AtomInput { atom, features }
            })
            .collect();
        Self {
            feature_version: INTERFACE_FEATURE_VERSION,
            atoms,
        }
    }
    pub fn validate(&self) -> Result<()> {
        let mut seen = BTreeSet::new();
        if self.feature_version != INTERFACE_FEATURE_VERSION
            || self.atoms.iter().any(|a| {
                a.features.len() != ATOM_FEATURES
                    || a.features.iter().any(|x| !x.is_finite())
                    || !seen.insert(&a.atom)
            })
        {
            bail!("invalid interface input schema");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterfaceModel {
    pub schema_version: u32,
    pub feature_version: u32,
    pub hidden: usize,
    pub parameters: Vec<f64>,
}
impl InterfaceModel {
    pub fn new(hidden: usize, mut seed: u64) -> Result<Self> {
        if !(1..=32).contains(&hidden) {
            bail!("interface hidden width must be 1..32");
        }
        let parameters = (0..hidden * (ATOM_FEATURES + 1) + 3 * (hidden + 1))
            .map(|_| {
                ((random(&mut seed) >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0)
                    / (hidden as f64).sqrt()
            })
            .collect();
        Ok(Self {
            schema_version: 1,
            feature_version: INTERFACE_FEATURE_VERSION,
            hidden,
            parameters,
        })
    }
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != 1
            || self.feature_version != INTERFACE_FEATURE_VERSION
            || !(1..=32).contains(&self.hidden)
            || self.parameters.len() != self.hidden * (ATOM_FEATURES + 1) + 3 * (self.hidden + 1)
            || self.parameters.iter().any(|v| !v.is_finite())
        {
            bail!("invalid interface model");
        }
        Ok(())
    }
    fn logits(&self, tape: &mut Tape, features: &[f64]) -> [usize; 3] {
        let h = self.hidden;
        let mut hidden = Vec::new();
        for i in 0..h {
            let base = i * (ATOM_FEATURES + 1);
            let mut value = base + ATOM_FEATURES;
            for (j, &x) in features.iter().enumerate() {
                let product = tape.scale(base + j, x);
                value = tape.add(value, product);
            }
            hidden.push(tape.tanh(value));
        }
        std::array::from_fn(|class| {
            let base = h * (ATOM_FEATURES + 1) + class * (h + 1);
            let mut value = base + h;
            for (j, &node) in hidden.iter().enumerate() {
                let product = tape.mul(base + j, node);
                value = tape.add(value, product);
            }
            value
        })
    }
    fn tape(&self, training: bool) -> Tape {
        let mut tape = if training {
            Tape::new()
        } else {
            Tape::inference()
        };
        for &p in &self.parameters {
            tape.constant(p);
        }
        tape
    }
    pub fn predict(&self, input: &InterfaceInput) -> Result<Vec<[f64; 3]>> {
        self.validate()?;
        input.validate()?;
        let mut result = Vec::new();
        for atom in &input.atoms {
            let mut tape = self.tape(false);
            let logits = self
                .logits(&mut tape, &atom.features)
                .map(|i| tape.value(i));
            let max = logits.into_iter().fold(f64::NEG_INFINITY, f64::max);
            let exp = logits.map(|x| (x - max).exp());
            let total: f64 = exp.iter().sum();
            if !total.is_finite() || total <= 0.0 {
                bail!("nonfinite interface prediction");
            }
            result.push(exp.map(|x| x / total));
        }
        Ok(result)
    }
    pub fn loss_gradient(
        &self,
        input: &InterfaceInput,
        targets: &[FluentStatus],
        class_weights: &[f64; 3],
    ) -> Result<(f64, Vec<f64>)> {
        self.evaluate(input, targets, class_weights, true)
    }
    pub fn loss(
        &self,
        input: &InterfaceInput,
        targets: &[FluentStatus],
        class_weights: &[f64; 3],
    ) -> Result<f64> {
        Ok(self.evaluate(input, targets, class_weights, false)?.0)
    }
    fn evaluate(
        &self,
        input: &InterfaceInput,
        targets: &[FluentStatus],
        class_weights: &[f64; 3],
        training: bool,
    ) -> Result<(f64, Vec<f64>)> {
        self.validate()?;
        input.validate()?;
        if input.atoms.is_empty()
            || targets.len() != input.atoms.len()
            || class_weights.iter().any(|w| !w.is_finite() || *w <= 0.0)
        {
            bail!("invalid interface targets/weights");
        }
        let mut value = 0.0;
        let mut gradient = if training {
            vec![0.0; self.parameters.len()]
        } else {
            Vec::new()
        };
        for (atom, target) in input.atoms.iter().zip(targets) {
            // Atoms are independent conditioned on their features. Accumulating
            // row gradients keeps the derivative graph independent of atom count.
            let mut tape = self.tape(training);
            let logits = self.logits(&mut tape, &atom.features);
            let mut sum = logits[0];
            for &logit in &logits[1..] {
                let negative = tape.scale(sum, -1.0);
                let difference = tape.add(logit, negative);
                let softplus = tape.softplus(difference);
                sum = tape.add(sum, softplus);
            }
            let negative = tape.scale(logits[target.index()], -1.0);
            let ce = tape.add(sum, negative);
            let weighted = tape.scale(ce, class_weights[target.index()] / targets.len() as f64);
            value += tape.value(weighted);
            if training {
                for (sum, g) in gradient
                    .iter_mut()
                    .zip(tape.backward(weighted, self.parameters.len()))
                {
                    *sum += g;
                }
            }
        }
        if !value.is_finite() || gradient.iter().any(|v| !v.is_finite()) {
            bail!("nonfinite interface loss");
        }
        Ok((value, gradient))
    }
}
impl InterfaceGuide for InterfaceModel {
    fn propose(&self, query: &InterfaceQuery<'_>) -> Result<BoundaryState, SolverError> {
        let input = InterfaceInput::for_boundary(query);
        let predictions = self
            .predict(&input)
            .map_err(|e| SolverError(e.to_string()))?;
        let mut boundary = BoundaryState::default();
        for (atom, probabilities) in input.atoms.iter().zip(predictions) {
            let class = (0..3)
                .max_by(|&a, &b| {
                    probabilities[a]
                        .total_cmp(&probabilities[b])
                        .then_with(|| b.cmp(&a))
                })
                .unwrap();
            match class {
                1 => {
                    boundary.must_be_false.insert(atom.atom.clone());
                }
                2 => {
                    boundary.must_be_true.insert(atom.atom.clone());
                }
                _ => {}
            }
        }
        Ok(boundary)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterfaceTrainOptions {
    pub hidden: usize,
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub seed: u64,
    pub split_seed: u64,
    pub max_records: usize,
    pub class_weights: [f64; 3],
    #[serde(default = "include_local_default")]
    pub include_local: bool,
}
fn include_local_default() -> bool {
    true
}
impl Default for InterfaceTrainOptions {
    fn default() -> Self {
        Self {
            hidden: 8,
            epochs: 50,
            batch_size: 8,
            learning_rate: 0.003,
            seed: 42,
            split_seed: 42,
            max_records: 10_000,
            class_weights: [0.25, 1.0, 1.0],
            include_local: true,
        }
    }
}
impl InterfaceTrainOptions {
    fn validate(&self) -> Result<()> {
        if !(1..=32).contains(&self.hidden)
            || self.epochs == 0
            || self.batch_size == 0
            || self.max_records == 0
            || !self.learning_rate.is_finite()
            || self.learning_rate <= 0.0
            || self
                .class_weights
                .iter()
                .any(|w| !w.is_finite() || *w <= 0.0)
        {
            bail!("invalid interface training options");
        }
        Ok(())
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterfaceSource {
    pub record: String,
    pub group: String,
    pub split: Split,
}
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InterfaceMetrics {
    pub interfaces: usize,
    pub atoms: usize,
    pub loss: f64,
    pub accuracy: Option<f64>,
    pub per_class_atoms: [usize; 3],
    pub per_class_accuracy: [Option<f64>; 3],
    #[serde(default)]
    pub global_interfaces: usize,
    #[serde(default)]
    pub local_interfaces: usize,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterfaceCheckpoint {
    pub schema_version: u32,
    pub model: InterfaceModel,
    pub optimizer: Adam,
    pub options: InterfaceTrainOptions,
    pub completed_epochs: usize,
    pub snapshot_id: String,
    pub sources: Vec<InterfaceSource>,
}
impl InterfaceCheckpoint {
    pub fn validate(&self) -> Result<()> {
        self.model.validate()?;
        self.options.validate()?;
        let mut seen = BTreeSet::new();
        if self.schema_version != 1
            || self.model.hidden != self.options.hidden
            || self.optimizer.first.len() != self.model.parameters.len()
            || self.optimizer.second.len() != self.model.parameters.len()
            || self
                .optimizer
                .first
                .iter()
                .chain(&self.optimizer.second)
                .any(|v| !v.is_finite())
            || self.optimizer.second.iter().any(|v| *v < 0.0)
            || self.sources.iter().any(|s| {
                !seen.insert(&s.record) || s.split != group_split(&s.group, self.options.split_seed)
            })
        {
            bail!("invalid interface checkpoint or split provenance");
        }
        Ok(())
    }
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
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterfaceTrainingReport {
    pub schema_version: u32,
    pub checkpoint_id: String,
    pub initial_training: InterfaceMetrics,
    pub training: InterfaceMetrics,
    pub validation: InterfaceMetrics,
    pub test: InterfaceMetrics,
    pub epoch_losses: Vec<f64>,
    pub skipped_without_interfaces: usize,
}
pub struct InterfaceExample {
    pub input: InterfaceInput,
    pub targets: Vec<FluentStatus>,
    pub scope: InterfaceEvidence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterfaceEvidence {
    CompletePlan,
    LocalWitness,
}

/// Complete-plan interfaces and full-trace local witnesses retain distinct scope.
pub fn examples_from_rollout(
    problem: &GroundProblem,
    record: &RolloutRecord,
) -> Result<Vec<InterfaceExample>> {
    record.validate()?;
    let mut examples = Vec::new();
    let mut seen = BTreeSet::new();
    for outcome in &record.outcomes {
        let Some(result) = outcome
            .result
            .as_ref()
            .filter(|r| r.status == MasterStatus::Feasible)
        else {
            continue;
        };
        crate::solver::validate_problem_certificate(
            problem,
            result
                .certificate
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("interface source lacks a global certificate"))?,
        )?;
        for (i, segment) in result
            .segments
            .iter()
            .enumerate()
            .take(result.segments.len().saturating_sub(1))
        {
            crate::solver::validate_certificate(problem, &segment.request, &segment.certificate)?;
            if segment.request.goal != segment.outgoing.condition()?
                || !segment
                    .outgoing
                    .condition()?
                    .holds(&segment.certificate.final_state)
            {
                bail!("unverified interface target");
            }
            let mut future_reads = State::new();
            condition_atoms(&problem.goal, &mut future_reads);
            for future in &result.segments[i + 1..] {
                future_reads.extend(footprint(problem, &future.request.network).reads);
            }
            let boundary: Vec<_> = segment
                .outgoing
                .must_be_true
                .union(&segment.outgoing.must_be_false)
                .cloned()
                .collect();
            let input = InterfaceInput::from_query(&InterfaceQuery {
                problem,
                network: &segment.request.network,
                incoming: &segment.request.initial_state,
                boundary: &boundary,
                future_reads: &future_reads,
            });
            if input.atoms.is_empty() {
                continue;
            }
            let targets: Vec<_> = input
                .atoms
                .iter()
                .map(|a| {
                    if segment.outgoing.must_be_true.contains(&a.atom) {
                        FluentStatus::True
                    } else if segment.outgoing.must_be_false.contains(&a.atom) {
                        FluentStatus::False
                    } else {
                        FluentStatus::Unknown
                    }
                })
                .collect();
            let key = serde_json::to_vec(&(&input, &targets))?;
            if seen.insert(key) {
                examples.push(InterfaceExample {
                    input,
                    targets,
                    scope: InterfaceEvidence::CompletePlan,
                });
            }
        }
    }
    for outcome in &record.outcomes {
        let mut requests = BTreeMap::new();
        for event in &outcome.events {
            match event {
                crate::instrumentation::MasterEvent::OracleStarted {
                    call,
                    request: Some(request),
                    interface_context: Some(context),
                    purpose: crate::instrumentation::OraclePurpose::Solve,
                    ..
                } if !context.boundary.is_empty() => {
                    requests.insert(*call, (request, context));
                }
                crate::instrumentation::MasterEvent::OracleFinished {
                    call,
                    status: crate::solver::ProofStatus::Feasible,
                    certificate: Some(certificate),
                    ..
                } => {
                    let Some((request, context)) = requests.get(call) else {
                        continue;
                    };
                    crate::solver::validate_certificate(problem, request, certificate)?;
                    let target = crate::conflicts::conjunction(&request.goal).ok_or_else(|| {
                        anyhow::anyhow!("local interface goal is not a conjunction")
                    })?;
                    let expected: State = context.boundary.iter().cloned().collect();
                    if !target.must_be_true.is_disjoint(&target.must_be_false)
                        || target
                            .must_be_true
                            .union(&target.must_be_false)
                            .cloned()
                            .collect::<State>()
                            != expected
                    {
                        bail!("local interface context/goal mismatch");
                    }
                    let input = InterfaceInput::from_query(&InterfaceQuery {
                        problem,
                        network: &request.network,
                        incoming: &request.initial_state,
                        boundary: &context.boundary,
                        future_reads: &context.future_reads,
                    });
                    let targets: Vec<_> = input
                        .atoms
                        .iter()
                        .map(|a| {
                            if target.must_be_true.contains(&a.atom) {
                                FluentStatus::True
                            } else if target.must_be_false.contains(&a.atom) {
                                FluentStatus::False
                            } else {
                                FluentStatus::Unknown
                            }
                        })
                        .collect();
                    if seen.insert(serde_json::to_vec(&(&input, &targets))?) {
                        examples.push(InterfaceExample {
                            input,
                            targets,
                            scope: InterfaceEvidence::LocalWitness,
                        });
                    }
                }
                _ => {}
            }
        }
    }
    Ok(examples)
}
fn metrics(
    model: &InterfaceModel,
    examples: &[InterfaceExample],
    weights: &[f64; 3],
) -> Result<InterfaceMetrics> {
    let mut result = InterfaceMetrics {
        interfaces: examples.len(),
        ..Default::default()
    };
    let mut correct = [0usize; 3];
    for example in examples {
        match example.scope {
            InterfaceEvidence::CompletePlan => result.global_interfaces += 1,
            InterfaceEvidence::LocalWitness => result.local_interfaces += 1,
        }
        result.loss += model.loss(&example.input, &example.targets, weights)?;
        for (probability, target) in model.predict(&example.input)?.iter().zip(&example.targets) {
            let prediction = (0..3)
                .max_by(|&a, &b| {
                    probability[a]
                        .total_cmp(&probability[b])
                        .then_with(|| b.cmp(&a))
                })
                .unwrap();
            let c = target.index();
            result.per_class_atoms[c] += 1;
            correct[c] += usize::from(prediction == c);
        }
    }
    result.atoms = result.per_class_atoms.iter().sum();
    if !examples.is_empty() {
        result.loss /= examples.len() as f64;
    }
    if result.atoms > 0 {
        result.accuracy = Some(correct.iter().sum::<usize>() as f64 / result.atoms as f64);
    }
    for (c, &count) in result.per_class_atoms.iter().enumerate() {
        if count > 0 {
            result.per_class_accuracy[c] = Some(correct[c] as f64 / count as f64);
        }
    }
    Ok(result)
}
pub fn train_interface_store(
    store: &DatasetStore,
    options: &InterfaceTrainOptions,
    resume: Option<InterfaceCheckpoint>,
) -> Result<(Entry, Entry, InterfaceTrainingReport)> {
    options.validate()?;
    let snapshot = store.snapshot()?;
    let snapshot_id = snapshot.fingerprint()?;
    let mut partitions = [Vec::new(), Vec::new(), Vec::new()];
    let mut sources = Vec::new();
    let mut auditor = RolloutAuditor::new(store, &snapshot);
    let mut skipped = 0;
    for entry in snapshot
        .entries
        .iter()
        .filter(|e| e.metadata.kind == RecordKind::Rollout)
        .take(options.max_records)
    {
        let record = auditor.audit(entry)?;
        let examples: Vec<_> = examples_from_rollout(auditor.model(&record)?, &record)?
            .into_iter()
            .filter(|e| options.include_local || e.scope == InterfaceEvidence::CompletePlan)
            .collect();
        if examples.is_empty() {
            skipped += 1;
            continue;
        }
        let split = group_split(&record.group_id, options.split_seed);
        partitions[match split {
            Split::Train => 0,
            Split::Validation => 1,
            Split::Test => 2,
        }]
        .extend(examples);
        sources.push(InterfaceSource {
            record: entry.id.clone(),
            group: record.group_id,
            split,
        });
    }
    drop(auditor);
    let [training, validation, test] = partitions;
    if training.is_empty() {
        bail!("no replay-validated multi-block interfaces in training split");
    }
    let (mut model, mut optimizer, start) = if let Some(checkpoint) = resume {
        checkpoint.validate()?;
        if checkpoint.options.seed != options.seed
            || checkpoint.options.split_seed != options.split_seed
            || checkpoint.model.hidden != options.hidden
        {
            bail!("interface resume changes seeds or model shape");
        }
        (
            checkpoint.model,
            checkpoint.optimizer,
            checkpoint.completed_epochs,
        )
    } else {
        let model = InterfaceModel::new(options.hidden, options.seed)?;
        let optimizer = Adam::new(model.parameters.len());
        (model, optimizer, 0)
    };
    let initial_training = metrics(&model, &training, &options.class_weights)?;
    let optimizer_options = TrainOptions {
        learning_rate: options.learning_rate,
        ..Default::default()
    };
    let end = start
        .checked_add(options.epochs)
        .ok_or_else(|| anyhow::anyhow!("interface epoch overflow"))?;
    let mut epoch_losses = Vec::new();
    for epoch in start..end {
        let mut order: Vec<_> = (0..training.len()).collect();
        let mut seed = options.seed.wrapping_add(epoch as u64);
        for i in (1..order.len()).rev() {
            let j = (random(&mut seed) % (i as u64 + 1)) as usize;
            order.swap(i, j);
        }
        let mut loss = 0.0;
        for batch in order.chunks(options.batch_size) {
            let mut gradient = vec![0.0; model.parameters.len()];
            for &i in batch {
                let (l, g) = model.loss_gradient(
                    &training[i].input,
                    &training[i].targets,
                    &options.class_weights,
                )?;
                loss += l;
                for (a, b) in gradient.iter_mut().zip(g) {
                    *a += b / batch.len() as f64;
                }
            }
            optimizer.update_parameters(&mut model.parameters, &gradient, &optimizer_options)?;
        }
        epoch_losses.push(loss / training.len() as f64);
    }
    let checkpoint = InterfaceCheckpoint {
        schema_version: 1,
        model,
        optimizer,
        options: options.clone(),
        completed_epochs: end,
        snapshot_id,
        sources,
    };
    checkpoint.validate()?;
    let model_entry = store.put(
        Metadata {
            kind: RecordKind::InterfaceModel,
            group_id: None,
            source_id: None,
        },
        &checkpoint,
    )?;
    let report = InterfaceTrainingReport {
        schema_version: 1,
        checkpoint_id: model_entry.id.clone(),
        initial_training,
        training: metrics(&checkpoint.model, &training, &options.class_weights)?,
        validation: metrics(&checkpoint.model, &validation, &options.class_weights)?,
        test: metrics(&checkpoint.model, &test, &options.class_weights)?,
        epoch_losses,
        skipped_without_interfaces: skipped,
    };
    let report_entry = store.put(
        Metadata {
            kind: RecordKind::InterfaceTrainingReport,
            group_id: None,
            source_id: Some(model_entry.id.clone()),
        },
        &report,
    )?;
    Ok((model_entry, report_entry, report))
}
