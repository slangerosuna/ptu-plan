use anyhow::{Context, Result, bail};
use serde::Serialize;

use super::{BoundaryLabel, Provenance, TrainingSample, make_sample};
use crate::parsing::json::{
    Formula, Htn, InitialTaskNetwork, Predicate, Program, Subtask, Symbol, TaskOrdering,
};
use crate::solver::*;

#[derive(Debug, Clone, Copy, Serialize)]
pub struct BackwardOptions {
    /// Full task-pool sampling with reachable fallback, or directly restrict
    /// sampling to descendants of the template's initial grounded tasks.
    pub grounding_mode: GroundingMode,
    pub count: usize,
    pub seed: u64,
    pub max_attempts: usize,
    pub max_depth: usize,
    pub max_actions: usize,
    pub max_decisions: usize,
    pub max_roots: usize,
    pub min_roots: usize,
    pub randomize_irrelevant: bool,
}

impl Default for BackwardOptions {
    fn default() -> Self {
        Self {
            grounding_mode: GroundingMode::Full,
            count: 128,
            seed: 42,
            max_attempts: 20_000,
            max_depth: 8,
            max_actions: 32,
            max_decisions: 128,
            max_roots: 3,
            min_roots: 1,
            randomize_irrelevant: false,
        }
    }
}

/// Specified SplitMix64 arithmetic gives reproducible sampling without relying
/// on an implementation-defined standard-library random source.
struct Random(u64);
impl Random {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    fn choose(&mut self, size: usize) -> usize {
        (self.next() % size as u64) as usize
    }
}

fn merge(target: &mut BoundaryState, source: BoundaryState) -> Result<()> {
    target.must_be_true.extend(source.must_be_true);
    target.must_be_false.extend(source.must_be_false);
    target.condition()?;
    Ok(())
}

/// This generator intentionally supports conjunctive literal regression only.
/// Richer solver conditions are rejected, rather than approximated as necessary
/// conjuncts. Equality has already compiled to True/False in the ground model.
pub fn literal_requirements(condition: &Condition) -> Result<BoundaryState> {
    match condition {
        Condition::True => Ok(BoundaryState::default()),
        Condition::Atom(atom) => Ok(BoundaryState {
            must_be_true: [atom.clone()].into(),
            ..BoundaryState::default()
        }),
        Condition::Not(inner) => match inner.as_ref() {
            Condition::Atom(atom) => Ok(BoundaryState {
                must_be_false: [atom.clone()].into(),
                ..BoundaryState::default()
            }),
            Condition::Not(inner) => literal_requirements(inner),
            Condition::True => bail!("false condition in regression"),
            Condition::False => Ok(BoundaryState::default()),
            _ => bail!("regression requires conjunctive literals"),
        },
        Condition::And(terms) => {
            let mut requirements = BoundaryState::default();
            for term in terms {
                merge(&mut requirements, literal_requirements(term)?)?;
            }
            Ok(requirements)
        }
        Condition::False => bail!("false condition in regression"),
        _ => bail!("regression requires conjunctive literals"),
    }
}

/// Exact STRIPS regression with add-wins simultaneous effects, followed by the
/// action's preconditions. Contradictions reject this sampled continuation.
pub fn regress(requirements: &BoundaryState, action: &GroundAction) -> Result<BoundaryState> {
    requirements.condition()?;
    if !requirements.must_be_false.is_disjoint(&action.add)
        || requirements
            .must_be_true
            .iter()
            .any(|atom| action.delete.contains(atom) && !action.add.contains(atom))
    {
        bail!("action effects contradict regressed requirements");
    }
    let mut before = requirements.clone();
    before
        .must_be_true
        .retain(|atom| !action.add.contains(atom));
    before
        .must_be_false
        .retain(|atom| !action.delete.contains(atom));
    merge(&mut before, literal_requirements(&action.precondition)?)?;
    Ok(before)
}

fn requirements_by_position(
    model: &GroundProblem,
    schedule: &[(Decision, GroundTask)],
    goal: &Condition,
) -> Result<Vec<BoundaryState>> {
    let mut requirements = literal_requirements(goal)?;
    let mut result = vec![BoundaryState::default(); schedule.len() + 1];
    result[schedule.len()] = requirements.clone();
    for (position, (decision, task)) in schedule.iter().enumerate().rev() {
        match decision.kind {
            DecisionKind::Execute => {
                requirements = regress(
                    &requirements,
                    model.actions.get(task).context("missing action")?,
                )?;
            }
            DecisionKind::Refine { method } => {
                merge(
                    &mut requirements,
                    literal_requirements(&model.methods[method].precondition)?,
                )?;
            }
        }
        result[position] = requirements.clone();
    }
    Ok(result)
}

pub fn boundary_labels(
    model: &GroundProblem,
    trace: &[ExecutionStep],
    goal: &Condition,
    optimal: bool,
) -> Result<Vec<BoundaryLabel>> {
    let schedule: Vec<_> = trace
        .iter()
        .map(|step| (step.decision.clone(), step.task.clone()))
        .collect();
    let requirements = requirements_by_position(model, &schedule, goal)?;
    let total = trace
        .iter()
        .filter(|step| step.decision.kind == DecisionKind::Execute)
        .count();
    let mut completed = 0;
    let mut labels = Vec::new();
    for position in 0..=trace.len() {
        if position > 0 && trace[position - 1].decision.kind == DecisionKind::Execute {
            completed += 1;
        }
        if position == 0
            || (position > 0 && trace[position - 1].decision.kind == DecisionKind::Execute)
        {
            if let Some(state) = trace
                .get(position)
                .map(|step| &step.before)
                .or_else(|| trace.last().map(|step| &step.after))
                && !requirements[position].condition()?.holds(state)
            {
                bail!("regressed boundary does not hold in the validated trace");
            }
            labels.push(BoundaryLabel {
                after_actions: completed,
                requirements: requirements[position].clone(),
                remaining_actions: total - completed,
                exact_remaining_cost: optimal,
            });
        }
    }
    Ok(labels)
}

fn ast_atom(atom: &GroundAtom) -> Predicate {
    Predicate {
        name: atom.predicate.clone(),
        variables: atom
            .arguments
            .iter()
            .map(|name| Symbol {
                name: name.clone(),
                symbol_type: None,
            })
            .collect(),
    }
}

fn ast_goal(goal: &BoundaryState) -> Formula {
    Formula::And(
        goal.must_be_true
            .iter()
            .map(|a| Formula::Atom(ast_atom(a)))
            .chain(
                goal.must_be_false
                    .iter()
                    .map(|a| Formula::Not(Box::new(Formula::Atom(ast_atom(a))))),
            )
            .collect(),
    )
}

pub struct GenerationReport {
    pub samples: Vec<TrainingSample>,
    pub attempts: usize,
    pub last_rejection: String,
}

/// Strict convenience API. Collection should use `generate_to` to retain
/// validated partial progress when a quota or job deadline cannot be reached.
pub fn generate(
    template: &Program,
    grounding: &GroundingOptions,
    options: &BackwardOptions,
) -> Result<Vec<TrainingSample>> {
    let report = generate_to(template, grounding, options, |_| Ok(()))?;
    if report.samples.len() != options.count {
        bail!(
            "generated {}/{} samples in {} attempts; last rejection: {}",
            report.samples.len(),
            options.count,
            report.attempts,
            report.last_rejection
        );
    }
    Ok(report.samples)
}

pub fn generate_to(
    template: &Program,
    grounding: &GroundingOptions,
    options: &BackwardOptions,
    mut accept: impl FnMut(&TrainingSample) -> Result<()>,
) -> Result<GenerationReport> {
    if options.min_roots == 0
        || options.min_roots > options.max_roots
        || options.max_actions == 0
        || options.max_decisions == 0
    {
        bail!("sampling limits must be positive");
    }
    let model = match if options.grounding_mode == GroundingMode::InitialReachable {
        ground_reachable_program(template, grounding)
    } else {
        ground_program(template, grounding)
    } {
        Err(error) if error.0.contains("grounding budget exhausted") => {
            ground_reachable_program(template, grounding)?
        }
        result => result?,
    };
    let roots: Vec<_> = model
        .methods
        .iter()
        .map(|method| method.task.clone())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    if roots.is_empty() {
        bail!("backward sampling requires grounded abstract tasks");
    }
    let mut random = Random(options.seed);
    let mut samples = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    let mut last_error = String::new();
    let mut attempts = 0;
    for attempt in 0..options.max_attempts {
        if samples.len() == options.count {
            break;
        }
        attempts = attempt + 1;
        let count = options.min_roots + random.choose(options.max_roots - options.min_roots + 1);
        let tasks = (0..count)
            .map(|_| roots[random.choose(roots.len())].clone())
            .collect::<Vec<_>>();
        let total_order = random.choose(2) == 0;
        let network = GroundNetwork {
            tasks,
            precedence: if total_order {
                (1..count).map(|i| (i - 1, i)).collect()
            } else {
                Default::default()
            },
        };
        let outcome = (|| -> Result<TrainingSample> {
            let schedule = sample_structure(
                &model,
                &network,
                options.max_depth,
                options.max_actions,
                options.max_decisions,
                &mut |size| random.choose(size),
            )?;
            let last_action = schedule
                .iter()
                .rev()
                .find(|(decision, _)| decision.kind == DecisionKind::Execute)
                .context("empty sampled plan")?;
            let action = &model.actions[&last_action.1];
            let candidates: Vec<_> = action
                .add
                .iter()
                .map(|atom| (atom.clone(), true))
                .chain(
                    action
                        .delete
                        .difference(&action.add)
                        .map(|atom| (atom.clone(), false)),
                )
                .collect();
            if candidates.is_empty() {
                bail!("last action has no observable effect");
            }
            let mut terminal = BoundaryState::default();
            let selected = &candidates[random.choose(candidates.len())];
            if selected.1 {
                terminal.must_be_true.insert(selected.0.clone());
            } else {
                terminal.must_be_false.insert(selected.0.clone());
            }
            let goal = terminal.condition()?;
            let requirements = requirements_by_position(&model, &schedule, &goal)?;
            let mut state = model.initial_state.clone();
            if options.randomize_irrelevant {
                let universe: std::collections::BTreeSet<_> = model
                    .actions
                    .values()
                    .flat_map(|a| a.add.iter().chain(&a.delete))
                    .cloned()
                    .collect();
                for atom in universe {
                    if !requirements[0].must_be_true.contains(&atom)
                        && !requirements[0].must_be_false.contains(&atom)
                        && random.choose(20) == 0
                        && !state.remove(&atom)
                    {
                        state.insert(atom);
                    }
                }
            }
            state.extend(requirements[0].must_be_true.iter().cloned());
            for atom in &requirements[0].must_be_false {
                state.remove(atom);
            }
            let request = LocalRequest {
                network: network.clone(),
                initial_state: state.clone(),
                goal,
            };
            let decisions: Vec<_> = schedule.into_iter().map(|(decision, _)| decision).collect();
            let mut certificate = certify_decisions(&model, &request, &decisions)?;
            let mut program = template.clone();
            let problem = program
                .problem
                .as_mut()
                .context("template has no problem")?;
            problem.name = format!("backward-{:016x}-{attempt:06}", options.seed);
            problem.init_state = state.iter().map(ast_atom).collect();
            problem.goal = Some(ast_goal(&terminal));
            problem.init_tn = Some(InitialTaskNetwork {
                parameters: None,
                tn: Htn {
                    subtasks: network
                        .tasks
                        .iter()
                        .enumerate()
                        .map(|(i, task)| Subtask {
                            id: Some(Symbol {
                                name: format!("r{i}"),
                                symbol_type: None,
                            }),
                            task: Symbol {
                                name: task.name.clone(),
                                symbol_type: None,
                            },
                            terms: task
                                .arguments
                                .iter()
                                .map(|name| Symbol {
                                    name: name.clone(),
                                    symbol_type: None,
                                })
                                .collect(),
                        })
                        .collect(),
                    orderings: if total_order {
                        TaskOrdering::Total
                    } else {
                        TaskOrdering::Partial(Vec::new())
                    },
                    constraints: None,
                },
            });
            let compiled = match model.grounding_mode {
                GroundingMode::Full => ground_program(&program, grounding)?,
                GroundingMode::InitialReachable => ground_reachable_program(&program, grounding)?,
            };
            if model.grounding_mode == GroundingMode::InitialReachable {
                for decision in &mut certificate.decisions {
                    if let DecisionKind::Refine { method } = &mut decision.kind {
                        let original = &model.methods[*method];
                        *method = compiled
                            .methods
                            .iter()
                            .position(|m| {
                                m.name == original.name
                                    && m.bindings == original.bindings
                                    && m.task == original.task
                            })
                            .context("sampled method is missing from reachable model")?;
                    }
                }
            }
            make_sample(
                program,
                &compiled,
                certificate,
                Provenance::Backward {
                    seed: options.seed,
                    attempt,
                    template: template.problem.as_ref().unwrap().name.clone(),
                },
            )
        })();
        match outcome {
            Ok(sample) if seen.insert(sample.id.clone()) => {
                accept(&sample)?;
                samples.push(sample);
            }
            Ok(_) => last_error = "duplicate symbolic problem and solution".into(),
            Err(error) => last_error = error.to_string(),
        }
    }
    Ok(GenerationReport {
        samples,
        attempts,
        last_rejection: last_error,
    })
}
