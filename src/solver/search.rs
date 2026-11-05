use std::collections::{HashSet, VecDeque};
use std::time::Instant;

use super::*;

#[derive(Debug, Clone)]
struct Pending {
    task: GroundTask,
    occurrence: usize,
    depth: usize,
}

#[derive(Debug, Clone)]
struct SearchNode {
    state: State,
    pending: Vec<Pending>,
    order: BTreeSet<(usize, usize)>,
    next_id: usize,
    initial: usize,
    decisions: Vec<Decision>,
    actions: Vec<GroundTask>,
}

#[derive(PartialEq, Eq, Hash)]
struct SearchKey {
    state: State,
    tasks: Vec<(GroundTask, usize)>,
    order: BTreeSet<(usize, usize)>,
    plan_position: Option<usize>,
}

impl SearchNode {
    fn new(request: &LocalRequest, initial: usize) -> Self {
        Self {
            state: request.initial_state.clone(),
            pending: request
                .network
                .tasks
                .iter()
                .enumerate()
                .map(|(occurrence, task)| Pending {
                    task: task.clone(),
                    occurrence,
                    depth: 0,
                })
                .collect(),
            order: request.network.precedence.clone(),
            next_id: request.network.tasks.len(),
            initial,
            decisions: Vec::new(),
            actions: Vec::new(),
        }
    }

    fn key(&self, exactness: ExactnessMode, fixed_plan: bool) -> SearchKey {
        SearchKey {
            state: self.state.clone(),
            tasks: self
                .pending
                .iter()
                .map(|task| {
                    (
                        task.task.clone(),
                        if exactness == ExactnessMode::Unbounded {
                            0
                        } else {
                            task.depth
                        },
                    )
                })
                .collect(),
            order: self.order.clone(),
            plan_position: fixed_plan.then_some(self.actions.len()),
        }
    }

    fn enabled(&self, index: usize) -> bool {
        !self.order.iter().any(|&(_, after)| after == index)
    }

    fn replace(&mut self, index: usize, network: &GroundNetwork) {
        let count = network.tasks.len();
        let remap = |old: usize| if old < index { old } else { old - 1 + count };
        let mut order = BTreeSet::new();
        let mut predecessors = Vec::new();
        let mut successors = Vec::new();
        for &(before, after) in &self.order {
            if before == index {
                successors.push(after);
                for child in 0..count {
                    order.insert((index + child, remap(after)));
                }
            } else if after == index {
                predecessors.push(before);
                for child in 0..count {
                    order.insert((remap(before), index + child));
                }
            } else {
                order.insert((remap(before), remap(after)));
            }
        }
        if count == 0 {
            for &before in &predecessors {
                for &after in &successors {
                    order.insert((remap(before), remap(after)));
                }
            }
        }
        order.extend(
            network
                .precedence
                .iter()
                .map(|&(before, after)| (index + before, index + after)),
        );
        let depth = self.pending[index].depth + 1;
        let children: Vec<_> = network
            .tasks
            .iter()
            .enumerate()
            .map(|(offset, task)| Pending {
                task: task.clone(),
                occurrence: self.next_id + offset,
                depth,
            })
            .collect();
        self.next_id += count;
        self.pending.splice(index..=index, children);
        self.order = order;
    }
}

fn check_request(problem: &GroundProblem, request: &LocalRequest) -> Result<(), SolverError> {
    network_closure(&request.network)?;
    for task in &request.network.tasks {
        if !problem.tasks.contains(task) {
            return Err(SolverError(format!("unknown ground task: {task:?}")));
        }
    }
    for atom in &request.initial_state {
        problem.context.check_atom(atom)?;
    }
    problem.context.check_condition(&request.goal)
}

fn apply(
    problem: &GroundProblem,
    node: &SearchNode,
    decision: &Decision,
) -> Result<SearchNode, SolverError> {
    let index = node
        .pending
        .iter()
        .position(|pending| pending.occurrence == decision.occurrence)
        .ok_or_else(|| SolverError("certificate references a missing task occurrence".into()))?;
    if !node.enabled(index) {
        return Err(SolverError(
            "task is blocked by ordering predecessors".into(),
        ));
    }
    let task = &node.pending[index].task;
    let mut next = node.clone();
    match decision.kind {
        DecisionKind::Execute => {
            let action = problem
                .actions
                .get(task)
                .ok_or_else(|| SolverError("cannot execute a compound task".into()))?;
            if !action.precondition.holds(&node.state) {
                return Err(SolverError("action precondition is false".into()));
            }
            for atom in &action.delete {
                next.state.remove(atom);
            }
            next.state.extend(action.add.iter().cloned());
            next.actions.push(task.clone());
            next.replace(index, &GroundNetwork::default());
        }
        DecisionKind::Refine { method } => {
            let method = problem
                .methods
                .get(method)
                .ok_or_else(|| SolverError("unknown grounded method".into()))?;
            if method.task != *task || !method.precondition.holds(&node.state) {
                return Err(SolverError(
                    "method head or precondition does not match".into(),
                ));
            }
            next.replace(index, &method.network);
        }
    }
    next.decisions.push(decision.clone());
    Ok(next)
}

fn requests(problem: &GroundProblem) -> Vec<LocalRequest> {
    problem
        .initial_networks
        .iter()
        .map(|initial| LocalRequest {
            network: initial.network.clone(),
            initial_state: problem.initial_state.clone(),
            goal: problem.goal.clone(),
        })
        .collect()
}

pub fn solve_problem(
    problem: &GroundProblem,
    budget: &SearchBudget,
) -> Result<SolveResult, SolverError> {
    search(problem, &requests(problem), budget, None)
}

/// Exhaustive progression for this induced network and concrete input state.
/// A feasible local certificate does not establish global composability.
pub fn solve_local(
    problem: &GroundProblem,
    request: &LocalRequest,
    budget: &SearchBudget,
) -> Result<SolveResult, SolverError> {
    search(problem, std::slice::from_ref(request), budget, None)
}

/// Recover an exact HTN witness for an externally supplied primitive sequence.
/// This validates feasibility, never optimality of that sequence.
pub fn solve_with_plan(
    problem: &GroundProblem,
    plan: &[GroundTask],
    budget: &SearchBudget,
) -> Result<SolveResult, SolverError> {
    search(problem, &requests(problem), budget, Some(plan))
}

fn search(
    problem: &GroundProblem,
    requests: &[LocalRequest],
    budget: &SearchBudget,
    plan: Option<&[GroundTask]>,
) -> Result<SolveResult, SolverError> {
    let started = Instant::now();
    let mut statistics = SearchStatistics::default();
    let result = |status, certificate, mut statistics: SearchStatistics| {
        statistics.elapsed_ms = started.elapsed().as_millis();
        Ok(SolveResult {
            status,
            certificate,
            statistics,
            exactness: budget.exactness,
        })
    };
    let timed_out = || started.elapsed().as_millis() >= u128::from(budget.max_time_ms);
    let mut queue = VecDeque::new();
    let mut visited = HashSet::new();
    for (index, request) in requests.iter().enumerate() {
        check_request(problem, request)?;
        let node = SearchNode::new(request, index);
        if visited.insert(node.key(budget.exactness, plan.is_some())) {
            if statistics.generated >= budget.max_generated || timed_out() {
                return result(ProofStatus::BudgetExhausted, None, statistics);
            }
            statistics.generated += 1;
            statistics.peak_pending = statistics.peak_pending.max(node.pending.len());
            queue.push_back(node);
            statistics.peak_frontier = statistics.peak_frontier.max(queue.len());
        }
    }
    while let Some(node) = queue.pop_front() {
        if node.pending.is_empty()
            && requests[node.initial].goal.holds(&node.state)
            && plan.is_none_or(|plan| node.actions.len() == plan.len())
        {
            let certificate = PlanCertificate {
                initial_network: node.initial,
                decisions: node.decisions,
                actions: node.actions,
                final_state: node.state,
            };
            // Replay the complete symbolic witness before returning feasibility.
            validate_certificate(
                problem,
                &requests[certificate.initial_network],
                &certificate,
            )?;
            return result(ProofStatus::Feasible, Some(certificate), statistics);
        }
        if statistics.expanded >= budget.max_expansions || timed_out() {
            return result(ProofStatus::BudgetExhausted, None, statistics);
        }
        statistics.expanded += 1;
        let mut applicable = 0;
        for (index, pending) in node.pending.iter().enumerate() {
            if !node.enabled(index) {
                statistics.ordering_blocked += 1;
                continue;
            }
            let mut choices = Vec::new();
            if let Some(action) = problem.actions.get(&pending.task) {
                let applicable = action.precondition.holds(&node.state);
                if !applicable {
                    statistics.guard_rejections += 1;
                }
                if applicable
                    && plan.is_none_or(|plan| plan.get(node.actions.len()) == Some(&pending.task))
                {
                    choices.push(DecisionKind::Execute);
                }
            } else if let Some(methods) = problem.methods_by_task.get(&pending.task) {
                if let ExactnessMode::Bounded {
                    max_decomposition_depth,
                } = budget.exactness
                    && pending.depth >= max_decomposition_depth
                {
                    statistics.depth_cutoffs += 1;
                    continue;
                }
                for &method in methods {
                    if problem.methods[method].precondition.holds(&node.state) {
                        choices.push(DecisionKind::Refine { method });
                    } else {
                        statistics.guard_rejections += 1;
                    }
                }
            }
            for kind in choices {
                applicable += 1;
                if timed_out() || statistics.generated >= budget.max_generated {
                    return result(ProofStatus::BudgetExhausted, None, statistics);
                }
                match kind {
                    DecisionKind::Execute => statistics.action_applications += 1,
                    DecisionKind::Refine { .. } => statistics.method_refinements += 1,
                }
                let next = apply(
                    problem,
                    &node,
                    &Decision {
                        occurrence: pending.occurrence,
                        kind,
                    },
                )?;
                if visited.insert(next.key(budget.exactness, plan.is_some())) {
                    statistics.peak_pending = statistics.peak_pending.max(next.pending.len());
                    statistics.max_decomposition_depth = statistics
                        .max_decomposition_depth
                        .max(next.pending.iter().map(|p| p.depth).max().unwrap_or(0));
                    statistics.generated += 1;
                    queue.push_back(next);
                    statistics.peak_frontier = statistics.peak_frontier.max(queue.len());
                } else {
                    statistics.duplicates += 1;
                }
            }
        }
        if applicable == 0 {
            statistics.dead_ends += 1;
        }
    }
    let status = if statistics.depth_cutoffs > 0 {
        ProofStatus::InfeasibleWithinBound
    } else {
        ProofStatus::Infeasible
    };
    result(status, None, statistics)
}

pub fn certify_decisions(
    problem: &GroundProblem,
    request: &LocalRequest,
    decisions: &[Decision],
) -> Result<PlanCertificate, SolverError> {
    check_request(problem, request)?;
    let mut node = SearchNode::new(request, 0);
    for decision in decisions {
        node = apply(problem, &node, decision)?;
    }
    let certificate = PlanCertificate {
        initial_network: 0,
        decisions: decisions.to_vec(),
        actions: node.actions,
        final_state: node.state,
    };
    validate_certificate(problem, request, &certificate)?;
    Ok(certificate)
}

pub fn trace_certificate(
    problem: &GroundProblem,
    request: &LocalRequest,
    certificate: &PlanCertificate,
) -> Result<Vec<ExecutionStep>, SolverError> {
    validate_certificate(problem, request, certificate)?;
    let mut node = SearchNode::new(request, certificate.initial_network);
    let mut trace = Vec::new();
    for decision in &certificate.decisions {
        let task = node
            .pending
            .iter()
            .find(|p| p.occurrence == decision.occurrence)
            .unwrap()
            .task
            .clone();
        let next = apply(problem, &node, decision)?;
        trace.push(ExecutionStep {
            decision: decision.clone(),
            task,
            before: node.state,
            after: next.state.clone(),
        });
        node = next;
    }
    Ok(trace)
}

/// Sample hierarchy and linearization before choosing any state. Method and
/// action guards are deliberately left to backward regression and replay.
pub(crate) fn sample_structure(
    problem: &GroundProblem,
    network: &GroundNetwork,
    max_depth: usize,
    max_actions: usize,
    max_decisions: usize,
    choose: &mut impl FnMut(usize) -> usize,
) -> Result<Vec<(Decision, GroundTask)>, SolverError> {
    let request = LocalRequest {
        network: network.clone(),
        initial_state: State::new(),
        goal: Condition::True,
    };
    check_request(problem, &request)?;
    let mut node = SearchNode::new(&request, 0);
    let mut trace = Vec::new();
    let mut actions = 0;
    while !node.pending.is_empty() {
        if trace.len() >= max_decisions {
            return Err(SolverError("sample exceeds decision limit".into()));
        }
        let enabled: Vec<_> = (0..node.pending.len())
            .filter(|&i| node.enabled(i))
            .collect();
        let index = enabled[choose(enabled.len())];
        let pending = node.pending[index].clone();
        let kind = if problem.actions.contains_key(&pending.task) {
            actions += 1;
            if actions > max_actions {
                return Err(SolverError("sample exceeds action limit".into()));
            }
            node.replace(index, &GroundNetwork::default());
            DecisionKind::Execute
        } else {
            if pending.depth >= max_depth {
                return Err(SolverError("sample exceeds depth limit".into()));
            }
            let methods = problem
                .methods_by_task
                .get(&pending.task)
                .ok_or_else(|| SolverError("sample has no refinement".into()))?;
            let method = methods[choose(methods.len())];
            node.replace(index, &problem.methods[method].network);
            DecisionKind::Refine { method }
        };
        trace.push((
            Decision {
                occurrence: pending.occurrence,
                kind,
            },
            pending.task,
        ));
    }
    Ok(trace)
}

/// Replay every refinement and action against exact formulas, ordering, and
/// effects. The redundant action list and final state must match the replay.
pub fn validate_certificate(
    problem: &GroundProblem,
    request: &LocalRequest,
    certificate: &PlanCertificate,
) -> Result<(), SolverError> {
    check_request(problem, request)?;
    let mut node = SearchNode::new(request, certificate.initial_network);
    for decision in &certificate.decisions {
        node = apply(problem, &node, decision)?;
    }
    if !node.pending.is_empty() || !request.goal.holds(&node.state) {
        return Err(SolverError(
            "certificate leaves tasks or a false goal".into(),
        ));
    }
    if node.actions != certificate.actions || node.state != certificate.final_state {
        return Err(SolverError(
            "certificate action list or final state differs from replay".into(),
        ));
    }
    Ok(())
}

pub fn validate_problem_certificate(
    problem: &GroundProblem,
    certificate: &PlanCertificate,
) -> Result<(), SolverError> {
    let requests = requests(problem);
    let request = requests
        .get(certificate.initial_network)
        .ok_or_else(|| SolverError("unknown initial grounding in certificate".into()))?;
    validate_certificate(problem, request, certificate)
}
