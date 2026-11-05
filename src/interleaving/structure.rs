use super::*;
use std::collections::BTreeMap;

pub(super) fn owners(
    network: &GroundNetwork,
    partitions: &[RootPartition],
) -> Result<Vec<usize>, SolverError> {
    network_closure(network)?;
    let mut ids = BTreeSet::new();
    let mut owner = vec![None; network.tasks.len()];
    for partition in partitions {
        if partition.roots.is_empty() || !ids.insert(partition.id) {
            return Err(SolverError("empty or duplicate POCL partition".into()));
        }
        for &root in &partition.roots {
            if root >= owner.len() || owner[root].replace(partition.id).is_some() {
                return Err(SolverError(
                    "POCL partitions overlap or reference an unknown root".into(),
                ));
            }
        }
    }
    owner
        .into_iter()
        .map(|id| {
            id.ok_or_else(|| SolverError("POCL partitions must cover every initial root".into()))
        })
        .collect()
}
pub(super) fn proposed(
    problem: &GroundProblem,
    task: GroundTask,
    partition: usize,
    depth: usize,
) -> ProposedStep {
    let kind = problem
        .actions
        .contains_key(&task)
        .then_some(DecisionKind::Execute);
    ProposedStep {
        task,
        partition,
        depth,
        kind,
        children: Vec::new(),
    }
}
pub(super) fn compile(
    problem: &GroundProblem,
    proposal: Proposal,
) -> Result<FixedPlan, SolverError> {
    let n = proposal.steps.len();
    let network = &problem.initial_networks[proposal.initial].network;
    let mut guards = Vec::with_capacity(n);
    let mut add = Vec::with_capacity(n);
    let mut delete = Vec::with_capacity(n);
    let mut writers = BTreeMap::<Literal, Vec<usize>>::new();
    let mut clobberers = BTreeMap::<Literal, Vec<usize>>::new();
    let mut base = BTreeSet::new();
    let mut exits: Vec<Vec<usize>> = (0..n).map(|i| vec![i + 1]).collect();
    for (i, step) in proposal.steps.iter().enumerate() {
        let (guard, adds, deletes) = match step
            .kind
            .as_ref()
            .ok_or_else(|| SolverError("unrefined POCL local plan".into()))?
        {
            DecisionKind::Execute => {
                let a = problem
                    .actions
                    .get(&step.task)
                    .ok_or_else(|| SolverError("invalid POCL primitive task".into()))?;
                (
                    a.precondition.clone(),
                    a.add.clone(),
                    a.delete.difference(&a.add).cloned().collect(),
                )
            }
            DecisionKind::Refine { method } => {
                let m = &problem.methods[*method];
                (m.precondition.clone(), State::new(), State::new())
            }
        };
        for (atoms, positive) in [(&adds, true), (&deletes, false)] {
            for atom in atoms {
                writers
                    .entry(Literal {
                        atom: atom.clone(),
                        positive,
                    })
                    .or_default()
                    .push(i + 1);
                clobberers
                    .entry(Literal {
                        atom: atom.clone(),
                        positive: !positive,
                    })
                    .or_default()
                    .push(i + 1);
            }
        }
        guards.push(guard);
        add.push(adds);
        delete.push(deletes);
        base.insert((0, i + 1));
        base.insert((i + 1, n + 1));
    }
    base.insert((0, n + 1));
    for (i, step) in proposal.steps.iter().enumerate().rev() {
        if let Some(DecisionKind::Refine { method }) = step.kind {
            let method = &problem.methods[method];
            let cs = &step.children;
            for &child in cs {
                base.insert((i + 1, child + 1));
            }
            let mut has_successor = vec![false; cs.len()];
            for &(a, b) in &method.network.precedence {
                has_successor[a] = true;
                for &last in &exits[cs[a]] {
                    base.insert((last, cs[b] + 1));
                }
            }
            let mut frontier = Vec::new();
            for (position, &child) in cs.iter().enumerate() {
                let mut tail = std::mem::take(&mut exits[child]);
                if !has_successor[position] {
                    if frontier.is_empty() {
                        frontier = tail;
                    } else {
                        frontier.append(&mut tail);
                    }
                }
            }
            if !cs.is_empty() {
                exits[i] = frontier;
            }
        }
    }
    for &(a, b) in &network.precedence {
        for &last in &exits[a] {
            base.insert((last, b + 1));
        }
    }
    Ok(FixedPlan {
        proposal,
        guards,
        add,
        delete,
        writers,
        clobberers,
        base,
    })
}

pub(super) fn proposal_from_certificate(
    problem: &GroundProblem,
    certificate: &InterleavingCertificate,
) -> Result<Proposal, SolverError> {
    let network = problem
        .initial_networks
        .get(certificate.initial_network)
        .ok_or_else(|| SolverError("unknown POCL initial network".into()))?;
    let owner = owners(&network.network, &certificate.partitions)?;
    let mut expected: Vec<_> = network
        .network
        .tasks
        .iter()
        .cloned()
        .zip(owner)
        .map(|(task, partition)| proposed(problem, task, partition, 0))
        .collect();
    for (i, step) in certificate.steps.iter().enumerate() {
        let current = expected
            .get(i)
            .ok_or_else(|| SolverError("foreign POCL decomposition event".into()))?;
        if step.decision.occurrence != i
            || current.task != step.task
            || current.partition != step.partition
        {
            return Err(SolverError("POCL task/owner/occurrence mismatch".into()));
        }
        let depth = current.depth;
        let partition = current.partition;
        match step.decision.kind {
            DecisionKind::Execute => {
                if !problem.actions.contains_key(&step.task) || !step.children.is_empty() {
                    return Err(SolverError("invalid POCL action event".into()));
                }
            }
            DecisionKind::Refine { method } => {
                let method = problem
                    .methods
                    .get(method)
                    .ok_or_else(|| SolverError("unknown POCL grounded method".into()))?;
                if problem.actions.contains_key(&step.task) || method.task != step.task {
                    return Err(SolverError("POCL method head mismatch".into()));
                }
                if let ExactnessMode::Bounded {
                    max_decomposition_depth,
                } = certificate.exactness
                    && depth >= max_decomposition_depth
                {
                    return Err(SolverError(
                        "POCL certificate exceeds decomposition bound".into(),
                    ));
                }
                let children: Vec<_> =
                    (expected.len()..expected.len() + method.network.tasks.len()).collect();
                if children != step.children {
                    return Err(SolverError(
                        "POCL refinement child allocation mismatch".into(),
                    ));
                }
                expected.extend(
                    method
                        .network
                        .tasks
                        .iter()
                        .cloned()
                        .map(|task| proposed(problem, task, partition, depth + 1)),
                );
            }
        }
        expected[i].kind = Some(step.decision.kind.clone());
        expected[i].children = step.children.clone();
    }
    if expected.len() != certificate.steps.len() {
        return Err(SolverError(
            "POCL certificate omits decomposition events".into(),
        ));
    }
    Ok(Proposal {
        initial: certificate.initial_network,
        steps: expected,
    })
}
pub(super) fn replay(
    problem: &GroundProblem,
    proposal: &Proposal,
    linearization: &[usize],
) -> Result<PlanCertificate, SolverError> {
    let network = &problem.initial_networks[proposal.initial].network;
    let mut occurrence: Vec<Option<usize>> = vec![None; proposal.steps.len()];
    for (i, slot) in occurrence.iter_mut().enumerate().take(network.tasks.len()) {
        *slot = Some(i);
    }
    let mut next = network.tasks.len();
    let mut decisions = Vec::with_capacity(linearization.len());
    for &i in linearization {
        let step = &proposal.steps[i];
        let id = occurrence[i]
            .ok_or_else(|| SolverError("POCL action precedes its refinement".into()))?;
        decisions.push(Decision {
            occurrence: id,
            kind: step.kind.clone().unwrap(),
        });
        for &child in &step.children {
            occurrence[child] = Some(next);
            next += 1;
        }
    }
    let request = LocalRequest {
        network: network.clone(),
        initial_state: problem.initial_state.clone(),
        goal: problem.goal.clone(),
    };
    let mut certificate = certify_decisions(problem, &request, &decisions)?;
    certificate.initial_network = proposal.initial;
    validate_problem_certificate(problem, &certificate)?;
    Ok(certificate)
}

/// Adapt validated legacy/sequential local evidence to the structural event
/// representation. Imports remain event-local obligations, not initial facts.
pub(super) fn from_local_sources(
    problem: &GroundProblem,
    initial: usize,
    sources: &[crate::master::LocalSegment],
) -> Result<(Proposal, BTreeSet<(usize, usize)>), SolverError> {
    let network = &problem
        .initial_networks
        .get(initial)
        .ok_or_else(|| SolverError("unknown POCL source initial network".into()))?
        .network;
    let partitions: Vec<_> = sources.iter().map(|s| s.partition.clone()).collect();
    let owners = owners(network, &partitions)?;
    let mut selected = BTreeMap::<Vec<usize>, usize>::new();
    let mut source_orders = Vec::new();
    for source in sources {
        let expected = network.induced(&source.partition.roots)?;
        if serde_json::to_value(&expected).map_err(|e| SolverError(e.to_string()))?
            != serde_json::to_value(&source.request.network)
                .map_err(|e| SolverError(e.to_string()))?
        {
            return Err(SolverError(
                "POCL local source has a foreign induced network".into(),
            ));
        }
        validate_certificate(problem, &source.request, &source.certificate)?;
        problem
            .context
            .check_condition(&source.incoming.condition()?)?;
        problem
            .context
            .check_condition(&source.outgoing.condition()?)?;
        if !source
            .incoming
            .condition()?
            .holds(&source.request.initial_state)
            || !source
                .outgoing
                .condition()?
                .holds(&source.certificate.final_state)
        {
            return Err(SolverError(
                "POCL local source contradicts its boundary annotations".into(),
            ));
        }
        let derived;
        let contract = if let Some(contract) = &source.causal {
            if contract.partition != source.partition.id {
                return Err(SolverError(
                    "POCL local source contract owner mismatch".into(),
                ));
            }
            causal::validate_causal_contract(
                problem,
                &source.request,
                &source.certificate,
                contract,
            )?;
            contract
        } else {
            derived = causal::causal_contract(
                problem,
                &source.request,
                &source.certificate,
                source.partition.id,
            )?;
            &derived
        };
        let mut paths: BTreeMap<usize, Vec<usize>> = source
            .partition
            .roots
            .iter()
            .enumerate()
            .map(|(i, &root)| (i, vec![root]))
            .collect();
        let mut next = source.partition.roots.len();
        let mut event_paths = Vec::new();
        for decision in &source.certificate.decisions {
            let path = paths
                .get(&decision.occurrence)
                .ok_or_else(|| SolverError("unknown POCL local source occurrence".into()))?
                .clone();
            event_paths.push(path.clone());
            if let DecisionKind::Refine { method } = decision.kind {
                selected.insert(path.clone(), method);
                for child in 0..problem.methods[method].network.tasks.len() {
                    let mut child_path = path.clone();
                    child_path.push(child);
                    paths.insert(next, child_path);
                    next += 1;
                }
            }
        }
        source_orders.extend(
            contract
                .internal_order
                .iter()
                .map(|&(a, b)| (event_paths[a].clone(), event_paths[b].clone())),
        );
    }
    let mut proposal = Proposal {
        initial,
        steps: network
            .tasks
            .iter()
            .cloned()
            .zip(owners)
            .map(|(task, owner)| proposed(problem, task, owner, 0))
            .collect(),
    };
    let mut paths: Vec<Vec<usize>> = (0..network.tasks.len()).map(|i| vec![i]).collect();
    let mut i = 0;
    while i < proposal.steps.len() {
        if proposal.steps[i].kind.is_none() {
            let method = *selected
                .get(&paths[i])
                .ok_or_else(|| SolverError("POCL source omits a compound refinement".into()))?;
            let step = proposal.steps[i].clone();
            let start = proposal.steps.len();
            let body = &problem.methods[method].network;
            proposal.steps[i].kind = Some(DecisionKind::Refine { method });
            proposal.steps[i].children = (start..start + body.tasks.len()).collect();
            for (position, task) in body.tasks.iter().cloned().enumerate() {
                let mut path = paths[i].clone();
                path.push(position);
                paths.push(path);
                proposal
                    .steps
                    .push(proposed(problem, task, step.partition, step.depth + 1));
            }
        }
        i += 1;
    }
    let positions: BTreeMap<_, _> = paths
        .into_iter()
        .enumerate()
        .map(|(i, path)| (path, i + 1))
        .collect();
    let mut extra = BTreeSet::new();
    for (a, b) in source_orders {
        let a = *positions
            .get(&a)
            .ok_or_else(|| SolverError("unknown local support-order event".into()))?;
        let b = *positions
            .get(&b)
            .ok_or_else(|| SolverError("unknown local support-order event".into()))?;
        extra.insert((a, b));
    }
    Ok((proposal, extra))
}
