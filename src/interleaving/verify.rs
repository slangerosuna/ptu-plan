use super::*;
/// Reconstruct the selected local hierarchies and original ordering, check every
/// support and threat, then independently reproduce and globally replay the plan.
pub fn validate_interleaving_certificate(
    problem: &GroundProblem,
    certificate: &InterleavingCertificate,
) -> Result<(), SolverError> {
    if certificate.schema_version != 1 {
        return Err(SolverError("unknown POCL certificate schema".into()));
    }
    let proposal = structure::proposal_from_certificate(problem, certificate)?;
    let fixed = structure::compile(problem, proposal)?;
    let n = fixed.proposal.steps.len();
    if certificate.guard_witnesses.len() != n || !fixed.base.is_subset(&certificate.ordering) {
        return Err(SolverError(
            "POCL certificate omits guards or required hierarchy order".into(),
        ));
    }
    if let Some(sources) = &certificate.local_sources {
        let (expected, extra) =
            structure::from_local_sources(problem, certificate.initial_network, sources)?;
        if expected.steps.len() != n
            || expected
                .steps
                .iter()
                .zip(&fixed.proposal.steps)
                .any(|(a, b)| {
                    a.task != b.task
                        || a.partition != b.partition
                        || a.kind != b.kind
                        || a.children != b.children
                })
            || sources
                .iter()
                .map(|s| &s.partition)
                .ne(certificate.partitions.iter())
            || !extra.is_subset(&certificate.ordering)
        {
            return Err(SolverError(
                "POCL certificate differs from supplied local plans".into(),
            ));
        }
    }
    let mut order = order::Ordering::new(n + 2);
    for &(a, b) in &certificate.ordering {
        if !order.add(a, b) {
            return Err(SolverError(
                "invalid or cyclic POCL certificate ordering".into(),
            ));
        }
    }
    let mut guards = vec![BoundaryState::default(); n];
    let mut goal = BoundaryState::default();
    for link in &certificate.causal_links {
        problem.context.check_atom(&link.literal.atom)?;
        let producer = order::endpoints(&link.producer, n)?;
        let consumer = order::endpoints(&link.consumer, n)?;
        if producer == n + 1 || consumer == 0 || !order.reach[producer][consumer] {
            return Err(SolverError(
                "POCL support link has invalid endpoints or order".into(),
            ));
        }
        let supports = if producer == 0 {
            problem.initial_state.contains(&link.literal.atom) == link.literal.positive
        } else if link.literal.positive {
            fixed.add[producer - 1].contains(&link.literal.atom)
        } else {
            fixed.delete[producer - 1].contains(&link.literal.atom)
        };
        if !supports {
            return Err(SolverError(
                "POCL producer does not establish its literal".into(),
            ));
        }
        if consumer == n + 1 {
            order::add_literal(&mut goal, &link.literal);
        } else {
            order::add_literal(&mut guards[consumer - 1], &link.literal);
        }
        if fixed.clobberers.get(&link.literal).is_some_and(|events| {
            events
                .iter()
                .any(|&t| t != consumer && !order.reach[t][producer] && !order.reach[consumer][t])
        }) {
            return Err(SolverError("unresolved POCL causal threat".into()));
        }
    }
    if guards != certificate.guard_witnesses || goal != certificate.goal_witness {
        return Err(SolverError(
            "POCL guard witness differs from support links".into(),
        ));
    }
    for (guard, witness) in fixed.guards.iter().zip(&guards) {
        witness.condition()?;
        if order::partial_holds(guard, witness) != Some(true) {
            return Err(SolverError(
                "POCL guard lacks sufficient causal support".into(),
            ));
        }
    }
    goal.condition()?;
    if order::partial_holds(&problem.goal, &goal) != Some(true) {
        return Err(SolverError(
            "POCL global goal lacks sufficient causal support".into(),
        ));
    }
    if certificate.linearization.len() != n {
        return Err(SolverError("POCL linearization omits events".into()));
    }
    let mut positions = vec![None; n + 2];
    positions[0] = Some(0);
    positions[n + 1] = Some(n + 1);
    for (position, &event) in certificate.linearization.iter().enumerate() {
        if event >= n || positions[event + 1].replace(position + 1).is_some() {
            return Err(SolverError(
                "invalid or duplicate POCL linearization event".into(),
            ));
        }
    }
    if certificate
        .ordering
        .iter()
        .any(|&(a, b)| positions[a] >= positions[b])
    {
        return Err(SolverError("POCL linearization violates ordering".into()));
    }
    let replayed = structure::replay(problem, &fixed.proposal, &certificate.linearization)?;
    if serde_json::to_value(&replayed).map_err(|e| SolverError(e.to_string()))?
        != serde_json::to_value(&certificate.plan).map_err(|e| SolverError(e.to_string()))?
    {
        return Err(SolverError(
            "POCL plan differs from global HTN replay".into(),
        ));
    }
    Ok(())
}
