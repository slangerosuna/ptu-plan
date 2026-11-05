//! Replay-derived causal certificates. Refinement guards are first-class events.
use super::*;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Deserialize, Serialize)]
pub struct Literal {
    pub atom: GroundAtom,
    pub positive: bool,
}
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub enum CausalEndpoint {
    Input,
    Step(usize),
    Output,
}
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct CausalLink {
    pub producer: CausalEndpoint,
    pub consumer: CausalEndpoint,
    pub literal: Literal,
}
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct CausalStep {
    pub decision: Decision,
    pub task: GroundTask,
    /// A sufficient literal witness for the full (possibly disjunctive) guard.
    pub requirements: BoundaryState,
    pub add: State,
    pub delete: State,
}
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct ThreatResolution {
    pub link: usize,
    pub threat: usize,
    /// Otherwise the threat is ordered after the consumer.
    pub before_producer: bool,
}
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct CausalContract {
    pub schema_version: u32,
    pub partition: usize,
    pub steps: Vec<CausalStep>,
    /// Event indices, including method refinements, with a certified linearization.
    pub internal_order: BTreeSet<(usize, usize)>,
    pub causal_links: Vec<CausalLink>,
    pub imports: Vec<usize>,
    pub exports: BoundaryState,
    /// Imported literals supported through the output boundary without a write.
    pub preserved: BoundaryState,
    pub input_assumptions: BoundaryState,
    pub output_guarantees: BoundaryState,
    /// Every opposite effect outside a protected support interval is recorded.
    pub threat_resolutions: Vec<ThreatResolution>,
}

fn literal_witness(condition: &Condition, state: &State, desired: bool, out: &mut BoundaryState) {
    match condition {
        Condition::Atom(atom) => {
            if desired {
                &mut out.must_be_true
            } else {
                &mut out.must_be_false
            }
            .insert(atom.clone());
        }
        Condition::Not(c) => literal_witness(c, state, !desired, out),
        Condition::And(cs) if desired => {
            for c in cs {
                literal_witness(c, state, true, out);
            }
        }
        Condition::Or(cs) if !desired => {
            for c in cs {
                literal_witness(c, state, false, out);
            }
        }
        Condition::And(cs) | Condition::Or(cs) => {
            // The replay has already established that such a branch exists.
            let c = cs.iter().find(|c| c.holds(state) == desired).unwrap();
            literal_witness(c, state, desired, out);
        }
        Condition::True | Condition::False => {}
    }
}
fn literals(boundary: &BoundaryState) -> impl Iterator<Item = Literal> + '_ {
    boundary
        .must_be_true
        .iter()
        .map(|a| Literal {
            atom: a.clone(),
            positive: true,
        })
        .chain(boundary.must_be_false.iter().map(|a| Literal {
            atom: a.clone(),
            positive: false,
        }))
}
fn insert(boundary: &mut BoundaryState, literal: &Literal) {
    if literal.positive {
        &mut boundary.must_be_true
    } else {
        &mut boundary.must_be_false
    }
    .insert(literal.atom.clone());
}

/// Derive support and threat protection from a valid concrete local witness.
/// The resulting partial order is conservative and not a global interleaving proof.
pub fn causal_contract(
    problem: &GroundProblem,
    request: &LocalRequest,
    witness: &PlanCertificate,
    partition: usize,
) -> Result<CausalContract, SolverError> {
    derive_contract(problem, request, witness, partition, 2)
}

fn derive_contract(
    problem: &GroundProblem,
    request: &LocalRequest,
    witness: &PlanCertificate,
    partition: usize,
    version: u32,
) -> Result<CausalContract, SolverError> {
    let trace = trace_certificate(problem, request, witness)?;
    let mut result = CausalContract {
        schema_version: version,
        partition,
        steps: Vec::new(),
        internal_order: BTreeSet::new(),
        causal_links: Vec::new(),
        imports: Vec::new(),
        exports: BoundaryState::default(),
        preserved: BoundaryState::default(),
        input_assumptions: BoundaryState::default(),
        output_guarantees: BoundaryState::default(),
        threat_resolutions: Vec::new(),
    };
    // Build the chosen hierarchy with the same stable occurrence allocation as replay.
    let mut children = BTreeMap::<usize, Vec<usize>>::new();
    let mut subtree = BTreeMap::<usize, Vec<usize>>::new();
    let mut orders: Vec<_> = if version == 1 {
        request.network.precedence.iter().copied().collect()
    } else {
        Vec::new()
    };
    let mut next_id = request.network.tasks.len();
    let mut writers = BTreeMap::<GroundAtom, (usize, bool)>::new();
    for (i, step) in trace.iter().enumerate() {
        let (guard, add, delete) = match step.decision.kind {
            DecisionKind::Execute => {
                let a = &problem.actions[&step.task];
                (
                    &a.precondition,
                    a.add.clone(),
                    a.delete.difference(&a.add).cloned().collect(),
                )
            }
            DecisionKind::Refine { method } => {
                let m = &problem.methods[method];
                let ids: Vec<_> = (next_id..next_id + m.network.tasks.len()).collect();
                if version == 1 {
                    orders.extend(m.network.precedence.iter().map(|&(a, b)| (ids[a], ids[b])));
                }
                next_id += ids.len();
                children.insert(step.decision.occurrence, ids);
                (&m.precondition, State::default(), State::default())
            }
        };
        if version == 1 {
            subtree.insert(step.decision.occurrence, vec![i]);
        }
        let mut requirements = BoundaryState::default();
        literal_witness(guard, &step.before, true, &mut requirements);
        for literal in literals(&requirements) {
            let producer = match writers.get(&literal.atom) {
                Some(&(index, positive)) if positive == literal.positive => {
                    CausalEndpoint::Step(index)
                }
                None => CausalEndpoint::Input,
                _ => {
                    return Err(SolverError(
                        "causal guard support differs from replay".into(),
                    ));
                }
            };
            result.causal_links.push(CausalLink {
                producer,
                consumer: CausalEndpoint::Step(i),
                literal,
            });
        }
        for a in &delete {
            writers.insert(a.clone(), (i, false));
        }
        for a in &add {
            writers.insert(a.clone(), (i, true));
        }
        result.steps.push(CausalStep {
            decision: step.decision.clone(),
            task: step.task.clone(),
            requirements,
            add,
            delete,
        });
    }
    if version == 1 {
        for step in trace.iter().rev() {
            let id = step.decision.occurrence;
            if let Some(cs) = children.get(&id) {
                let mut descendants = Vec::new();
                for child in cs {
                    descendants.extend(subtree[child].iter().copied());
                }
                let parent_event = subtree[&id][0];
                result
                    .internal_order
                    .extend(descendants.iter().map(|&j| (parent_event, j)));
                subtree.get_mut(&id).unwrap().extend(descendants);
            }
        }
        for (a, b) in orders {
            for &i in &subtree[&a] {
                for &j in &subtree[&b] {
                    result.internal_order.insert((i, j));
                }
            }
        }
    } else {
        // Entry/exit frontiers encode the same hierarchy reachability without
        // materializing every ancestor's descendants or their Cartesian products.
        let event: BTreeMap<_, _> = trace
            .iter()
            .enumerate()
            .map(|(i, s)| (s.decision.occurrence, i))
            .collect();
        let mut exits: BTreeMap<_, _> = event.iter().map(|(&id, &i)| (id, vec![i])).collect();
        for step in trace.iter().rev() {
            let id = step.decision.occurrence;
            let DecisionKind::Refine { method } = step.decision.kind else {
                continue;
            };
            let cs = &children[&id];
            let network = &problem.methods[method].network;
            for &child in cs {
                result.internal_order.insert((event[&id], event[&child]));
            }
            let mut has_successor = vec![false; cs.len()];
            for &(a, b) in &network.precedence {
                has_successor[a] = true;
                for &last in &exits[&cs[a]] {
                    result.internal_order.insert((last, event[&cs[b]]));
                }
            }
            let mut frontier = Vec::new();
            for (i, &child) in cs.iter().enumerate() {
                let mut tail = exits.remove(&child).unwrap();
                if !has_successor[i] {
                    if frontier.is_empty() {
                        frontier = tail;
                    } else {
                        frontier.append(&mut tail);
                    }
                }
            }
            if !cs.is_empty() {
                exits.insert(id, frontier);
            }
        }
        for &(a, b) in &request.network.precedence {
            for &last in &exits[&a] {
                result.internal_order.insert((last, event[&b]));
            }
        }
    }
    for (atom, &(_, positive)) in &writers {
        insert(
            &mut result.exports,
            &Literal {
                atom: atom.clone(),
                positive,
            },
        );
    }
    literal_witness(
        &request.goal,
        &witness.final_state,
        true,
        &mut result.output_guarantees,
    );
    result
        .output_guarantees
        .must_be_true
        .extend(result.exports.must_be_true.iter().cloned());
    result
        .output_guarantees
        .must_be_false
        .extend(result.exports.must_be_false.iter().cloned());
    for literal in literals(&result.output_guarantees) {
        let producer = writers
            .get(&literal.atom)
            .map_or(CausalEndpoint::Input, |&(i, _)| CausalEndpoint::Step(i));
        result.causal_links.push(CausalLink {
            producer,
            consumer: CausalEndpoint::Output,
            literal,
        });
    }
    // Sparse effect indexes avoid visiting unrelated events for each link.
    let mut additions = BTreeMap::<&GroundAtom, Vec<usize>>::new();
    let mut deletions = BTreeMap::<&GroundAtom, Vec<usize>>::new();
    for (i, step) in result.steps.iter().enumerate() {
        for atom in &step.add {
            additions.entry(atom).or_default().push(i);
        }
        for atom in &step.delete {
            deletions.entry(atom).or_default().push(i);
        }
    }
    for (index, link) in result.causal_links.iter().enumerate() {
        if link.producer == CausalEndpoint::Input {
            result.imports.push(index);
            insert(&mut result.input_assumptions, &link.literal);
            if link.consumer == CausalEndpoint::Output {
                insert(&mut result.preserved, &link.literal);
            }
        }
        if let (CausalEndpoint::Step(a), CausalEndpoint::Step(b)) = (&link.producer, &link.consumer)
        {
            result.internal_order.insert((*a, *b));
        }
        let opposite = if link.literal.positive {
            &deletions
        } else {
            &additions
        };
        for &threat in opposite.get(&link.literal.atom).into_iter().flatten() {
            if link.consumer == CausalEndpoint::Step(threat) {
                // Guards are checked before this event's own effect.
                continue;
            }
            let before = match link.producer {
                CausalEndpoint::Step(p) if threat < p => {
                    result.internal_order.insert((threat, p));
                    true
                }
                _ => match link.consumer {
                    CausalEndpoint::Step(c) if threat > c => {
                        result.internal_order.insert((c, threat));
                        false
                    }
                    _ => return Err(SolverError("unresolved causal threat in replay".into())),
                },
            };
            result.threat_resolutions.push(ThreatResolution {
                link: index,
                threat,
                before_producer: before,
            });
        }
    }
    if result.internal_order.iter().any(|&(a, b)| a >= b) {
        return Err(SolverError(
            "causal order contradicts certified linearization".into(),
        ));
    }
    result.input_assumptions.condition()?;
    result.output_guarantees.condition()?;
    Ok(result)
}

/// Independently replay and reconstruct all claims; unverified imports must not
/// be used as a feasibility or global-composability certificate.
pub fn validate_causal_contract(
    problem: &GroundProblem,
    request: &LocalRequest,
    witness: &PlanCertificate,
    contract: &CausalContract,
) -> Result<(), SolverError> {
    if !matches!(contract.schema_version, 1 | 2) {
        return Err(SolverError("unknown causal certificate version".into()));
    }
    if *contract
        != derive_contract(
            problem,
            request,
            witness,
            contract.partition,
            contract.schema_version,
        )?
    {
        return Err(SolverError(
            "causal contract differs from replay-derived support/order".into(),
        ));
    }
    Ok(())
}

/// Validate a chosen topological ordering, remap refinement occurrences, and
/// certify it in a concrete world satisfying the contract's imported literals.
/// This proves local executability only, not cross-partition reconciliation.
pub fn certify_causal_linearization(
    problem: &GroundProblem,
    source: &LocalRequest,
    witness: &PlanCertificate,
    contract: &CausalContract,
    order: &[usize],
    initial_state: &State,
) -> Result<PlanCertificate, SolverError> {
    validate_causal_contract(problem, source, witness, contract)?;
    if !contract.input_assumptions.condition()?.holds(initial_state) {
        return Err(SolverError(
            "causal linearization violates imported assumptions".into(),
        ));
    }
    let n = contract.steps.len();
    let mut position = vec![None; n];
    if order.len() != n {
        return Err(SolverError(
            "causal linearization must cover every event".into(),
        ));
    }
    for (i, &event) in order.iter().enumerate() {
        if event >= n || position[event].replace(i).is_some() {
            return Err(SolverError("invalid or duplicate causal event".into()));
        }
    }
    if contract
        .internal_order
        .iter()
        .any(|&(a, b)| position[a] >= position[b])
    {
        return Err(SolverError(
            "causal linearization violates internal order".into(),
        ));
    }
    let mut allocation = BTreeMap::new();
    let mut next = source.network.tasks.len();
    for decision in &witness.decisions {
        if let DecisionKind::Refine { method } = decision.kind {
            let count = problem.methods[method].network.tasks.len();
            allocation.insert(
                decision.occurrence,
                (next..next + count).collect::<Vec<_>>(),
            );
            next += count;
        }
    }
    let mut occurrences: BTreeMap<_, _> = (0..source.network.tasks.len()).map(|i| (i, i)).collect();
    let mut next = source.network.tasks.len();
    let mut decisions = Vec::with_capacity(n);
    for &event in order {
        let old = &contract.steps[event].decision;
        let occurrence = *occurrences
            .get(&old.occurrence)
            .ok_or_else(|| SolverError("causal event precedes its refinement".into()))?;
        decisions.push(Decision {
            occurrence,
            kind: old.kind.clone(),
        });
        if matches!(old.kind, DecisionKind::Refine { .. }) {
            for &child in &allocation[&old.occurrence] {
                occurrences.insert(child, next);
                next += 1;
            }
        }
    }
    let request = LocalRequest {
        initial_state: initial_state.clone(),
        ..source.clone()
    };
    let mut certificate = certify_decisions(problem, &request, &decisions)?;
    certificate.initial_network = witness.initial_network;
    if !contract
        .output_guarantees
        .condition()?
        .holds(&certificate.final_state)
    {
        return Err(SolverError(
            "causal linearization violates output guarantees".into(),
        ));
    }
    Ok(certificate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    fn hierarchy(depth: usize, branching: usize) -> (GroundProblem, LocalRequest, PlanCertificate) {
        let program:crate::parsing::json::Program=serde_json::from_value(json!({"domain":{
            "name":"nested","requirements":["Hierarchy"],"predicates":[],"functions":[],
            "compound_tasks":(0..depth).map(|i|json!({"name":format!("c{i}"),"parameters":[]})).collect::<Vec<_>>(),
            "methods":(0..depth).map(|i| {
                let child=if i+1==depth {"noop".into()} else {format!("c{}",i+1)};
                json!({"name":{"name":format!("m{i}")},"task":{"name":format!("c{i}")},"params":[],"task_terms":[],
                    "tn":{"orderings":"Total","subtasks":(0..branching).map(|_|json!({"task":{"name":child},"terms":[]})).collect::<Vec<_>>()}})
            }).collect::<Vec<_>>(),
            "actions":[{"name":"noop","parameters":[]}]},
            "problem":{"name":"nested","domain_name":"nested","requirements":[],"objects":[],"init_state":[],
                "init_tn":{"tn":{"orderings":"Total","subtasks":[{"task":{"name":"c0"},"terms":[]}]}}}})).unwrap();
        let model = ground_reachable_program(&program, &GroundingOptions::default()).unwrap();
        let request = LocalRequest {
            network: model.initial_networks[0].network.clone(),
            initial_state: State::new(),
            goal: Condition::True,
        };
        let witness = solve_local(&model, &request, &SearchBudget::default())
            .unwrap()
            .certificate
            .unwrap();
        (model, request, witness)
    }
    fn closure(contract: &CausalContract) -> Vec<Vec<bool>> {
        network_closure(&GroundNetwork {
            tasks: contract.steps.iter().map(|s| s.task.clone()).collect(),
            precedence: contract.internal_order.clone(),
        })
        .unwrap()
    }
    #[test]
    fn sparse_frontiers_preserve_legacy_reachability_and_legacy_validation() {
        for (depth, branching) in [(5, 1), (4, 2), (2, 3)] {
            let (model, request, witness) = hierarchy(depth, branching);
            let sparse = causal_contract(&model, &request, &witness, 0).unwrap();
            let legacy = derive_contract(&model, &request, &witness, 0, 1).unwrap();
            assert_eq!(closure(&sparse), closure(&legacy));
            assert!(sparse.internal_order.len() < legacy.internal_order.len());
            let reloaded: CausalContract =
                serde_json::from_slice(&serde_json::to_vec(&legacy).unwrap()).unwrap();
            validate_causal_contract(&model, &request, &witness, &reloaded).unwrap();
            let mut bad = sparse.clone();
            bad.schema_version = 99;
            assert!(validate_causal_contract(&model, &request, &witness, &bad).is_err());
        }
        // A successor of an unordered compound must wait for every exit, not
        // just the last event in the selected witness's serialization.
        let (mut model, mut request, _) = hierarchy(2, 2);
        for method in &mut model.methods {
            method.network.precedence.clear();
        }
        request.network.tasks.push(GroundTask {
            name: "noop".into(),
            arguments: vec![],
        });
        request.network.precedence.insert((0, 1));
        let witness = solve_local(&model, &request, &SearchBudget::default())
            .unwrap()
            .certificate
            .unwrap();
        let sparse = causal_contract(&model, &request, &witness, 0).unwrap();
        let legacy = derive_contract(&model, &request, &witness, 0, 1).unwrap();
        assert_eq!(closure(&sparse), closure(&legacy));
        let successor = sparse
            .steps
            .iter()
            .position(|s| s.decision.occurrence == 1)
            .unwrap();
        assert_eq!(
            sparse
                .internal_order
                .iter()
                .filter(|&&(_, b)| b == successor)
                .count(),
            4
        );
    }
    #[test]
    fn deep_hierarchies_store_linear_order_edges_instead_of_all_descendants() {
        let (model, request, witness) = hierarchy(96, 1);
        let sparse = causal_contract(&model, &request, &witness, 0).unwrap();
        let legacy = derive_contract(&model, &request, &witness, 0, 1).unwrap();
        assert_eq!(sparse.internal_order.len(), 96);
        assert_eq!(legacy.internal_order.len(), 96 * 97 / 2);
        let order: Vec<_> = (0..sparse.steps.len()).collect();
        certify_causal_linearization(
            &model,
            &request,
            &witness,
            &sparse,
            &order,
            &request.initial_state,
        )
        .unwrap();
    }
}
