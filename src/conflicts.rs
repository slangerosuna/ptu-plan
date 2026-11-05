//! Logic-based Benders cuts with exact dependency-scoped incoming assumptions.
use crate::master::RootPartition;
use crate::solver::{
    BoundaryState, Condition, ExactnessMode, GroundAtom, GroundNetwork, GroundProblem,
    LocalRequest, ProofStatus, SearchBudget, SolverError, State, solve_local,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct VerifiedConflictCore {
    pub initial_network: usize,
    pub partition: usize,
    pub initial_state: State,
    /// None retains the legacy exact-world scope. Some fixes every reachable
    /// action/method guard read and retained goal atom; other facts are irrelevant.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub incoming: Option<BoundaryState>,
    pub source_goal: Condition,
    pub core: BoundaryState,
    pub exactness: ExactnessMode,
    pub proof: ProofStatus,
    pub source_call: usize,
    /// Last oracle call proving the retained conjunction infeasible.
    pub verification_call: usize,
    pub removed_literals: usize,
}

/// Return None for formulas outside the literal-conjunction cut language.
pub fn conjunction(condition: &Condition) -> Option<BoundaryState> {
    fn visit(c: &Condition, b: &mut BoundaryState) -> Option<()> {
        match c {
            Condition::True => {}
            Condition::Atom(a) => {
                b.must_be_true.insert(a.clone());
            }
            Condition::Not(c) => match c.as_ref() {
                Condition::Atom(a) => {
                    b.must_be_false.insert(a.clone());
                }
                _ => return None,
            },
            Condition::And(cs) => {
                for c in cs {
                    visit(c, b)?;
                }
            }
            _ => return None,
        }
        Some(())
    }
    let mut result = BoundaryState::default();
    visit(condition, &mut result)?;
    result.condition().ok()?;
    Some(result)
}

impl VerifiedConflictCore {
    pub fn matches(
        &self,
        initial: usize,
        partition: usize,
        state: &State,
        goal: &Condition,
        exactness: ExactnessMode,
    ) -> bool {
        self.initial_network == initial
            && self.partition == partition
            && self.incoming_matches(state)
            && self.exactness == exactness
            && conjunction(goal).is_some_and(|g| {
                self.core.must_be_true.is_subset(&g.must_be_true)
                    && self.core.must_be_false.is_subset(&g.must_be_false)
            })
    }

    pub fn incoming_matches(&self, state: &State) -> bool {
        self.incoming.as_ref().map_or_else(
            || self.initial_state == *state,
            |b| b.must_be_true.is_subset(state) && b.must_be_false.is_disjoint(state),
        )
    }

    /// Logical coverage between validated cuts in the same network/bound scope.
    pub fn subsumes(&self, other: &Self) -> bool {
        if self.initial_network != other.initial_network
            || self.partition != other.partition
            || self.exactness != other.exactness
            || (self.proof != ProofStatus::Infeasible && self.proof != other.proof)
            || !self.core.must_be_true.is_subset(&other.core.must_be_true)
            || !self.core.must_be_false.is_subset(&other.core.must_be_false)
        {
            return false;
        }
        match (&self.incoming, &other.incoming) {
            (Some(a), Some(b)) => {
                a.must_be_true.is_subset(&b.must_be_true)
                    && a.must_be_false.is_subset(&b.must_be_false)
            }
            (Some(_), None) => self.incoming_matches(&other.initial_state),
            (None, None) => self.initial_state == other.initial_state,
            (None, Some(_)) => false,
        }
    }

    pub fn validate_scope(&self) -> Result<(), SolverError> {
        let source = conjunction(&self.source_goal)
            .ok_or_else(|| SolverError("core source is not a literal conjunction".into()))?;
        self.core.condition()?;
        if let Some(incoming) = &self.incoming
            && !incoming.condition()?.holds(&self.initial_state)
        {
            return Err(SolverError(
                "incoming core contradicts its source world".into(),
            ));
        }
        if !self.core.must_be_true.is_subset(&source.must_be_true)
            || !self.core.must_be_false.is_subset(&source.must_be_false)
            || self.removed_literals
                != source.must_be_true.len() + source.must_be_false.len()
                    - self.core.must_be_true.len()
                    - self.core.must_be_false.len()
            || !matches!(
                self.proof,
                ProofStatus::Infeasible | ProofStatus::InfeasibleWithinBound
            )
            || (self.proof == ProofStatus::InfeasibleWithinBound
                && self.exactness == ExactnessMode::Unbounded)
        {
            return Err(SolverError("invalid conflict-core scope".into()));
        }
        Ok(())
    }
}

/// Projection is a bisimulation: every possible local guard and the retained
/// goal depend only on these atoms; unconditional effects preserve agreement.
pub fn incoming_projection(
    problem: &GroundProblem,
    network: &GroundNetwork,
    state: &State,
    core: &BoundaryState,
) -> BoundaryState {
    incoming_projection_from_reads(
        &crate::master::network_footprint(problem, network).reads,
        state,
        core,
    )
}

pub(crate) fn incoming_projection_from_reads(
    reads: &State,
    state: &State,
    core: &BoundaryState,
) -> BoundaryState {
    let mut relevant = reads.clone();
    relevant.extend(core.must_be_true.iter().chain(&core.must_be_false).cloned());
    BoundaryState {
        must_be_true: relevant.intersection(state).cloned().collect(),
        must_be_false: relevant.difference(state).cloned().collect(),
    }
}

/// Called only after a cut matches the current goal. Skip the contiguous ordinal
/// interval for which none of the core's boundary bits can change. XOR proposal
/// ordering does not change the bit positions or these block boundaries.
pub(crate) fn after_forbidden_cube(
    ordinal: usize,
    boundary: &[GroundAtom],
    core: &BoundaryState,
) -> usize {
    let limit = 1usize << boundary.len();
    if let Some(bit) = boundary
        .iter()
        .position(|a| core.must_be_true.contains(a) || core.must_be_false.contains(a))
    {
        (ordinal | ((1usize << bit) - 1))
            .saturating_add(1)
            .min(limit)
    } else {
        limit
    }
}

#[derive(Default)]
pub(crate) struct CoreIndex {
    buckets: std::collections::BTreeMap<(usize, usize), Vec<usize>>,
    pub active: usize,
}
impl CoreIndex {
    pub fn insert(&mut self, archive: &[VerifiedConflictCore], index: usize) -> (bool, usize) {
        let new = &archive[index];
        let bucket = self
            .buckets
            .entry((new.initial_network, new.partition))
            .or_default();
        if bucket.iter().any(|&i| archive[i].subsumes(new)) {
            return (false, 0);
        }
        let before = bucket.len();
        bucket.retain(|&i| !new.subsumes(&archive[i]));
        let retired = before - bucket.len();
        bucket.push(index);
        bucket.sort_by_key(|&i| {
            let c = &archive[i];
            (
                c.proof == ProofStatus::InfeasibleWithinBound,
                c.core.must_be_true.len() + c.core.must_be_false.len(),
                i,
            )
        });
        self.active = self.active + 1 - retired;
        (true, retired)
    }
    pub fn matching(
        &self,
        archive: &[VerifiedConflictCore],
        initial: usize,
        partition: usize,
        state: &State,
        goal: &Condition,
        exactness: ExactnessMode,
    ) -> Option<usize> {
        let goal = conjunction(goal)?;
        self.buckets
            .get(&(initial, partition))?
            .iter()
            .copied()
            .find(|&i| {
                let c = &archive[i];
                c.exactness == exactness
                    && c.incoming_matches(state)
                    && c.core.must_be_true.is_subset(&goal.must_be_true)
                    && c.core.must_be_false.is_subset(&goal.must_be_false)
            })
    }
}

/// Independently re-run the oracle for a cut; resource exhaustion never verifies it.
pub fn validate_conflict_core(
    problem: &GroundProblem,
    partition: &RootPartition,
    cut: &VerifiedConflictCore,
    budget: &SearchBudget,
) -> Result<(), SolverError> {
    cut.validate_scope()?;
    if cut.partition != partition.id || cut.exactness != budget.exactness {
        return Err(SolverError("conflict-core partition/bound mismatch".into()));
    }
    let network = problem
        .initial_networks
        .get(cut.initial_network)
        .ok_or_else(|| SolverError("unknown conflict-core initial network".into()))?;
    let induced = network.network.induced(&partition.roots)?;
    if let Some(incoming) = &cut.incoming
        && *incoming != incoming_projection(problem, &induced, &cut.initial_state, &cut.core)
    {
        return Err(SolverError(
            "incoming core omits a dependency or has a foreign assumption".into(),
        ));
    }
    let result = solve_local(
        problem,
        &LocalRequest {
            network: induced,
            initial_state: cut.initial_state.clone(),
            goal: cut.core.condition()?,
        },
        budget,
    )?;
    if result.status != ProofStatus::Infeasible
        && !(cut.proof == ProofStatus::InfeasibleWithinBound && result.status == cut.proof)
    {
        return Err(SolverError("oracle did not verify conflict core".into()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn atom(i: usize) -> GroundAtom {
        GroundAtom {
            predicate: format!("p{i}"),
            arguments: vec![],
        }
    }
    #[test]
    fn cube_jumps_skip_only_forbidden_assignments_under_every_xor_order() {
        let boundary: Vec<_> = (0..5).map(atom).collect();
        let limit = 1usize << boundary.len();
        // Each atom is absent/false/true in the forbidden conjunction.
        for ternary in 0..3usize.pow(5) {
            let mut code = ternary;
            let mut core = BoundaryState::default();
            for a in &boundary {
                match code % 3 {
                    1 => {
                        core.must_be_false.insert(a.clone());
                    }
                    2 => {
                        core.must_be_true.insert(a.clone());
                    }
                    _ => {}
                }
                code /= 3;
            }
            let condition = core.condition().unwrap();
            for order in 0..limit {
                let world = |ordinal: usize| {
                    boundary
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| (ordinal ^ order) & (1 << i) != 0)
                        .map(|(_, a)| a.clone())
                        .collect::<State>()
                };
                for ordinal in 0..limit {
                    if !condition.holds(&world(ordinal)) {
                        continue;
                    }
                    let next = after_forbidden_cube(ordinal, &boundary, &core);
                    assert!(next > ordinal && next <= limit);
                    for skipped in ordinal..next {
                        assert!(condition.holds(&world(skipped)));
                    }
                    if next < limit {
                        assert!(!condition.holds(&world(next)));
                    }
                }
            }
        }
    }

    #[test]
    fn core_index_retires_subsumed_cuts_but_keeps_the_proof_archive() {
        let make = |lits: &[usize], incoming: &[usize]| VerifiedConflictCore {
            initial_network: 0,
            partition: 3,
            initial_state: State::new(),
            incoming: Some(BoundaryState {
                must_be_false: incoming.iter().map(|&i| atom(i)).collect(),
                ..Default::default()
            }),
            source_goal: BoundaryState {
                must_be_true: [atom(0), atom(1)].into_iter().collect(),
                ..Default::default()
            }
            .condition()
            .unwrap(),
            core: BoundaryState {
                must_be_true: lits.iter().map(|&i| atom(i)).collect(),
                ..Default::default()
            },
            exactness: ExactnessMode::Unbounded,
            proof: ProofStatus::Infeasible,
            source_call: 0,
            verification_call: 0,
            removed_literals: 2 - lits.len(),
        };
        let mut archive = vec![make(&[0, 1], &[0, 1])];
        let mut index = CoreIndex::default();
        assert_eq!(index.insert(&archive, 0), (true, 0));
        archive.push(make(&[0], &[0]));
        assert_eq!(index.insert(&archive, 1), (true, 1));
        archive.push(make(&[0, 1], &[0, 1]));
        assert_eq!(index.insert(&archive, 2), (false, 0));
        assert_eq!(archive.len(), 3);
        assert_eq!(index.active, 1);
        let goal = archive[0].source_goal.clone();
        assert_eq!(
            index.matching(
                &archive,
                0,
                3,
                &State::new(),
                &goal,
                ExactnessMode::Unbounded
            ),
            Some(1)
        );
        assert!(
            index
                .matching(
                    &archive,
                    1,
                    3,
                    &State::new(),
                    &goal,
                    ExactnessMode::Unbounded
                )
                .is_none()
        );
        assert!(
            index
                .matching(
                    &archive,
                    0,
                    4,
                    &State::new(),
                    &goal,
                    ExactnessMode::Unbounded
                )
                .is_none()
        );
        let mut world = State::new();
        world.insert(atom(0));
        assert!(
            index
                .matching(&archive, 0, 3, &world, &goal, ExactnessMode::Unbounded)
                .is_none()
        );
    }
}
