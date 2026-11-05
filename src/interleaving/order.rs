use super::*;
#[derive(Clone)]
pub(super) struct Ordering {
    pub reach: Vec<Vec<bool>>,
    pub edges: BTreeSet<(usize, usize)>,
}
impl Ordering {
    pub fn new(n: usize) -> Self {
        Self {
            reach: vec![vec![false; n]; n],
            edges: BTreeSet::new(),
        }
    }
    pub fn add(&mut self, a: usize, b: usize) -> bool {
        if a >= self.reach.len() || b >= self.reach.len() || a == b || self.reach[b][a] {
            return false;
        }
        self.edges.insert((a, b));
        if self.reach[a][b] {
            return true;
        }
        let before: Vec<_> = (0..self.reach.len())
            .filter(|&i| i == a || self.reach[i][a])
            .collect();
        let after: Vec<_> = (0..self.reach.len())
            .filter(|&i| i == b || self.reach[b][i])
            .collect();
        for i in before {
            for &j in &after {
                self.reach[i][j] = true;
            }
        }
        true
    }
    pub fn linearization(&self) -> Result<Vec<usize>, SolverError> {
        let n = self.reach.len();
        let mut seen = vec![false; n];
        let mut result = Vec::with_capacity(n);
        for _ in 0..n {
            let next = (0..n)
                .find(|&i| !seen[i] && (0..n).all(|j| !self.reach[j][i] || seen[j]))
                .ok_or_else(|| SolverError("cyclic POCL ordering".into()))?;
            seen[next] = true;
            result.push(next);
        }
        Ok(result
            .into_iter()
            .filter(|&i| i > 0 && i + 1 < n)
            .map(|i| i - 1)
            .collect())
    }
}
pub(super) fn endpoints(endpoint: &causal::CausalEndpoint, n: usize) -> Result<usize, SolverError> {
    use causal::CausalEndpoint::*;
    match endpoint {
        Input => Ok(0),
        Output => Ok(n + 1),
        Step(i) if *i < n => Ok(i + 1),
        _ => Err(SolverError("unknown POCL event endpoint".into())),
    }
}
pub(super) fn endpoint(i: usize, n: usize) -> causal::CausalEndpoint {
    if i == 0 {
        causal::CausalEndpoint::Input
    } else if i == n + 1 {
        causal::CausalEndpoint::Output
    } else {
        causal::CausalEndpoint::Step(i - 1)
    }
}
pub(super) fn add_literal(witness: &mut BoundaryState, literal: &Literal) {
    if literal.positive {
        &mut witness.must_be_true
    } else {
        &mut witness.must_be_false
    }
    .insert(literal.atom.clone());
}
/// Sufficient (not maximal) three-valued entailment for the chosen branch.
pub(super) fn partial_holds(condition: &Condition, witness: &BoundaryState) -> Option<bool> {
    match condition {
        Condition::True => Some(true),
        Condition::False => Some(false),
        Condition::Atom(a) => {
            if witness.must_be_true.contains(a) {
                Some(true)
            } else if witness.must_be_false.contains(a) {
                Some(false)
            } else {
                None
            }
        }
        Condition::Not(c) => partial_holds(c, witness).map(|v| !v),
        Condition::And(cs) => {
            let mut unknown = false;
            for c in cs {
                match partial_holds(c, witness) {
                    Some(false) => return Some(false),
                    None => unknown = true,
                    _ => {}
                }
            }
            (!unknown).then_some(true)
        }
        Condition::Or(cs) => {
            let mut unknown = false;
            for c in cs {
                match partial_holds(c, witness) {
                    Some(true) => return Some(true),
                    None => unknown = true,
                    _ => {}
                }
            }
            (!unknown).then_some(false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn incremental_closure_and_cycle_checks_match_independent_network_closure() {
        let n = 4;
        let edges: Vec<_> = (0..n)
            .flat_map(|a| (0..n).filter(move |&b| a != b).map(move |b| (a, b)))
            .collect();
        for mask in 0..(1usize << edges.len()) {
            let mut order = Ordering::new(n);
            let mut network = GroundNetwork {
                tasks: (0..n)
                    .map(|i| GroundTask {
                        name: format!("t{i}"),
                        arguments: vec![],
                    })
                    .collect(),
                precedence: BTreeSet::new(),
            };
            for (bit, &(a, b)) in edges.iter().enumerate() {
                if mask & (1 << bit) == 0 {
                    continue;
                }
                let mut trial = network.clone();
                trial.precedence.insert((a, b));
                let expected = network_closure(&trial);
                assert_eq!(order.add(a, b), expected.is_ok());
                if let Ok(closure) = expected {
                    network = trial;
                    assert_eq!(order.reach, closure);
                } else {
                    assert_eq!(order.reach, network_closure(&network).unwrap());
                }
            }
        }
    }
}
