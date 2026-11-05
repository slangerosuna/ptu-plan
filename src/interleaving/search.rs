use super::*;
use std::time::Instant;

struct Search<'a> {
    problem: &'a GroundProblem,
    partitions: &'a [RootPartition],
    options: &'a InterleavingOptions,
    start: Instant,
    stats: InterleavingStatistics,
    uncertain: bool,
    halted: bool,
}
impl Search<'_> {
    fn running(&mut self) -> bool {
        if self.halted {
            return false;
        }
        if self.start.elapsed().as_millis() >= self.options.max_time_ms as u128 {
            self.cutoff();
            self.halted = true;
            return false;
        }
        true
    }
    fn tick(&mut self) -> bool {
        if !self.running() {
            return false;
        }
        if self.stats.nodes >= self.options.max_nodes {
            self.cutoff();
            self.halted = true;
            return false;
        }
        self.stats.nodes += 1;
        true
    }
    fn cutoff(&mut self) {
        self.uncertain = true;
        self.stats.resource_cutoffs += 1;
    }
    fn push<T>(&mut self, frontier: &mut Vec<T>, node: T, other_pending: usize) {
        if !self.running() {
            return;
        }
        if self.stats.generated >= self.options.max_generated {
            self.cutoff();
            self.halted = true;
            return;
        }
        if frontier.len().saturating_add(other_pending) >= self.options.max_frontier {
            self.cutoff();
            return;
        }
        frontier.push(node);
        self.stats.generated += 1;
        self.stats.peak_frontier = self
            .stats
            .peak_frontier
            .max(frontier.len().saturating_add(other_pending));
    }
    fn edge(&mut self, node: &mut PartialPlan, a: usize, b: usize) -> bool {
        if !node.order.add(a, b) {
            self.stats.cycle_rejections += 1;
            return false;
        }
        true
    }
    fn reconcile(
        &mut self,
        fixed: &FixedPlan,
        other_pending: usize,
    ) -> Result<Option<InterleavingCertificate>, SolverError> {
        let n = fixed.proposal.steps.len();
        let mut order = order::Ordering::new(n + 2);
        for &(a, b) in &fixed.base {
            if !order.add(a, b) {
                self.stats.cycle_rejections += 1;
                return Ok(None);
            }
        }
        let mut open: Vec<_> = fixed
            .guards
            .iter()
            .enumerate()
            .map(|(i, c)| Obligation {
                consumer: i + 1,
                condition: c.clone(),
                positive: true,
            })
            .collect();
        open.push(Obligation {
            consumer: n + 1,
            condition: self.problem.goal.clone(),
            positive: true,
        });
        let mut frontier = Vec::new();
        self.push(
            &mut frontier,
            PartialPlan {
                order,
                open,
                links: Vec::new(),
            },
            other_pending,
        );
        while let Some(mut node) = frontier.pop() {
            if !self.tick() {
                break;
            }
            self.stats.causal_nodes += 1;
            // Resolve threats before spending more work on open conditions.
            let threat = node.links.iter().find_map(|link| {
                fixed
                    .clobberers
                    .get(&link.literal)
                    .and_then(|events| {
                        events.iter().copied().find(|&t| {
                            t != link.consumer
                                && !node.order.reach[t][link.producer]
                                && !node.order.reach[link.consumer][t]
                        })
                    })
                    .map(|t| (link.producer, link.consumer, t))
            });
            if let Some((producer, consumer, threat)) = threat {
                // Promotion first in deterministic DFS; both resolutions retain
                // completeness. Start/end boundaries naturally reject cycles.
                for (a, b) in [(threat, producer), (consumer, threat)] {
                    let mut child = node.clone();
                    self.stats.threat_choices += 1;
                    if self.edge(&mut child, a, b) {
                        self.push(&mut frontier, child, other_pending);
                    }
                }
                continue;
            }
            let Some(obligation) = node.open.pop() else {
                let linearization = node.order.linearization()?;
                self.stats.replay_checks += 1;
                let plan = structure::replay(self.problem, &fixed.proposal, &linearization)?;
                let mut guards = vec![BoundaryState::default(); n];
                let mut goal = BoundaryState::default();
                let links = node
                    .links
                    .iter()
                    .map(|link| {
                        if link.consumer == n + 1 {
                            order::add_literal(&mut goal, &link.literal);
                        } else {
                            order::add_literal(&mut guards[link.consumer - 1], &link.literal);
                        }
                        CausalLink {
                            producer: order::endpoint(link.producer, n),
                            consumer: order::endpoint(link.consumer, n),
                            literal: link.literal.clone(),
                        }
                    })
                    .collect();
                let certificate = InterleavingCertificate {
                    schema_version: 1,
                    initial_network: fixed.proposal.initial,
                    exactness: self.options.exactness,
                    partitions: self.partitions.to_vec(),
                    steps: fixed
                        .proposal
                        .steps
                        .iter()
                        .enumerate()
                        .map(|(i, s)| PlanStep {
                            partition: s.partition,
                            task: s.task.clone(),
                            decision: Decision {
                                occurrence: i,
                                kind: s.kind.clone().unwrap(),
                            },
                            children: s.children.clone(),
                        })
                        .collect(),
                    ordering: node.order.edges,
                    causal_links: links,
                    guard_witnesses: guards,
                    goal_witness: goal,
                    linearization,
                    plan,
                    local_sources: None,
                };
                validate_interleaving_certificate(self.problem, &certificate)?;
                return Ok(Some(certificate));
            };
            let Obligation {
                consumer,
                condition,
                positive,
            } = obligation;
            let conjunctive = matches!(condition, Condition::And(_));
            match condition {
                Condition::True | Condition::False => {
                    if matches!(condition, Condition::True) == positive {
                        self.push(&mut frontier, node, other_pending);
                    } else {
                        self.stats.unsupported_conditions += 1;
                    }
                }
                Condition::Not(inner) => {
                    node.open.push(Obligation {
                        consumer,
                        condition: *inner,
                        positive: !positive,
                    });
                    self.push(&mut frontier, node, other_pending);
                }
                Condition::And(cs) | Condition::Or(cs) => {
                    let all = conjunctive == positive;
                    if all {
                        node.open.extend(cs.into_iter().map(|condition| Obligation {
                            consumer,
                            condition,
                            positive,
                        }));
                        self.push(&mut frontier, node, other_pending);
                    } else {
                        if cs.is_empty() {
                            self.stats.unsupported_conditions += 1;
                        }
                        for condition in cs.into_iter().rev() {
                            if !self.running() {
                                break;
                            }
                            self.stats.condition_choices += 1;
                            let mut child = node.clone();
                            child.open.push(Obligation {
                                consumer,
                                condition,
                                positive,
                            });
                            self.push(&mut frontier, child, other_pending);
                        }
                    }
                }
                Condition::Atom(atom) => {
                    let literal = Literal { atom, positive };
                    if node
                        .links
                        .iter()
                        .any(|l| l.consumer == consumer && l.literal == literal)
                    {
                        self.push(&mut frontier, node, other_pending);
                        continue;
                    }
                    let mut producers = fixed.writers.get(&literal).cloned().unwrap_or_default();
                    if self.problem.initial_state.contains(&literal.atom) == literal.positive {
                        producers.insert(0, 0);
                    }
                    let mut supported = false;
                    for producer in producers.into_iter().rev() {
                        if !self.running() {
                            break;
                        }
                        if producer == consumer {
                            continue;
                        }
                        let mut child = node.clone();
                        self.stats.support_choices += 1;
                        if !self.edge(&mut child, producer, consumer) {
                            continue;
                        }
                        supported = true;
                        if producer > 0
                            && consumer <= n
                            && fixed.proposal.steps[producer - 1].partition
                                != fixed.proposal.steps[consumer - 1].partition
                        {
                            self.stats.cross_partition_links += 1;
                        }
                        child.links.push(Link {
                            producer,
                            consumer,
                            literal: literal.clone(),
                        });
                        self.push(&mut frontier, child, other_pending);
                    }
                    if !supported {
                        self.stats.unsupported_conditions += 1;
                    }
                }
            }
        }
        Ok(None)
    }
}

/// Enumerate method choices independently of current guards, then reconcile
/// event-local requirements by causal support and threat-protection orderings.
/// Exhaustion is global only when every grounding, refinement and POCL branch
/// was exhausted without resource cutoffs (or within the explicit depth bound).
pub fn solve_interleaved(
    problem: &GroundProblem,
    partitions: &[RootPartition],
    options: &InterleavingOptions,
) -> Result<InterleavingResult, SolverError> {
    if options.max_events > 4096 {
        return Err(SolverError(
            "POCL dense ordering supports at most 4096 events; choose a smaller explicit event cap"
                .into(),
        ));
    }
    problem.context.check_condition(&problem.goal)?;
    let mut search = Search {
        problem,
        partitions,
        options,
        start: Instant::now(),
        stats: InterleavingStatistics::default(),
        uncertain: false,
        halted: false,
    };
    let mut certificate = None;
    'initial: for (initial, network) in problem.initial_networks.iter().enumerate() {
        let owners = structure::owners(&network.network, partitions)?;
        if network.network.tasks.len() > options.max_events {
            search.cutoff();
            continue;
        }
        let proposal = Proposal {
            initial,
            steps: network
                .network
                .tasks
                .iter()
                .cloned()
                .zip(owners)
                .map(|(task, owner)| structure::proposed(problem, task, owner, 0))
                .collect(),
        };
        let mut frontier = Vec::new();
        search.push(&mut frontier, proposal, 0);
        while let Some(proposal) = frontier.pop() {
            if !search.tick() {
                break 'initial;
            }
            search.stats.structural_nodes += 1;
            let Some(pending) = proposal.steps.iter().position(|s| s.kind.is_none()) else {
                search.stats.complete_decompositions += 1;
                let fixed = structure::compile(problem, proposal)?;
                if let Some(found) = search.reconcile(&fixed, frontier.len())? {
                    certificate = Some(found);
                    break 'initial;
                }
                if search.halted {
                    break 'initial;
                }
                continue;
            };
            let step = &proposal.steps[pending];
            if let ExactnessMode::Bounded {
                max_decomposition_depth,
            } = options.exactness
                && step.depth >= max_decomposition_depth
            {
                search.stats.depth_cutoffs += 1;
                continue;
            }
            let Some(methods) = problem.methods_by_task.get(&step.task) else {
                continue;
            };
            for &method in methods.iter().rev() {
                if !search.running() {
                    break;
                }
                let body = &problem.methods[method].network;
                if proposal.steps.len().saturating_add(body.tasks.len()) > options.max_events {
                    search.cutoff();
                    continue;
                }
                let mut child = proposal.clone();
                let start = child.steps.len();
                child.steps[pending].kind = Some(DecisionKind::Refine { method });
                child.steps[pending].children = (start..start + body.tasks.len()).collect();
                child.steps.extend(body.tasks.iter().cloned().map(|task| {
                    structure::proposed(problem, task, step.partition, step.depth + 1)
                }));
                search.stats.local_refinements += 1;
                search.push(&mut frontier, child, 0);
            }
        }
        if search.halted {
            break;
        }
    }
    Ok(search.finish(certificate, SearchScope::AllDecompositions))
}

impl Search<'_> {
    fn finish(
        mut self,
        certificate: Option<InterleavingCertificate>,
        scope: SearchScope,
    ) -> InterleavingResult {
        let status = if certificate.is_some() {
            ProofStatus::Feasible
        } else if self.uncertain {
            ProofStatus::BudgetExhausted
        } else if self.stats.depth_cutoffs > 0 {
            ProofStatus::InfeasibleWithinBound
        } else {
            ProofStatus::Infeasible
        };
        self.stats.elapsed_ms = self.start.elapsed().as_millis();
        InterleavingResult {
            status,
            exactness: self.options.exactness,
            scope,
            options: *self.options,
            certificate,
            statistics: self.stats,
        }
    }
}

/// Reconcile supplied local certificates, retaining their chosen methods and
/// certified internal ordering. Original local output goals are not additional
/// global/boundary goals; event effects supply the actual consumer obligations.
/// Infeasibility is scoped to this fixed bundle;
/// callers must not infer that every decomposition of the problem is impossible.
pub fn reconcile_local_certificates(
    problem: &GroundProblem,
    initial_network: usize,
    sources: &[crate::master::LocalSegment],
    options: &InterleavingOptions,
) -> Result<InterleavingResult, SolverError> {
    if options.max_events > 4096 {
        return Err(SolverError(
            "POCL dense ordering supports at most 4096 events".into(),
        ));
    }
    let partitions: Vec<_> = sources.iter().map(|s| s.partition.clone()).collect();
    problem.context.check_condition(&problem.goal)?;
    let mut search = Search {
        problem,
        partitions: &partitions,
        options,
        start: Instant::now(),
        stats: InterleavingStatistics::default(),
        uncertain: false,
        halted: false,
    };
    let (proposal, extra) = structure::from_local_sources(problem, initial_network, sources)?;
    if let ExactnessMode::Bounded {
        max_decomposition_depth,
    } = options.exactness
        && proposal.steps.iter().any(|s| {
            matches!(s.kind, Some(DecisionKind::Refine { .. }))
                && s.depth >= max_decomposition_depth
        })
    {
        return Err(SolverError(
            "supplied POCL certificate exceeds requested decomposition bound".into(),
        ));
    }
    if proposal.steps.len() > options.max_events {
        search.cutoff();
        return Ok(search.finish(None, SearchScope::SuppliedLocalCertificates));
    }
    let mut fixed = structure::compile(problem, proposal)?;
    fixed.base.extend(extra);
    let mut certificate = search.reconcile(&fixed, 0)?;
    if let Some(certificate) = &mut certificate {
        certificate.local_sources = Some(sources.to_vec());
        validate_interleaving_certificate(problem, certificate)?;
    }
    Ok(search.finish(certificate, SearchScope::SuppliedLocalCertificates))
}
