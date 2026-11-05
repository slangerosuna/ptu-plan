use super::*;
use crate::analysis::{HtnAnalysis, NetworkAnalysis};

impl InteractionGraph {
    /// Build all domain schemas and method bodies, retaining reachability flags
    /// so callers can select problem-relevant nodes without losing the domain.
    pub fn from_analysis(analysis: &HtnAnalysis) -> Self {
        let mut graph = Self::default();
        let task_nodes: Vec<_> = analysis
            .tasks
            .iter()
            .map(|task| {
                graph.push_node(
                    task.name.clone(),
                    NodeKind::TaskSchema {
                        task: task.id,
                        kind: task.kind,
                    },
                    analysis.reachable_tasks.contains(&task.id),
                    task.summary.clone(),
                )
            })
            .collect();
        for method in &analysis.methods {
            let reachable = analysis.reachable_tasks.contains(&method.task);
            let node = graph.push_node(
                method.name.clone(),
                NodeKind::Method {
                    method: method.id,
                    task: method.task,
                },
                reachable,
                method.summary.clone(),
            );
            graph.push_edge(
                task_nodes[method.task.0],
                node,
                EdgeKind::Hierarchy {
                    relation: HierarchyRelation::Decomposes,
                },
            );
            graph.add_network(
                analysis,
                &method.network,
                Some(node),
                &task_nodes,
                reachable,
            );
        }
        if let Some(initial) = &analysis.initial_network {
            graph.add_network(analysis, initial, None, &task_nodes, true);
        }
        graph
    }

    fn push_node(
        &mut self,
        label: String,
        kind: NodeKind,
        reachable: bool,
        summary: TaskSummary,
    ) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(Node {
            id,
            label,
            kind,
            reachable,
            summary,
        });
        id
    }

    fn push_edge(&mut self, source: NodeId, target: NodeId, kind: EdgeKind) {
        self.edges.push(Edge {
            source,
            target,
            kind,
        });
    }

    fn add_network(
        &mut self,
        analysis: &HtnAnalysis,
        network: &NetworkAnalysis,
        method: Option<NodeId>,
        task_nodes: &[NodeId],
        reachable: bool,
    ) {
        let nodes: Vec<_> = network
            .occurrences
            .iter()
            .map(|child| {
                let task = &analysis.tasks[child.task.0];
                let node = self.push_node(
                    child
                        .label
                        .clone()
                        .unwrap_or_else(|| format!("{}[{}]", task.name, child.index)),
                    NodeKind::Occurrence {
                        network: network.id,
                        index: child.index,
                        task: child.task,
                        terms: child.terms.clone(),
                    },
                    reachable,
                    task.summary.clone(),
                );
                if let Some(method) = method {
                    self.push_edge(
                        method,
                        node,
                        EdgeKind::Hierarchy {
                            relation: HierarchyRelation::Contains,
                        },
                    );
                }
                self.push_edge(
                    node,
                    task_nodes[child.task.0],
                    EdgeKind::Hierarchy {
                        relation: HierarchyRelation::InstanceOf,
                    },
                );
                node
            })
            .collect();
        for order in &network.precedence {
            self.push_edge(nodes[order.before], nodes[order.after], EdgeKind::Ordering);
        }
        for (left, &source) in nodes.iter().enumerate() {
            let source_summary = &analysis.tasks[network.occurrences[left].task.0].summary;
            for (right, &target) in nodes.iter().enumerate() {
                if left == right {
                    continue;
                }
                let target_summary = &analysis.tasks[network.occurrences[right].task.0].summary;
                for effect in &source_summary.effects {
                    if target_summary.conditions.contains(effect) {
                        self.push_edge(
                            source,
                            target,
                            EdgeKind::Support {
                                literal: effect.clone(),
                            },
                        );
                    }
                    let complement = effect.complement();
                    for (literals, kind) in [
                        (&target_summary.conditions, ThreatTarget::Condition),
                        (&target_summary.effects, ThreatTarget::Effect),
                    ] {
                        if literals.contains(&complement) {
                            self.push_edge(
                                source,
                                target,
                                EdgeKind::Threat {
                                    literal: complement.clone(),
                                    target: kind,
                                },
                            );
                        }
                    }
                }
                if left < right {
                    let source_fluents = source_summary.fluent_families();
                    let target_fluents = target_summary.fluent_families();
                    for predicate in source_fluents.intersection(&target_fluents) {
                        self.push_edge(
                            source,
                            target,
                            EdgeKind::SharedFluent {
                                predicate: (*predicate).into(),
                            },
                        );
                    }
                }
            }
        }
    }
}
