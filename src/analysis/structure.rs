use std::collections::{BTreeMap, BTreeSet};

use super::*;
use crate::parsing::json::{Htn, Symbol, TaskOrdering};

pub(super) fn analyze_network(
    id: NetworkId,
    parameters: Vec<Symbol>,
    network: &Htn,
    names: &BTreeMap<String, TaskId>,
    tasks: &[TaskAnalysis],
) -> Result<NetworkAnalysis, AnalysisError> {
    let fail = |message| AnalysisError(format!("network {id:?}: {message}"));
    let mut labels = BTreeMap::new();
    let mut occurrences = Vec::new();
    for (index, subtask) in network.subtasks.iter().enumerate() {
        let task = *names
            .get(&subtask.task.name)
            .ok_or_else(|| fail(format!("unknown task '{}'", subtask.task.name)))?;
        if subtask.terms.len() != tasks[task.0].parameters.len() {
            return Err(fail(format!(
                "task '{}' expects {} arguments, got {}",
                subtask.task.name,
                tasks[task.0].parameters.len(),
                subtask.terms.len()
            )));
        }
        let label = subtask.id.as_ref().map(|symbol| symbol.name.clone());
        if let Some(label) = &label
            && labels.insert(label.clone(), index).is_some()
        {
            return Err(fail(format!("duplicate subtask label '{label}'")));
        }
        occurrences.push(TaskOccurrence {
            index,
            label,
            task,
            terms: subtask.terms.clone(),
        });
    }

    let size = occurrences.len();
    let precedence: BTreeSet<_> = match &network.orderings {
        TaskOrdering::Total => (1..size)
            .map(|after| Precedence {
                before: after - 1,
                after,
            })
            .collect(),
        TaskOrdering::Partial(edges) => {
            let mut precedence = BTreeSet::new();
            for (before, after) in edges {
                let resolve = |label: &str| {
                    labels.get(label).copied().ok_or_else(|| {
                        fail(format!(
                            "ordering references unknown subtask label '{label}'"
                        ))
                    })
                };
                precedence.insert(Precedence {
                    before: resolve(before)?,
                    after: resolve(after)?,
                });
            }
            precedence
        }
    };

    // Dense closure is intentionally simple for the scaffold. It also detects
    // cycles in explicit partial orders, including self edges.
    let mut closure = vec![vec![false; size]; size];
    for edge in &precedence {
        closure[edge.before][edge.after] = true;
    }
    for via in 0..size {
        for from in 0..size {
            if closure[from][via] {
                for to in 0..size {
                    closure[from][to] |= closure[via][to];
                }
            }
        }
    }
    if (0..size).any(|index| closure[index][index]) {
        return Err(fail("cyclic task ordering".into()));
    }
    let transitive_precedence: Vec<_> = closure
        .iter()
        .enumerate()
        .flat_map(|(before, row)| {
            row.iter()
                .enumerate()
                .filter_map(move |(after, ordered)| ordered.then_some(Precedence { before, after }))
        })
        .collect();
    let ordering = if transitive_precedence.is_empty() {
        OrderingClass::Unordered
    } else if transitive_precedence.len() == size * (size - 1) / 2 {
        OrderingClass::Total
    } else {
        OrderingClass::Partial
    };
    Ok(NetworkAnalysis {
        id,
        parameters,
        occurrences,
        precedence: precedence.into_iter().collect(),
        transitive_precedence,
        ordering,
        constraints: network.constraints.clone().unwrap_or_default(),
    })
}

pub(super) fn dependency_graph(
    tasks: &[TaskAnalysis],
    methods: &[MethodAnalysis],
) -> Vec<Vec<usize>> {
    let mut edges = vec![BTreeSet::new(); tasks.len()];
    for method in methods {
        edges[method.task.0].extend(method.network.occurrences.iter().map(|child| child.task.0));
    }
    edges
        .into_iter()
        .map(|children| children.into_iter().collect())
        .collect()
}

pub(super) fn reachable(
    roots: impl Iterator<Item = usize>,
    edges: &[Vec<usize>],
) -> BTreeSet<TaskId> {
    let mut found = BTreeSet::new();
    let mut pending: Vec<_> = roots.collect();
    while let Some(task) = pending.pop() {
        if found.insert(TaskId(task)) {
            pending.extend(&edges[task]);
        }
    }
    found
}

pub(super) fn recursion_groups(
    edges: &[Vec<usize>],
    methods: &[MethodAnalysis],
) -> Vec<RecursionGroup> {
    // Iterative Kosaraju avoids depending on the Rust call stack for deep HTNs.
    let mut visited = vec![false; edges.len()];
    let mut finished = Vec::new();
    for start in 0..edges.len() {
        let mut stack = vec![(start, false)];
        while let Some((task, exiting)) = stack.pop() {
            if exiting {
                finished.push(task);
            } else if !visited[task] {
                visited[task] = true;
                stack.push((task, true));
                stack.extend(edges[task].iter().rev().map(|&child| (child, false)));
            }
        }
    }
    let mut reverse = vec![Vec::new(); edges.len()];
    for (from, children) in edges.iter().enumerate() {
        for &to in children {
            reverse[to].push(from);
        }
    }
    visited.fill(false);
    let mut groups = Vec::new();
    for &start in finished.iter().rev() {
        if visited[start] {
            continue;
        }
        let mut component = Vec::new();
        let mut stack = vec![start];
        while let Some(task) = stack.pop() {
            if !visited[task] {
                visited[task] = true;
                component.push(TaskId(task));
                stack.extend(&reverse[task]);
            }
        }
        component.sort();
        let recursive = component.len() > 1 || edges[start].contains(&start);
        let tail_recursive = recursive
            && methods
                .iter()
                .filter(|method| component.binary_search(&method.task).is_ok())
                .all(|method| {
                    let calls: Vec<_> = method
                        .network
                        .occurrences
                        .iter()
                        .filter(|child| component.binary_search(&child.task).is_ok())
                        .collect();
                    match calls.as_slice() {
                        [] => true,
                        [call] => method.network.occurrences.iter().all(|peer| {
                            peer.index == call.index
                                || method.network.precedes(peer.index, call.index)
                        }),
                        _ => false,
                    }
                });
        groups.push(RecursionGroup {
            tasks: component,
            recursive,
            tail_recursive,
        });
    }
    groups.sort_by_key(|group| group.tasks[0]);
    groups
}

pub(super) fn signature(
    tasks: &[TaskAnalysis],
    methods: &[MethodAnalysis],
    initial: Option<&NetworkAnalysis>,
    reachable: &BTreeSet<TaskId>,
    groups: &[RecursionGroup],
) -> ComplexitySignature {
    let active_methods: Vec<_> = methods
        .iter()
        .filter(|method| reachable.contains(&method.task))
        .collect();
    let groups: Vec<_> = groups
        .iter()
        .filter(|group| group.recursive && reachable.contains(&group.tasks[0]))
        .collect();
    let networks: Vec<_> = active_methods
        .iter()
        .map(|method| &method.network)
        .chain(initial)
        .collect();
    let ordered_pairs = networks
        .iter()
        .map(|net| net.transitive_precedence.len())
        .sum();
    let possible_ordered_pairs = networks
        .iter()
        .map(|net| net.occurrences.len() * net.occurrences.len().saturating_sub(1) / 2)
        .sum();
    let ordering = if ordered_pairs == 0 {
        OrderingClass::Unordered
    } else if ordered_pairs == possible_ordered_pairs {
        OrderingClass::Total
    } else {
        OrderingClass::Partial
    };
    let has_lifted_variables = reachable.iter().any(|id| {
        !tasks[id.0].parameters.is_empty()
            || tasks[id.0]
                .summary
                .formula_features
                .contains(&FormulaFeature::Quantification)
    }) || active_methods
        .iter()
        .any(|method| !method.parameters.is_empty() || method.task_terms.iter().any(is_variable))
        || networks.iter().any(|net| {
            !net.parameters.is_empty()
                || net
                    .occurrences
                    .iter()
                    .flat_map(|child| &child.terms)
                    .any(is_variable)
                || net.constraints.iter().any(|constraint| match constraint {
                    crate::parsing::json::Constraint::Equal(a, b)
                    | crate::parsing::json::Constraint::NotEqual(a, b) => {
                        a.starts_with('?') || b.starts_with('?')
                    }
                })
        });
    ComplexitySignature {
        recursion: if groups.is_empty() {
            RecursionClass::Acyclic
        } else if groups.iter().all(|group| group.tail_recursive) {
            RecursionClass::TailRecursive
        } else {
            RecursionClass::GeneralRecursive
        },
        ordering,
        has_lifted_variables,
        fully_unordered: ordering == OrderingClass::Unordered,
        reachable_tasks: reachable.len(),
        reachable_methods: active_methods.len(),
        largest_recursive_scc: groups
            .iter()
            .map(|group| group.tasks.len())
            .max()
            .unwrap_or(0),
        max_method_branching: reachable
            .iter()
            .map(|id| tasks[id.0].methods.len())
            .max()
            .unwrap_or(0),
        max_network_size: networks
            .iter()
            .map(|net| net.occurrences.len())
            .max()
            .unwrap_or(0),
        ordered_pairs,
        possible_ordered_pairs,
    }
}

fn is_variable(symbol: &Symbol) -> bool {
    symbol.name.starts_with('?')
}
