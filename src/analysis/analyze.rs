use std::collections::{BTreeMap, BTreeSet};

use super::formulas::summarize;
use super::structure::{analyze_network, dependency_graph, reachable, recursion_groups, signature};
use super::*;
use crate::parsing::json::Program;

/// Validate task references, arities, method heads, and local ordering DAGs;
/// compute conservative summaries to a finite fixed point, including recursion.
/// Type compatibility, variable binding, and formula satisfiability are outside
/// this structural analyzer's validation scope.
pub fn analyze_program(program: &Program) -> Result<HtnAnalysis, AnalysisError> {
    let domain = &program.domain;
    if let Some(problem) = &program.problem
        && !problem.domain_name.is_empty()
        && !problem.domain_name.eq_ignore_ascii_case(&domain.name)
    {
        return Err(AnalysisError(format!(
            "problem '{}' names domain '{}', expected '{}'",
            problem.name, problem.domain_name, domain.name
        )));
    }
    let mut names = BTreeMap::new();
    let mut tasks = Vec::new();
    let schemas = domain
        .actions
        .iter()
        .map(|action| {
            (
                &action.name,
                &action.parameters,
                TaskKind::Primitive,
                summarize(action.preconditions.as_ref(), action.effects.as_ref()),
            )
        })
        .chain(domain.compound_tasks.iter().map(|task| {
            (
                &task.name,
                &task.parameters,
                TaskKind::Compound,
                TaskSummary::default(),
            )
        }));
    for (name, parameters, kind, summary) in schemas {
        let id = TaskId(tasks.len());
        if names.insert(name.clone(), id).is_some() {
            return Err(AnalysisError(format!("duplicate task name '{name}'")));
        }
        tasks.push(TaskAnalysis {
            id,
            name: name.clone(),
            kind,
            parameters: parameters.clone(),
            methods: Vec::new(),
            summary,
        });
    }

    let mut methods = Vec::new();
    let mut method_names = BTreeSet::new();
    for method in &domain.methods {
        let name = &method.name.name;
        if !method_names.insert(name) {
            return Err(AnalysisError(format!("duplicate method name '{name}'")));
        }
        let task = *names.get(&method.task.name).ok_or_else(|| {
            AnalysisError(format!(
                "method '{name}' refines unknown task '{}'",
                method.task.name
            ))
        })?;
        if tasks[task.0].kind != TaskKind::Compound {
            return Err(AnalysisError(format!(
                "method '{name}' refines primitive task '{}'",
                method.task.name
            )));
        }
        if method.task_terms.len() != tasks[task.0].parameters.len() {
            return Err(AnalysisError(format!(
                "method '{name}' head expects {} arguments, got {}",
                tasks[task.0].parameters.len(),
                method.task_terms.len()
            )));
        }
        let id = MethodId(methods.len());
        let network = analyze_network(
            NetworkId::Method(id),
            method.params.clone(),
            &method.tn,
            &names,
            &tasks,
        )
        .map_err(|error| AnalysisError(format!("method '{name}': {error}")))?;
        methods.push(MethodAnalysis {
            id,
            name: name.clone(),
            task,
            parameters: method.params.clone(),
            task_terms: method.task_terms.clone(),
            network,
            summary: summarize(method.precondition.as_ref(), None),
        });
        tasks[task.0].methods.push(id);
    }
    let initial_network = program
        .problem
        .as_ref()
        .and_then(|problem| problem.init_tn.as_ref())
        .map(|initial| {
            analyze_network(
                NetworkId::Initial,
                initial.parameters.clone().unwrap_or_default(),
                &initial.tn,
                &names,
                &tasks,
            )
        })
        .transpose()?;

    // Only unions are added: termination follows from the finite number of
    // predicate families, polarities, and formula feature flags in the input.
    loop {
        let mut changed = false;
        for method in &mut methods {
            let old = method.summary.clone();
            for child in &method.network.occurrences {
                method.summary.extend(&tasks[child.task.0].summary);
            }
            changed |= old != method.summary;
            let parent = &mut tasks[method.task.0].summary;
            let old = parent.clone();
            parent.extend(&method.summary);
            changed |= old != *parent;
        }
        if !changed {
            break;
        }
    }

    let dependencies = dependency_graph(&tasks, &methods);
    let roots: Vec<_> = if program.problem.is_none() {
        (0..tasks.len()).collect()
    } else {
        initial_network
            .iter()
            .flat_map(|net| net.occurrences.iter().map(|child| child.task.0))
            .collect()
    };
    let reachable_tasks = reachable(roots.into_iter(), &dependencies);
    let recursion_groups = recursion_groups(&dependencies, &methods);
    let signature = signature(
        &tasks,
        &methods,
        initial_network.as_ref(),
        &reachable_tasks,
        &recursion_groups,
    );
    let mut warnings = Vec::new();
    for task in &tasks {
        if task.kind == TaskKind::Compound && task.methods.is_empty() {
            warnings.push(format!("compound task '{}' has no methods", task.name));
        }
        if !task.summary.formula_features.is_empty() {
            warnings.push(format!(
                "task '{}' uses {:?}; summaries record possible predicate-family accesses only",
                task.name, task.summary.formula_features
            ));
        }
    }
    if !domain.functions.is_empty() {
        warnings.push(
            "numeric functions are retained in the AST; resource interactions are not inferred"
                .into(),
        );
    }
    Ok(HtnAnalysis {
        instance: Some(program.clone()),
        domain_name: domain.name.clone(),
        problem_name: program.problem.as_ref().map(|problem| problem.name.clone()),
        tasks,
        methods,
        initial_network,
        reachable_tasks,
        recursion_groups,
        signature,
        warnings,
    })
}
