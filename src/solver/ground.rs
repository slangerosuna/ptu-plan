use super::*;
use crate::analysis::{NetworkAnalysis, analyze_program};
use crate::parsing::json::{Constraint, Formula, Predicate, Program, RequirementType, Symbol};

#[derive(Debug, Clone)]
pub(crate) struct Context {
    types: BTreeMap<String, BTreeSet<String>>,
    objects: BTreeMap<String, String>,
    predicates: BTreeMap<String, Vec<Symbol>>,
    tasks: BTreeMap<String, Vec<Symbol>>,
    task_names: Vec<String>,
}

struct Counter {
    used: usize,
    limit: usize,
}

impl Counter {
    fn charge(&mut self) -> Result<(), SolverError> {
        if self.used >= self.limit {
            return Err(SolverError(
                "grounding budget exhausted; no infeasibility conclusion is available".into(),
            ));
        }
        self.used += 1;
        Ok(())
    }
    fn remaining(&self) -> usize {
        self.limit - self.used
    }
}

impl Context {
    fn new(program: &Program) -> Result<Self, SolverError> {
        let domain = &program.domain;
        let mut ctx = Self {
            types: BTreeMap::from([("object".into(), BTreeSet::new())]),
            objects: BTreeMap::new(),
            predicates: BTreeMap::new(),
            tasks: BTreeMap::new(),
            task_names: Vec::new(),
        };
        for ty in domain.types.iter().flatten() {
            if ty.name.eq_ignore_ascii_case("object") && ty.symbol_type.is_none() {
                continue;
            }
            ctx.types
                .entry(ty.name.to_ascii_lowercase())
                .or_default()
                .insert(
                    ty.symbol_type
                        .as_deref()
                        .unwrap_or("object")
                        .to_ascii_lowercase(),
                );
        }
        // A supertype mentioned on the right of a typed list is itself a type,
        // even without a separate left-hand declaration (e.g. RING - OBJ).
        let parents: Vec<_> = ctx.types.values().flatten().cloned().collect();
        for parent in parents {
            ctx.types
                .entry(parent)
                .or_insert_with(|| BTreeSet::from(["object".into()]));
        }
        for name in ctx.types.keys() {
            let mut visited = BTreeSet::new();
            let mut pending: Vec<_> = ctx.types[name].iter().collect();
            while let Some(ty) = pending.pop() {
                if ty == name {
                    return Err(SolverError(format!("cyclic type hierarchy at '{ty}'")));
                }
                if visited.insert(ty) {
                    pending.extend(ctx.types[ty].iter());
                }
            }
        }
        let problem = program
            .problem
            .as_ref()
            .ok_or_else(|| SolverError("solving requires a problem".into()))?;
        for object in domain.constants.iter().flatten().chain(&problem.objects) {
            let ty = object
                .symbol_type
                .as_deref()
                .unwrap_or("object")
                .to_ascii_lowercase();
            ctx.known_type(&ty)?;
            if object.name.starts_with('?') {
                return Err(SolverError("object names cannot be variables".into()));
            }
            if let Some(previous) = ctx.objects.insert(object.name.clone(), ty.clone())
                && previous != ty
            {
                return Err(SolverError(format!(
                    "conflicting types for object '{}'",
                    object.name
                )));
            }
        }
        for predicate in &domain.predicates {
            ctx.parameters(&predicate.variables)?;
            if ctx
                .predicates
                .insert(predicate.name.clone(), predicate.variables.clone())
                .is_some()
            {
                return Err(SolverError(format!(
                    "duplicate predicate '{}'",
                    predicate.name
                )));
            }
        }
        for (name, parameters) in domain
            .actions
            .iter()
            .map(|a| (&a.name, &a.parameters))
            .chain(
                domain
                    .compound_tasks
                    .iter()
                    .map(|t| (&t.name, &t.parameters)),
            )
        {
            ctx.parameters(parameters)?;
            ctx.tasks.insert(name.clone(), parameters.clone());
            ctx.task_names.push(name.clone());
        }
        Ok(ctx)
    }

    fn known_type(&self, ty: &str) -> Result<(), SolverError> {
        if self.types.contains_key(&ty.to_ascii_lowercase()) {
            Ok(())
        } else {
            Err(SolverError(format!("unknown type '{ty}'")))
        }
    }

    fn subtype(&self, child: &str, parent: &str) -> bool {
        let parent = parent.to_ascii_lowercase();
        let mut pending = vec![child.to_ascii_lowercase()];
        let mut seen = BTreeSet::new();
        while let Some(child) = pending.pop() {
            if child == parent {
                return true;
            }
            if seen.insert(child.clone()) {
                pending.extend(self.types.get(&child).into_iter().flatten().cloned());
            }
        }
        false
    }

    fn parameters(&self, parameters: &[Symbol]) -> Result<(), SolverError> {
        let mut seen = BTreeSet::new();
        for parameter in parameters {
            self.known_type(parameter.symbol_type.as_deref().unwrap_or("object"))?;
            if !parameter.name.starts_with('?') || !seen.insert(&parameter.name) {
                return Err(SolverError(format!(
                    "invalid or duplicate parameter '{}'",
                    parameter.name
                )));
            }
        }
        Ok(())
    }

    fn assignments(
        &self,
        parameters: &[Symbol],
        limit: usize,
    ) -> Result<Vec<Bindings>, SolverError> {
        self.parameters(parameters)?;
        let mut rows = vec![Bindings::new()];
        for parameter in parameters {
            let ty = parameter.symbol_type.as_deref().unwrap_or("object");
            let candidates: Vec<_> = self
                .objects
                .iter()
                .filter(|(_, actual)| self.subtype(actual, ty))
                .map(|(name, _)| name)
                .collect();
            let mut next = Vec::new();
            for row in &rows {
                for object in &candidates {
                    if next.len() >= limit {
                        return Err(SolverError(
                            "grounding budget exhausted while enumerating bindings".into(),
                        ));
                    }
                    let mut bound = row.clone();
                    bound.insert(parameter.name.clone(), (*object).clone());
                    next.push(bound);
                }
            }
            rows = next;
        }
        Ok(rows)
    }

    fn term(&self, name: &str, bindings: &Bindings) -> Result<String, SolverError> {
        let value = if name.starts_with('?') {
            bindings
                .get(name)
                .ok_or_else(|| SolverError(format!("unbound variable '{name}'")))?
                .as_str()
        } else {
            name
        };
        if !self.objects.contains_key(value) {
            return Err(SolverError(format!("unknown object '{value}'")));
        }
        Ok(value.into())
    }

    fn arguments(
        &self,
        terms: &[Symbol],
        parameters: &[Symbol],
        bindings: &Bindings,
    ) -> Result<Vec<String>, SolverError> {
        if terms.len() != parameters.len() {
            return Err(SolverError("predicate/task arity mismatch".into()));
        }
        terms
            .iter()
            .zip(parameters)
            .map(|(term, parameter)| {
                let value = self.term(&term.name, bindings)?;
                let expected = parameter.symbol_type.as_deref().unwrap_or("object");
                if !self.subtype(&self.objects[&value], expected) {
                    return Err(SolverError(format!(
                        "object '{value}' does not have required type '{expected}'"
                    )));
                }
                Ok(value)
            })
            .collect()
    }

    fn atom(&self, predicate: &Predicate, bindings: &Bindings) -> Result<GroundAtom, SolverError> {
        let parameters = self
            .predicates
            .get(&predicate.name)
            .ok_or_else(|| SolverError(format!("unknown predicate '{}'", predicate.name)))?;
        Ok(GroundAtom {
            predicate: predicate.name.clone(),
            arguments: self.arguments(&predicate.variables, parameters, bindings)?,
        })
    }

    fn task(
        &self,
        name: &str,
        terms: &[Symbol],
        bindings: &Bindings,
    ) -> Result<GroundTask, SolverError> {
        let parameters = self
            .tasks
            .get(name)
            .ok_or_else(|| SolverError(format!("unknown task '{name}'")))?;
        Ok(GroundTask {
            name: name.into(),
            arguments: self.arguments(terms, parameters, bindings)?,
        })
    }

    pub(crate) fn check_atom(&self, atom: &GroundAtom) -> Result<(), SolverError> {
        let predicate = Predicate {
            name: atom.predicate.clone(),
            variables: atom
                .arguments
                .iter()
                .map(|name| Symbol {
                    name: name.clone(),
                    symbol_type: None,
                })
                .collect(),
        };
        self.atom(&predicate, &Bindings::new()).map(|_| ())
    }

    pub(crate) fn check_condition(&self, condition: &Condition) -> Result<(), SolverError> {
        match condition {
            Condition::Atom(atom) => self.check_atom(atom),
            Condition::Not(inner) => self.check_condition(inner),
            Condition::And(terms) | Condition::Or(terms) => {
                for term in terms {
                    self.check_condition(term)?;
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }
}

fn validate_formula(formula: &Formula, effect: bool) -> Result<(), SolverError> {
    use Formula::*;
    match formula {
        Empty | Atom(_) => Ok(()),
        Not(inner) if !effect || matches!(inner.as_ref(), Atom(_)) => {
            validate_formula(inner, effect)
        }
        And(terms) => {
            for term in terms {
                validate_formula(term, effect)?;
            }
            Ok(())
        }
        ForAll(_, inner) => validate_formula(inner, effect),
        Exists(_, inner) if !effect => validate_formula(inner, false),
        Or(terms) if !effect => {
            for term in terms {
                validate_formula(term, false)?;
            }
            Ok(())
        }
        Imply(a, b) if !effect => {
            validate_formula(a, false)?;
            validate_formula(b, false)
        }
        Equals(_, _) if !effect => Ok(()),
        _ => Err(SolverError(format!(
            "unsupported {} formula: {formula:?}",
            if effect { "effect" } else { "condition" }
        ))),
    }
}

fn condition(
    ctx: &Context,
    formula: Option<&Formula>,
    env: &Bindings,
    count: &mut Counter,
) -> Result<Condition, SolverError> {
    count.charge()?;
    use Formula::*;
    Ok(match formula {
        None | Some(Empty) => Condition::True,
        Some(Atom(atom)) => Condition::Atom(ctx.atom(atom, env)?),
        Some(Not(inner)) => Condition::Not(Box::new(condition(ctx, Some(inner), env, count)?)),
        Some(And(terms)) => Condition::And(
            terms
                .iter()
                .map(|term| condition(ctx, Some(term), env, count))
                .collect::<Result<_, _>>()?,
        ),
        Some(Or(terms)) => Condition::Or(
            terms
                .iter()
                .map(|term| condition(ctx, Some(term), env, count))
                .collect::<Result<_, _>>()?,
        ),
        Some(Imply(a, b)) => Condition::Or(vec![
            Condition::Not(Box::new(condition(ctx, Some(a), env, count)?)),
            condition(ctx, Some(b), env, count)?,
        ]),
        Some(Equals(a, b)) => {
            if ctx.term(a, env)? == ctx.term(b, env)? {
                Condition::True
            } else {
                Condition::False
            }
        }
        Some(Exists(params, inner) | ForAll(params, inner)) => {
            let mut terms = Vec::new();
            for row in ctx.assignments(params, count.remaining())? {
                let mut bound = env.clone();
                bound.extend(row);
                terms.push(condition(ctx, Some(inner), &bound, count)?);
            }
            if matches!(formula, Some(Exists(_, _))) {
                Condition::Or(terms)
            } else {
                Condition::And(terms)
            }
        }
        Some(other) => {
            return Err(SolverError(format!(
                "unsupported condition formula: {other:?}"
            )));
        }
    })
}

fn effects(
    ctx: &Context,
    formula: Option<&Formula>,
    env: &Bindings,
    count: &mut Counter,
    add: &mut State,
    delete: &mut State,
) -> Result<(), SolverError> {
    count.charge()?;
    match formula {
        None | Some(Formula::Empty) => {}
        Some(Formula::Atom(atom)) => {
            add.insert(ctx.atom(atom, env)?);
        }
        Some(Formula::Not(inner)) => match inner.as_ref() {
            Formula::Atom(atom) => {
                delete.insert(ctx.atom(atom, env)?);
            }
            _ => return Err(SolverError("unsupported nonliteral negative effect".into())),
        },
        Some(Formula::And(terms)) => {
            for term in terms {
                effects(ctx, Some(term), env, count, add, delete)?;
            }
        }
        Some(Formula::ForAll(params, inner)) => {
            for row in ctx.assignments(params, count.remaining())? {
                let mut bound = env.clone();
                bound.extend(row);
                effects(ctx, Some(inner), &bound, count, add, delete)?;
            }
        }
        Some(other) => {
            return Err(SolverError(format!(
                "unsupported effect formula: {other:?}"
            )));
        }
    }
    Ok(())
}

fn network(
    ctx: &Context,
    network: &NetworkAnalysis,
    env: &Bindings,
    count: &mut Counter,
) -> Result<Option<GroundNetwork>, SolverError> {
    for constraint in &network.constraints {
        let (a, b, equal) = match constraint {
            Constraint::Equal(a, b) => (a, b, true),
            Constraint::NotEqual(a, b) => (a, b, false),
        };
        if (ctx.term(a, env)? == ctx.term(b, env)?) != equal {
            return Ok(None);
        }
    }
    let mut tasks = Vec::new();
    for child in &network.occurrences {
        count.charge()?;
        // AST order matches the analyzer's TaskId assignment.
        let name = &ctx.task_names[child.task.0];
        tasks.push(ctx.task(name, &child.terms, env)?);
    }
    Ok(Some(GroundNetwork {
        tasks,
        precedence: network
            .precedence
            .iter()
            .map(|edge| (edge.before, edge.after))
            .collect(),
    }))
}

pub fn ground_program(
    program: &Program,
    options: &GroundingOptions,
) -> Result<GroundProblem, SolverError> {
    compile(program, options, GroundingMode::Full)
}

/// Compile only task instances reachable through some decomposition of an
/// initial root. Method head arguments are unified before enumerating free
/// parameters. Guards are never filtered using the initial world state.
pub fn ground_reachable_program(
    program: &Program,
    options: &GroundingOptions,
) -> Result<GroundProblem, SolverError> {
    compile(program, options, GroundingMode::InitialReachable)
}

fn compile(
    program: &Program,
    options: &GroundingOptions,
    mode: GroundingMode,
) -> Result<GroundProblem, SolverError> {
    let analysis = analyze_program(program).map_err(|error| SolverError(error.to_string()))?;
    let domain = &program.domain;
    let problem = program
        .problem
        .as_ref()
        .ok_or_else(|| SolverError("solving requires a problem".into()))?;
    if !domain.functions.is_empty() {
        return Err(SolverError(
            "numeric functions are unsupported by the exact Boolean solver".into(),
        ));
    }
    for requirement in domain.requirements.iter().chain(&problem.requirements) {
        if matches!(
            requirement,
            RequirementType::ProbabilisticEffects
                | RequirementType::ConditionalEffects
                | RequirementType::DomainAxioms
        ) {
            return Err(SolverError(format!(
                "unsupported solver requirement: {requirement:?}"
            )));
        }
    }
    for (formula, effect) in domain
        .actions
        .iter()
        .flat_map(|a| {
            [
                (a.preconditions.as_ref(), false),
                (a.effects.as_ref(), true),
            ]
        })
        .chain(
            domain
                .methods
                .iter()
                .map(|m| (m.precondition.as_ref(), false)),
        )
        .chain([(problem.goal.as_ref(), false)])
    {
        if let Some(formula) = formula {
            validate_formula(formula, effect)?;
        }
    }
    let ctx = Context::new(program)?;
    let mut count = Counter {
        used: 0,
        limit: options.max_instances,
    };
    if mode == GroundingMode::InitialReachable {
        return reachable_model(program, &analysis, ctx, &mut count);
    }
    let mut tasks = BTreeSet::new();
    for task in &analysis.tasks {
        for bindings in ctx.assignments(&task.parameters, count.remaining())? {
            count.charge()?;
            tasks.insert(ctx.task(&task.name, &task.parameters, &bindings)?);
        }
    }
    let mut actions = BTreeMap::new();
    for action in &domain.actions {
        for bindings in ctx.assignments(&action.parameters, count.remaining())? {
            count.charge()?;
            let task = ctx.task(&action.name, &action.parameters, &bindings)?;
            let precondition =
                condition(&ctx, action.preconditions.as_ref(), &bindings, &mut count)?;
            let (mut add, mut delete) = (State::new(), State::new());
            effects(
                &ctx,
                action.effects.as_ref(),
                &bindings,
                &mut count,
                &mut add,
                &mut delete,
            )?;
            actions.insert(
                task.clone(),
                GroundAction {
                    task,
                    precondition,
                    add,
                    delete,
                },
            );
        }
    }
    let mut methods = Vec::new();
    let mut methods_by_task: BTreeMap<_, Vec<_>> = BTreeMap::new();
    for (schema, analyzed) in domain.methods.iter().zip(&analysis.methods) {
        for bindings in ctx.assignments(&schema.params, count.remaining())? {
            count.charge()?;
            let Some(network) = network(&ctx, &analyzed.network, &bindings, &mut count)? else {
                continue;
            };
            let task = ctx.task(&schema.task.name, &schema.task_terms, &bindings)?;
            let precondition =
                condition(&ctx, schema.precondition.as_ref(), &bindings, &mut count)?;
            let id = methods.len();
            methods_by_task.entry(task.clone()).or_default().push(id);
            methods.push(GroundMethod {
                id,
                name: schema.name.name.clone(),
                bindings,
                task,
                precondition,
                network,
            });
        }
    }
    let mut initial_networks = Vec::new();
    if let Some(initial) = &analysis.initial_network {
        for bindings in ctx.assignments(&initial.parameters, count.remaining())? {
            count.charge()?;
            if let Some(network) = network(&ctx, initial, &bindings, &mut count)? {
                initial_networks.push(InitialNetwork { bindings, network });
            }
        }
    } else {
        initial_networks.push(InitialNetwork {
            bindings: Bindings::new(),
            network: GroundNetwork::default(),
        });
    }
    let initial_state = problem
        .init_state
        .iter()
        .map(|atom| ctx.atom(atom, &Bindings::new()))
        .collect::<Result<_, _>>()?;
    let goal = condition(&ctx, problem.goal.as_ref(), &Bindings::new(), &mut count)?;
    Ok(GroundProblem {
        actions,
        methods,
        initial_networks,
        initial_state,
        goal,
        tasks,
        methods_by_task,
        context: ctx,
        grounding_mode: mode,
    })
}

fn reachable_model(
    program: &Program,
    analysis: &crate::analysis::HtnAnalysis,
    ctx: Context,
    count: &mut Counter,
) -> Result<GroundProblem, SolverError> {
    let mut initial_networks = Vec::new();
    if let Some(initial) = &analysis.initial_network {
        for bindings in ctx.assignments(&initial.parameters, count.remaining())? {
            count.charge()?;
            if let Some(network) = network(&ctx, initial, &bindings, count)? {
                initial_networks.push(InitialNetwork { bindings, network });
            }
        }
    } else {
        initial_networks.push(InitialNetwork {
            bindings: Bindings::new(),
            network: GroundNetwork::default(),
        });
    }
    let problem = program.problem.as_ref().unwrap();
    let initial_state = problem
        .init_state
        .iter()
        .map(|a| ctx.atom(a, &Bindings::new()))
        .collect::<Result<_, _>>()?;
    let goal = condition(&ctx, problem.goal.as_ref(), &Bindings::new(), count)?;
    let mut pending: BTreeSet<_> = initial_networks
        .iter()
        .flat_map(|n| n.network.tasks.iter().cloned())
        .collect();
    let mut tasks = BTreeSet::new();
    let mut actions = BTreeMap::new();
    let mut methods = Vec::new();
    let mut methods_by_task: BTreeMap<_, Vec<_>> = BTreeMap::new();
    while let Some(task) = pending.pop_first() {
        if !tasks.insert(task.clone()) {
            continue;
        }
        count.charge()?;
        if let Some(schema) = program.domain.actions.iter().find(|a| a.name == task.name) {
            let bindings = schema
                .parameters
                .iter()
                .zip(&task.arguments)
                .map(|(p, a)| (p.name.clone(), a.clone()))
                .collect();
            let precondition = condition(&ctx, schema.preconditions.as_ref(), &bindings, count)?;
            let (mut add, mut delete) = (State::new(), State::new());
            effects(
                &ctx,
                schema.effects.as_ref(),
                &bindings,
                count,
                &mut add,
                &mut delete,
            )?;
            actions.insert(
                task.clone(),
                GroundAction {
                    task,
                    precondition,
                    add,
                    delete,
                },
            );
            continue;
        }
        for (schema, analyzed) in program
            .domain
            .methods
            .iter()
            .zip(&analysis.methods)
            .filter(|(m, _)| m.task.name == task.name)
        {
            ctx.parameters(&schema.params)?;
            let mut fixed = Bindings::new();
            let mut matches = true;
            for (term, value) in schema.task_terms.iter().zip(&task.arguments) {
                if term.name.starts_with('?') {
                    let parameter = schema
                        .params
                        .iter()
                        .find(|p| p.name == term.name)
                        .ok_or_else(|| {
                            SolverError(format!("unbound method head variable '{}'", term.name))
                        })?;
                    if !ctx.subtype(
                        &ctx.objects[value],
                        parameter.symbol_type.as_deref().unwrap_or("object"),
                    ) {
                        matches = false;
                        break;
                    }
                    if fixed
                        .insert(term.name.clone(), value.clone())
                        .is_some_and(|previous| previous != *value)
                    {
                        matches = false;
                        break;
                    }
                } else {
                    ctx.term(&term.name, &Bindings::new())?;
                    if term.name != *value {
                        matches = false;
                        break;
                    }
                }
            }
            if !matches {
                continue;
            }
            let free: Vec<_> = schema
                .params
                .iter()
                .filter(|p| !fixed.contains_key(&p.name))
                .cloned()
                .collect();
            for mut bindings in ctx.assignments(&free, count.remaining())? {
                count.charge()?;
                bindings.extend(fixed.clone());
                let Some(network) = network(&ctx, &analyzed.network, &bindings, count)? else {
                    continue;
                };
                let precondition = condition(&ctx, schema.precondition.as_ref(), &bindings, count)?;
                // Include every alternative, even if its guard is currently false.
                pending.extend(
                    network
                        .tasks
                        .iter()
                        .filter(|t| !tasks.contains(*t))
                        .cloned(),
                );
                let id = methods.len();
                methods_by_task.entry(task.clone()).or_default().push(id);
                methods.push(GroundMethod {
                    id,
                    name: schema.name.name.clone(),
                    bindings,
                    task: task.clone(),
                    precondition,
                    network,
                });
            }
        }
    }
    Ok(GroundProblem {
        actions,
        methods,
        initial_networks,
        initial_state,
        goal,
        tasks,
        methods_by_task,
        context: ctx,
        grounding_mode: GroundingMode::InitialReachable,
    })
}
