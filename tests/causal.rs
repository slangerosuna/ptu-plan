use ptu_plan::{
    parsing::json::Program,
    solver::{causal::*, *},
};
use serde_json::{Value, json};
use std::collections::BTreeSet;
fn fixture() -> Program {
    serde_json::from_str(include_str!("fixtures/solver.json")).unwrap()
}
fn solved(program: &Program) -> (GroundProblem, LocalRequest, PlanCertificate) {
    let model = ground_reachable_program(program, &GroundingOptions::default()).unwrap();
    let request = LocalRequest {
        network: model.initial_networks[0].network.clone(),
        initial_state: model.initial_state.clone(),
        goal: model.goal.clone(),
    };
    let cert = solve_local(&model, &request, &SearchBudget::default())
        .unwrap()
        .certificate
        .unwrap();
    (model, request, cert)
}
fn orders(n: usize, edges: &BTreeSet<(usize, usize)>) -> Vec<Vec<usize>> {
    fn visit(
        n: usize,
        edges: &BTreeSet<(usize, usize)>,
        prefix: &mut Vec<usize>,
        out: &mut Vec<Vec<usize>>,
    ) {
        if prefix.len() == n {
            out.push(prefix.clone());
            return;
        }
        for i in 0..n {
            if !prefix.contains(&i) && edges.iter().all(|&(a, b)| b != i || prefix.contains(&a)) {
                prefix.push(i);
                visit(n, edges, prefix, out);
                prefix.pop();
            }
        }
    }
    let mut out = Vec::new();
    visit(n, edges, &mut Vec::new(), &mut out);
    out
}
#[test]
fn every_causal_linearization_replays_with_refinement_occurrences_remapped() {
    let (model, request, witness) = solved(&fixture());
    let contract = causal_contract(&model, &request, &witness, 7).unwrap();
    validate_causal_contract(&model, &request, &witness, &contract).unwrap();
    assert!(
        contract
            .input_assumptions
            .must_be_true
            .iter()
            .any(|a| a.predicate == "ready")
    );
    assert!(
        contract
            .input_assumptions
            .must_be_false
            .iter()
            .any(|a| a.predicate == "locked")
    );
    let linearizations = orders(contract.steps.len(), &contract.internal_order);
    assert!(
        linearizations.len() > 1,
        "independent tasks should retain partial order"
    );
    for order in linearizations {
        certify_causal_linearization(
            &model,
            &request,
            &witness,
            &contract,
            &order,
            &request.initial_state,
        )
        .unwrap();
    }
    // The imports suffice under arbitrary assignments to irrelevant initial facts.
    let mut changed = request.clone();
    changed.initial_state.insert(GroundAtom {
        predicate: "done".into(),
        arguments: vec!["a".into()],
    });
    let order: Vec<_> = (0..contract.steps.len()).collect();
    certify_causal_linearization(
        &model,
        &request,
        &witness,
        &contract,
        &order,
        &changed.initial_state,
    )
    .unwrap();
    for bad_order in [
        vec![],
        vec![0; order.len()],
        order.iter().rev().copied().collect(),
        vec![usize::MAX; order.len()],
    ] {
        assert!(
            certify_causal_linearization(
                &model,
                &request,
                &witness,
                &contract,
                &bad_order,
                &request.initial_state
            )
            .is_err()
        );
    }
    assert!(
        certify_causal_linearization(&model, &request, &witness, &contract, &order, &State::new())
            .is_err()
    );
    let mut forged = contract.clone();
    forged.imports.clear();
    assert!(validate_causal_contract(&model, &request, &witness, &forged).is_err());
    let mut forged = contract.clone();
    forged.internal_order.clear();
    assert!(validate_causal_contract(&model, &request, &witness, &forged).is_err());
}
fn atom(p: &str) -> Value {
    json!({"Atom":{"name":p,"variables":[]}})
}
#[test]
fn method_guards_negative_support_frame_guarantees_and_threats_are_explicit() {
    let program:Program=serde_json::from_value(json!({"domain":{
        "name":"causal","requirements":["Hierarchy"],"functions":[],
        "predicates":[{"name":"p","variables":[]},{"name":"q","variables":[]},{"name":"frame","variables":[]}],
        "compound_tasks":[{"name":"check","parameters":[]}],
        "methods":[{"name":{"name":"check-p"},"task":{"name":"check"},"task_terms":[],"params":[],"precondition":atom("p"),"tn":{"subtasks":[],"orderings":"Total"}}],
        "actions":[
          {"name":"use","parameters":[],"preconditions":{"And":[atom("p"),{"Not":atom("q")}]},"effects":{"And":[atom("q"),{"Not":atom("p")}] }},
          {"name":"reset","parameters":[],"effects":{"Not":atom("q")}},
          {"name":"finish","parameters":[],"preconditions":{"Not":atom("q")},"effects":atom("q")}
        ]},"problem":{"name":"proof","domain_name":"causal","requirements":[],"objects":[],"init_state":[{"name":"p","variables":[]}],
          "init_tn":{"tn":{"subtasks":(["check","use","reset","finish"].map(|n|json!({"task":{"name":n},"terms":[]}))),"orderings":"Total"}},
          "goal":{"And":[atom("q"),{"Not":atom("frame")}]}}})).unwrap();
    let (model, request, witness) = solved(&program);
    let contract = causal_contract(&model, &request, &witness, 0).unwrap();
    assert_eq!(
        contract.steps[0].decision.kind,
        DecisionKind::Refine { method: 0 }
    );
    assert!(
        contract
            .input_assumptions
            .must_be_true
            .iter()
            .any(|a| a.predicate == "p")
    );
    assert!(
        contract
            .preserved
            .must_be_false
            .iter()
            .any(|a| a.predicate == "frame")
    );
    assert!(
        contract
            .exports
            .must_be_false
            .iter()
            .any(|a| a.predicate == "p")
    );
    assert!(
        contract
            .causal_links
            .iter()
            .any(|l| l.producer == CausalEndpoint::Step(2)
                && l.consumer == CausalEndpoint::Step(3)
                && !l.literal.positive)
    );
    assert!(!contract.threat_resolutions.is_empty());
    assert!(
        contract
            .threat_resolutions
            .iter()
            .any(|t| t.before_producer)
    );
    assert!(
        contract
            .threat_resolutions
            .iter()
            .any(|t| !t.before_producer)
    );
    let mut bad = contract.clone();
    bad.threat_resolutions.clear();
    assert!(validate_causal_contract(&model, &request, &witness, &bad).is_err());
    for order in orders(contract.steps.len(), &contract.internal_order) {
        let decisions: Vec<_> = order
            .iter()
            .map(|&i| contract.steps[i].decision.clone())
            .collect();
        certify_decisions(&model, &request, &decisions).unwrap();
    }
}
#[test]
fn disjunctive_and_nested_negative_conditions_choose_sufficient_witnesses() {
    let mut program = fixture();
    // JSON replacement keeps this test independent of the parser enum layout.
    let mut value = serde_json::to_value(&program).unwrap();
    value["domain"]["actions"][0]["preconditions"] = json!({"Or":[
        {"And":[{"Atom":{"name":"ready","variables":[{"name":"?x"}]}},{"Not":{"Atom":{"name":"locked","variables":[{"name":"?x"}]}}}]},
        {"Not":{"Not":{"Atom":{"name":"locked","variables":[{"name":"?x"}]}}}}
    ]});
    program = serde_json::from_value(value).unwrap();
    let (model, request, witness) = solved(&program);
    let contract = causal_contract(&model, &request, &witness, 0).unwrap();
    validate_causal_contract(&model, &request, &witness, &contract).unwrap();
    for step in &contract.steps {
        assert!(
            step.requirements
                .condition()
                .unwrap()
                .holds(&request.initial_state)
                || !step.add.is_empty()
        );
    }
}

#[test]
fn interleaving_contracts_name_import_consumers_and_export_producers() {
    let program: Program =
        serde_json::from_str(include_str!("fixtures/interleaving.json")).unwrap();
    let model = ground_reachable_program(&program, &GroundingOptions::default()).unwrap();
    let mut contracts = Vec::new();
    for (partition, imported, exported) in [(0, "q", "p"), (1, "p", "q")] {
        let request = LocalRequest {
            network: model.initial_networks[0]
                .network
                .induced(&[partition])
                .unwrap(),
            initial_state: [GroundAtom {
                predicate: imported.into(),
                arguments: vec![],
            }]
            .into_iter()
            .collect(),
            goal: Condition::True,
        };
        let witness = solve_local(&model, &request, &SearchBudget::default())
            .unwrap()
            .certificate
            .unwrap();
        let contract = causal_contract(&model, &request, &witness, partition).unwrap();
        let import = contract
            .imports
            .iter()
            .map(|&i| &contract.causal_links[i])
            .find(|l| l.literal.atom.predicate == imported)
            .unwrap();
        assert!(matches!(import.consumer, CausalEndpoint::Step(_)));
        assert!(contract.causal_links.iter().any(|l| {
            l.literal.atom.predicate == exported
                && l.consumer == CausalEndpoint::Output
                && matches!(l.producer, CausalEndpoint::Step(_))
        }));
        assert!(
            !contract
                .input_assumptions
                .condition()
                .unwrap()
                .holds(&model.initial_state)
        );
        contracts.push(contract);
    }
    // Both local contracts depend on external support. Their certificates alone
    // do not establish global feasibility; a separate global witness does.
    let global = solve_problem(&model, &SearchBudget::default())
        .unwrap()
        .certificate
        .unwrap();
    validate_problem_certificate(&model, &global).unwrap();
    assert_eq!(contracts.len(), 2);
}
