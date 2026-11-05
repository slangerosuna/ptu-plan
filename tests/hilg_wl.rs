use ptu_plan::{
    agglomeration::{self, AgglomerationOptions},
    analysis::{NetworkId, analyze_program},
    interaction::InteractionGraph,
    learning::{
        features::Observation,
        model::{MergeModel, ModelConfig, Supervision, ValueTarget},
    },
    master::MasterOptions,
    parsing::json::{Formula, Predicate, Program, Symbol},
    partition::{SpectralOptions, micropartition_network},
};
fn fixture() -> Program {
    serde_json::from_str(include_str!("fixtures/solver.json")).unwrap()
}
fn observation(p: &Program) -> Observation {
    let analysis = analyze_program(p).unwrap();
    let graph = InteractionGraph::from_analysis(&analysis);
    let seeds = micropartition_network(
        &graph,
        NetworkId::Initial,
        &SpectralOptions {
            max_cluster_size: 1,
            ..Default::default()
        },
    )
    .unwrap();
    let options = AgglomerationOptions::default();
    let state = agglomeration::observe(
        &analysis,
        &graph,
        NetworkId::Initial,
        &seeds.clusters,
        &options,
    )
    .unwrap();
    let (mut obs, _) = Observation::build(&analysis, &graph, &state, &options).unwrap();
    obs.set_search_context(&MasterOptions::default());
    obs
}
#[test]
fn hilg_wl_gradients_match_finite_differences_and_context_changes_rankings() {
    let obs = observation(&fixture());
    let mut model = MergeModel::new_wl(
        ModelConfig {
            hidden: 1,
            layers: 1,
        },
        19,
    )
    .unwrap();
    let labels = Supervision {
        preferences: vec![(0, 1)],
        values: vec![ValueTarget {
            action: 0,
            log_effort: 2.0,
            lower_bound: false,
        }],
    };
    let (_, gradient) = model.loss_gradient(&obs, &labels, 1.0).unwrap();
    for (index, &analytic) in gradient.iter().enumerate() {
        let old = model.parameters[index];
        let eps = 1e-5;
        model.parameters[index] = old + eps;
        let plus = model.loss(&obs, &labels, 1.0).unwrap();
        model.parameters[index] = old - eps;
        let minus = model.loss(&obs, &labels, 1.0).unwrap();
        model.parameters[index] = old;
        let numeric = (plus - minus) / (2.0 * eps);
        assert!(
            (numeric - analytic).abs() < 2e-6,
            "parameter {index}: numeric {numeric}, analytic {analytic}"
        );
    }
    let mut budget = obs.clone();
    budget.context[2] = 0.0;
    let a = model.predict(&obs).unwrap();
    let b = model.predict(&budget).unwrap();
    assert!(((a.logits[0] - a.logits[1]) - (b.logits[0] - b.logits[1])).abs() > 1e-8);
    let encoded = model.encode(&obs).unwrap();
    let decoded = model.predict_encoded(&obs, &encoded).unwrap();
    assert_eq!(a.logits, decoded.logits);
    let mut legacy = obs;
    legacy.hilg = None;
    assert!(model.predict(&legacy).is_err());
}
#[test]
fn hilg_preserves_occurrences_positions_goal_roles_and_renaming_invariance() {
    let mut p = fixture();
    let problem = p.problem.as_mut().unwrap();
    let atom = Predicate {
        name: "ready".into(),
        variables: vec![Symbol {
            name: "a".into(),
            symbol_type: None,
        }],
    };
    problem.goal = Some(Formula::And(vec![
        Formula::Atom(atom.clone()),
        Formula::Not(Box::new(Formula::Atom(Predicate {
            name: "locked".into(),
            ..atom.clone()
        }))),
    ]));
    let repeated = problem.init_tn.as_ref().unwrap().tn.subtasks[0].clone();
    let mut repeated = repeated;
    repeated.id = None;
    problem.init_tn.as_mut().unwrap().tn.subtasks.push(repeated);
    let obs = observation(&p);
    let graph = obs.hilg.as_ref().unwrap();
    assert_eq!(graph.nodes.iter().filter(|f| f[15] == 1.0).count(), 3);
    assert!(
        graph
            .nodes
            .iter()
            .any(|f| f[13] == 1.0 && f[16] == 1.0 && f[17] == 1.0 && f[18] == 1.0)
    );
    assert!(
        graph
            .nodes
            .iter()
            .any(|f| f[13] == 1.0 && f[19] == 1.0 && f[18] == 1.0)
    );
    let renamed = serde_json::to_string(&p)
        .unwrap()
        .replace("\"a\"", "\"renamed-z\"")
        .replace("\"b\"", "\"renamed-x\"");
    let renamed: Program = serde_json::from_str(&renamed).unwrap();
    let mut other = observation(&renamed);
    // Reindex every non-cluster node, including all edge endpoints.
    let g = other.hilg.as_mut().unwrap();
    let n = g.nodes.len();
    let c = other.nodes.len();
    g.nodes[c..].reverse();
    g.colors[c..].reverse();
    let permute = |id: usize| if id < c { id } else { c + n - 1 - id };
    for edge in &mut g.edges {
        edge.source = permute(edge.source);
        edge.target = permute(edge.target);
    }
    g.wl = Some(ptu_plan::learning::wl::WlFeatures::build(g, c).unwrap());
    let model = MergeModel::new_wl(
        ModelConfig {
            hidden: 8,
            layers: 3,
        },
        7,
    )
    .unwrap();
    let a = model.predict(&obs).unwrap();
    let b = model.predict(&other).unwrap();
    assert!(
        a.logits
            .iter()
            .zip(b.logits)
            .all(|(a, b)| (a - b).abs() < 1e-12)
    );
    // A repeated argument still contributes two independently positioned edges.
    p.problem.as_mut().unwrap().init_state[0]
        .variables
        .push(Symbol {
            name: "a".into(),
            symbol_type: None,
        });
    let repeated = observation(&p);
    let g = repeated.hilg.as_ref().unwrap();
    assert!(g.edges.iter().any(|e| e.channel == 0 && e.position == 2));
    let mut swapped = obs.clone();
    let g = swapped.hilg.as_mut().unwrap();
    for edge in &mut g.edges {
        if edge.channel < 2 {
            edge.position += 1;
        }
    }
    g.wl = Some(ptu_plan::learning::wl::WlFeatures::build(g, swapped.nodes.len()).unwrap());
    let b = model.predict(&swapped).unwrap();
    assert!(
        a.logits
            .iter()
            .zip(b.logits)
            .any(|(a, b)| (a - b).abs() > 1e-9)
    );
}
