//! Höller et al.'s HTN Instance Learning Graph, extended with partition nodes,
//! ordering, and formula syntax. This is an input representation, not a proof.
use super::features::Observation;
use crate::{
    agglomeration::Agglomeration,
    analysis::{HtnAnalysis, NetworkId, TaskKind},
    interaction::{InteractionGraph, NodeKind},
    parsing::json::{Formula, Predicate, Symbol},
};
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const FEATURES: usize = 32;
pub const CHANNELS: usize = 20;
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstanceEdge {
    pub source: usize,
    pub target: usize,
    pub channel: usize,
    /// Zero for structural edges; one-based for argument/child positions.
    pub position: usize,
    pub weight: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstanceGraph {
    pub schema_version: u32,
    /// First C nodes are partition nodes, aligned with Observation.nodes.
    pub nodes: Vec<Vec<f64>>,
    pub edges: Vec<InstanceEdge>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub colors: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wl: Option<super::wl::WlFeatures>,
}
impl InstanceGraph {
    pub fn validate(&self, clusters: usize) -> Result<()> {
        if self.schema_version != 1
            || self.nodes.len() < clusters
            || self
                .nodes
                .iter()
                .any(|r| r.len() != FEATURES || r.iter().any(|x| !x.is_finite()))
            || self.edges.iter().any(|e| {
                e.source >= self.nodes.len()
                    || e.target >= self.nodes.len()
                    || e.channel >= CHANNELS
                    || !e.weight.is_finite()
                    || e.weight <= 0.0
            })
        {
            bail!("invalid HILG instance graph");
        }
        if !self.colors.is_empty() && self.colors.len() != self.nodes.len() {
            bail!("invalid HILG colors");
        }
        if let Some(features) = &self.wl {
            features.validate(clusters)?;
        }
        Ok(())
    }
    pub fn build(
        analysis: &HtnAnalysis,
        interaction: &InteractionGraph,
        state: &Agglomeration,
        observation: &Observation,
    ) -> Result<Self> {
        let program = analysis
            .instance
            .as_ref()
            .context("HILG requires the original instance")?;
        let mut b = Builder {
            graph: Self {
                schema_version: 1,
                nodes: Vec::new(),
                edges: Vec::new(),
                colors: Vec::new(),
                wl: None,
            },
            objects: BTreeMap::new(),
            atoms: BTreeMap::new(),
            current: BTreeMap::new(),
        };
        for cluster in &observation.nodes {
            let id = b.node(23, "");
            b.graph.nodes[id][..12].copy_from_slice(&cluster.features);
        }
        for object in program
            .domain
            .constants
            .iter()
            .flatten()
            .chain(program.problem.iter().flat_map(|p| &p.objects))
        {
            b.object(&object.name);
        }
        if let Some(problem) = &program.problem {
            for atom in &problem.init_state {
                b.current.insert(key(atom), true);
                let id = b.atom(atom, false, false);
                b.graph.nodes[id][16] = 1.0;
            }
            if let Some(goal) = &problem.goal {
                b.formula(goal);
            }
        }
        let mut occurrences = BTreeMap::new();
        for node in &interaction.nodes {
            if let NodeKind::Occurrence {
                network: NetworkId::Initial,
                index,
                task,
                terms,
            } = &node.kind
            {
                let schema = &analysis.tasks[task.0];
                let id = b.node(
                    if schema.kind == TaskKind::Primitive {
                        14
                    } else {
                        15
                    },
                    &schema.name,
                );
                b.arguments(id, terms);
                occurrences.insert(*index, id);
                for (position, cluster) in state.clusters.iter().enumerate() {
                    if cluster.nodes.contains(&node.id) {
                        b.edge(id, position, 2, 0, 1.0);
                    }
                }
            }
        }
        if let Some(network) = &analysis.initial_network {
            for order in &network.precedence {
                b.edge(
                    occurrences[&order.before],
                    occurrences[&order.after],
                    4,
                    0,
                    1.0,
                );
            }
        }
        for relation in &observation.relations {
            b.graph.edges.push(InstanceEdge {
                source: relation.source,
                target: relation.target,
                channel: 8 + relation.channel,
                position: 0,
                weight: relation.weight,
            });
        }
        b.graph.wl = Some(super::wl::WlFeatures::build(
            &b.graph,
            observation.nodes.len(),
        )?);
        b.graph.validate(observation.nodes.len())?;
        Ok(b.graph)
    }
}
fn key(atom: &Predicate) -> (String, Vec<String>) {
    (
        atom.name.clone(),
        atom.variables.iter().map(|s| s.name.clone()).collect(),
    )
}
struct Builder {
    graph: InstanceGraph,
    objects: BTreeMap<String, usize>,
    atoms: BTreeMap<(String, Vec<String>, bool), usize>,
    current: BTreeMap<(String, Vec<String>), bool>,
}
impl Builder {
    fn node(&mut self, role: usize, color: &str) -> usize {
        let mut features = vec![0.0; FEATURES];
        features[role] = 1.0;
        if !color.is_empty() {
            // Compact color fingerprints for inspection; WL uses the full strings.
            // Object identifiers are never encoded.
            use sha2::{Digest, Sha256};
            let hash = Sha256::digest(color.as_bytes());
            for i in 0..8 {
                features[24 + i] = if hash[i] & 1 == 0 { -0.35 } else { 0.35 };
            }
        }
        self.graph.colors.push(format!("{role}:{color}"));
        let id = self.graph.nodes.len();
        self.graph.nodes.push(features);
        id
    }
    fn object(&mut self, name: &str) -> usize {
        if let Some(&id) = self.objects.get(name) {
            return id;
        }
        let id = self.node(if name.starts_with('?') { 21 } else { 12 }, "");
        self.objects.insert(name.to_owned(), id);
        id
    }
    fn edge(&mut self, a: usize, b: usize, channel: usize, position: usize, weight: f64) {
        self.graph.edges.push(InstanceEdge {
            source: a,
            target: b,
            channel,
            position,
            weight,
        });
        self.graph.edges.push(InstanceEdge {
            source: b,
            target: a,
            channel: channel + 1,
            position,
            weight,
        });
    }
    fn arguments(&mut self, id: usize, terms: &[Symbol]) {
        self.graph.nodes[id][22] = (terms.len() as f64).ln_1p();
        for (index, term) in terms.iter().enumerate() {
            let object = self.object(&term.name);
            self.edge(id, object, 0, index + 1, 1.0);
        }
    }
    fn atom(&mut self, atom: &Predicate, negative: bool, goal: bool) -> usize {
        let (name, terms) = key(atom);
        let key = (name.clone(), terms.clone(), negative);
        let id = if let Some(&id) = self.atoms.get(&key) {
            id
        } else {
            let id = self.node(13, &name);
            self.arguments(id, &atom.variables);
            self.graph.nodes[id][19] = f64::from(negative);
            self.atoms.insert(key, id);
            id
        };
        if goal {
            self.graph.nodes[id][17] = 1.0;
            self.graph.nodes[id][18] =
                f64::from(self.current.contains_key(&(name, terms)) != negative);
        }
        id
    }
    fn formula(&mut self, formula: &Formula) -> usize {
        if let Formula::Atom(atom) = formula {
            return self.atom(atom, false, true);
        }
        if let Formula::Not(inner) = formula
            && let Formula::Atom(atom) = inner.as_ref()
        {
            return self.atom(atom, true, true);
        }
        let color = match formula {
            Formula::Empty => "empty",
            Formula::Not(_) => "not",
            Formula::And(_) => "and",
            Formula::Or(_) => "or",
            Formula::Xor(_) => "xor",
            Formula::Imply(..) => "imply",
            Formula::Exists(..) => "exists",
            Formula::ForAll(..) => "forall",
            Formula::Probabilistic(..) => "probabilistic",
            Formula::Equals(..) => "equals",
            Formula::Atom(_) => unreachable!(),
        };
        let id = self.node(20, color);
        let mut children = Vec::new();
        match formula {
            Formula::Not(inner) | Formula::Probabilistic(_, inner) => {
                children.push(self.formula(inner))
            }
            Formula::And(terms) | Formula::Or(terms) | Formula::Xor(terms) => {
                for term in terms {
                    children.push(self.formula(term));
                }
            }
            Formula::Imply(a, b) => {
                children.push(self.formula(a));
                children.push(self.formula(b));
            }
            Formula::Exists(vars, inner) | Formula::ForAll(vars, inner) => {
                let old = self.objects.clone();
                for var in vars {
                    let object = self.node(21, "");
                    self.objects.insert(var.name.clone(), object);
                    self.edge(id, object, 6, 0, 1.0);
                }
                // Bound variables have scope-specific IDs and cannot alias objects.
                // Goal atoms under binders must not be interned across scopes.
                let atoms = std::mem::take(&mut self.atoms);
                children.push(self.formula(inner));
                self.atoms = atoms;
                self.objects = old;
            }
            Formula::Equals(a, b) => {
                children.push(self.object(a));
                children.push(self.object(b));
            }
            _ => {}
        }
        for (index, child) in children.into_iter().enumerate() {
            // Conjunction/disjunction are permutation invariant; implication is ordered.
            let position = if matches!(formula, Formula::Imply(..) | Formula::Equals(..)) {
                index + 1
            } else {
                0
            };
            self.edge(id, child, 6, position, 1.0);
        }
        id
    }
}
