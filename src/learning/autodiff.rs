//! Small CPU reverse-mode tape. The model and gradient checks share this path.
pub type Var = usize;
struct Node {
    parents: [(Var, f64); 2],
    count: usize,
}
pub struct Tape {
    values: Vec<f64>,
    /// Inference retains values only, with no derivative graph or adjoints.
    nodes: Option<Vec<Node>>,
}
impl Tape {
    pub fn new() -> Self {
        Self {
            values: Vec::new(),
            nodes: Some(Vec::new()),
        }
    }
    pub fn inference() -> Self {
        Self {
            values: Vec::new(),
            nodes: None,
        }
    }
    fn node(&mut self, value: f64, parents: [(Var, f64); 2], count: usize) -> Var {
        let id = self.values.len();
        self.values.push(value);
        if let Some(nodes) = &mut self.nodes {
            nodes.push(Node { parents, count });
        }
        id
    }
    pub fn constant(&mut self, value: f64) -> Var {
        self.node(value, [(0, 0.0); 2], 0)
    }
    pub fn value(&self, v: Var) -> f64 {
        self.values[v]
    }
    pub fn add(&mut self, a: Var, b: Var) -> Var {
        self.node(self.value(a) + self.value(b), [(a, 1.0), (b, 1.0)], 2)
    }
    pub fn scale(&mut self, a: Var, w: f64) -> Var {
        self.node(self.value(a) * w, [(a, w), (0, 0.0)], 1)
    }
    pub fn mul(&mut self, a: Var, b: Var) -> Var {
        self.node(
            self.value(a) * self.value(b),
            [(a, self.value(b)), (b, self.value(a))],
            2,
        )
    }
    pub fn tanh(&mut self, a: Var) -> Var {
        let v = self.value(a).tanh();
        self.node(v, [(a, 1.0 - v * v), (0, 0.0)], 1)
    }
    pub fn abs(&mut self, a: Var) -> Var {
        let v = self.value(a);
        self.node(
            v.abs(),
            [(a, if v == 0.0 { 0.0 } else { v.signum() }), (0, 0.0)],
            1,
        )
    }
    pub fn softplus(&mut self, a: Var) -> Var {
        let v = self.value(a);
        let value = v.max(0.0) + (-v.abs()).exp().ln_1p();
        let gradient = if v >= 0.0 {
            1.0 / (1.0 + (-v).exp())
        } else {
            let e = v.exp();
            e / (1.0 + e)
        };
        self.node(value, [(a, gradient), (0, 0.0)], 1)
    }
    pub fn huber(&mut self, a: Var) -> Var {
        let v = self.value(a);
        self.node(
            if v.abs() <= 1.0 {
                0.5 * v * v
            } else {
                v.abs() - 0.5
            },
            [(a, v.clamp(-1.0, 1.0)), (0, 0.0)],
            1,
        )
    }
    pub fn backward(&mut self, output: Var, parameters: usize) -> Vec<f64> {
        let nodes = self
            .nodes
            .as_ref()
            .expect("backward requires a training tape");
        let mut gradients = vec![0.0; nodes.len()];
        gradients[output] = 1.0;
        for id in (0..nodes.len()).rev() {
            let gradient = gradients[id];
            let parents = nodes[id].parents;
            let count = nodes[id].count;
            for &(parent, weight) in &parents[..count] {
                gradients[parent] += gradient * weight;
            }
        }
        gradients.truncate(parameters);
        gradients
    }
}
