use crate::graph::Node;

pub struct DominatingSet(pub Vec<Node>);

impl DominatingSet {
    /// Creates a new DominatingSet
    pub fn new() -> Self {
        Self(Vec::new())
    }

    /// Returns the size of the DominatingSet
    pub fn len(&self) -> Node {
        self.0.len() as Node
    }

    /// Adds a node to the DominatingSet
    pub fn add_node(&mut self, u: Node) {
        self.0.push(u);
    }

    /// Iterates over the DominatingSet
    pub fn iter(&self) -> impl Iterator<Item = Node> {
        self.0.iter().copied()
    }
}
