use crate::graph::*;

/// A DominatingSet that allows differentiation between fixed and non-fixed nodes in the
/// set. Supports constant time queries of membership in set.
#[derive(Clone)]
pub struct DominatingSet {
    /// List of all nodes in the set, partitioned by fixed/non-fixed
    solution: Vec<Node>,
    /// Position for each possible node in the set (= NumNodes::MAX if not)
    positions: Vec<NumNodes>,
}

impl DominatingSet {
    /// Creates a new dominating set
    pub fn new(n: NumNodes) -> Self {
        Self {
            solution: Vec::new(),
            positions: vec![Node::MAX; n as usize],
        }
    }

    /// Returns the number of nodes in the dominating set.
    pub fn len(&self) -> usize {
        self.solution.len()
    }

    /// Adds a node to the dominating set.
    pub fn add_node(&mut self, u: Node) {
        debug_assert!(self.positions[u as usize] == NumNodes::MAX);
        self.positions[u as usize] = self.len() as NumNodes;
        self.solution.push(u);
    }

    /// Removes a node from the dominating set.
    pub fn remove_node(&mut self, u: Node) {
        let pos = self.positions[u as usize] as usize;
        debug_assert!(pos < self.len());

        self.solution.swap_remove(pos);
        if pos < self.len() {
            self.positions[self.solution[pos] as usize] = pos as NumNodes;
        }

        self.positions[u as usize] = NumNodes::MAX;
    }

    /// Returns an iterator over the nodes in the dominating set.
    pub fn iter(&self) -> impl Iterator<Item = Node> + '_ {
        self.solution.iter().copied()
    }

    /// Returns the ith node in the dominating set.
    pub fn ith_node(&self, i: usize) -> Node {
        self.solution[i]
    }

    /// Returns true if the dominating set is valid, ie. it covers all nodes.
    pub fn is_valid(&self, graph: &Graph) -> bool {
        let mut covered = BitSet::new(graph.len());
        for &u in &self.solution {
            covered.set_bits(graph.closed_neighbors_of(u));
        }

        covered.are_all_set()
    }
}
