use smallvec::SmallVec;

use crate::{domset::DominatingSet, graph::*, marker::NodeMarker};

const NOT_SET: Node = Node::MAX;

/// Rule1
///
/// For node u, partition its neighborhood into 3 distinct sets:
/// - N1(u) = set of all neighbors v that have a neighbor x not incident to u               (=Type1-Neighbors)
/// - N2(u) = set of all neighbors v not in N1(u) that are incident to a node in N1(u)      (=Type2-Neighbors)
/// - N3(u) = set of all remaining neighbors (ie. only incident to u and N2(u))             (=Type3-Neighbors)
///
/// It can be shown that if |N3(u)| > 0, u is part of an optimal dominating set and all nodes in
/// N2(u) and N3(u) must not be considered in further computations.
///
/// This algorithm correctly computes (and fixes in the provided DominatingSet) all nodes u with
/// |N3(u)| > 0. It also returns a BitSet indicating which nodes are in the Type(2 or 3)-Neighborhood
/// of u and can thus be removed. Note that this will *not* return all such Type(2 or 3)-Neighbors,
/// only some.
///
/// Rough steps of the algorithm are:
/// (1) We first compute a superset of possible candidates (u, v) where u is a possible Type3-Neighbor
/// for v. We only consider nodes for v which have the maximum degree among the neighborhood of u.
/// (2) We compute a mapping f: V -> V that maps Type3-Nodes to their respective dominating node.
/// Here, we break ties in the opposite direction and prefer dominating nodes with smaller degrees.
/// (3) We iterate over each candidate-pair (u,v) and confirm whether u is truly a Type3-Neighbor
/// for u. If true, we mark u as redundant and fix v as a dominating node.
pub struct Rule1Naive;

impl Rule1Naive {
    pub fn apply(graph: &mut Graph, sol: &mut DominatingSet, cov: &mut BitSet) {
        let mut marked = BitSet::new(graph.len());
        let mut type23 = BitSet::new(graph.len());
        let mut redundant = BitSet::new(graph.len());

        for u in graph.vertices() {
            if graph.degree_of(u) == 0 {
                // We assume singletons to be fixed beforehand
                debug_assert!(cov.get_bit(u));
                continue;
            }

            marked.clear_all();
            type23.clear_all();

            marked.set_bits(graph.closed_neighbors_of(u));

            for v in graph.neighbors_of(u) {
                if graph.closed_neighbors_of(v).all(|x| marked.get_bit(x)) {
                    if graph.degree_of(v) == graph.degree_of(u) && u < v {
                        continue;
                    }

                    type23.set_bit(v);
                }
            }

            let mut type3 = false;
            for v in graph.neighbors_of(u) {
                if graph
                    .closed_neighbors_of(v)
                    .all(|x| x == u || type23.get_bit(x))
                {
                    type3 = true;
                    break;
                }
            }

            if type3 {
                sol.add_node(u);
                redundant |= &type23;
                redundant.set_bit(u);
                cov.set_bits(graph.closed_neighbors_of(u));
            }
        }

        graph.remove_edges_at_nodes(&redundant);
    }
}

/// Rule1
///
/// For node u, partition its neighborhood into 3 distinct sets:
/// - N1(u) = set of all neighbors v that have a neighbor x not incident to u               (=Type1-Neighbors)
/// - N2(u) = set of all neighbors v not in N1(u) that are incident to a node in N1(u)      (=Type2-Neighbors)
/// - N3(u) = set of all remaining neighbors (ie. only incident to u and N2(u))             (=Type3-Neighbors)
///
/// It can be shown that if |N3(u)| > 0, u is part of an optimal dominating set and all nodes in
/// N2(u) and N3(u) must not be considered in further computations.
///
/// This algorithm correctly computes (and fixes in the provided DominatingSet) all nodes u with
/// |N3(u)| > 0. It also returns a BitSet indicating which nodes are in the Type(2 or 3)-Neighborhood
/// of u and can thus be removed. Note that this will *not* return all such Type(2 or 3)-Neighbors,
/// only some.
///
/// Rough steps of the algorithm are:
/// (1) We first compute a superset of possible candidates (u, v) where u is a possible Type3-Neighbor
/// for v. We only consider nodes for v which have the maximum degree among the neighborhood of u.
/// (2) We compute a mapping f: V -> V that maps Type3-Nodes to their respective dominating node.
/// Here, we break ties in the opposite direction and prefer dominating nodes with smaller degrees.
/// (3) We iterate over each candidate-pair (u,v) and confirm whether u is truly a Type3-Neighbor
/// for u. If true, we mark u as redundant and fix v as a dominating node.
pub struct Rule1 {
    /// Inverse mappings of step (1) and (2)
    inv_mappings: Vec<SmallVec<[Node; 4]>>,
    /// Used for confirming whether neighborhoods of nodes are subsets of other neighborhoods.
    marked: NodeMarker,
    /// Parent[u] = v if (u,v) is a possible candidate
    parent: NodeMarker,
    /// BitSet indicating that a node is a Type(2 or 3)-Candidate (not confirmed yet)
    type2_nodes: BitSet,
    /// Helper-BitSet to ensure we only process each node once later
    processed: BitSet,
    /// Number of uncovered nodes in closed neighborhood
    non_perm_degree: Vec<NumNodes>,
}

impl Rule1 {
    /// Creates the datastructures
    pub fn new(n: NumNodes) -> Self {
        Self {
            inv_mappings: vec![Default::default(); n as usize],
            marked: NodeMarker::new(n, NOT_SET),
            parent: NodeMarker::new(n, NOT_SET),
            type2_nodes: BitSet::new(n),
            processed: BitSet::new(n),
            non_perm_degree: vec![NOT_SET; n as usize],
        }
    }

    /// Resets the datastructures
    ///
    /// `self.inv_mappings` should always be reset outside of an iteration
    #[inline(always)]
    pub fn reset(&mut self) {
        self.marked.reset();
        self.parent.reset();
        self.type2_nodes.clear_all();
        self.processed.clear_all();
    }

    pub fn apply_rule<const RESET: bool>(
        &mut self,
        graph: &Graph,
        sol: &mut DominatingSet,
        cov: &mut BitSet,
    ) {
        if RESET {
            self.reset();
        }

        // Compute permanently covered nodes and degrees
        for u in graph.vertices() {
            self.non_perm_degree[u as usize] = graph.degree_of(u) + 1;
        }

        for u in cov.iter_set_bits() {
            for v in graph.closed_neighbors_of(u) {
                self.non_perm_degree[v as usize] -= 1;
            }
        }

        // (1) Compute first mapping and fix possible singletons
        for u in graph.vertices() {
            if graph.degree_of(u) == 0 {
                continue;
            }

            let max_neighbor = graph
                .closed_neighbors_of(u)
                .map(|v| (self.non_perm_degree[v as usize], v))
                .max()
                .map(|(_, v)| v)
                .unwrap();

            if max_neighbor != u {
                self.inv_mappings[max_neighbor as usize].push(u);
            }
        }

        // (1) Compute list of candidate-pairs based on mapping
        for u in graph.vertices() {
            // Mark closed neighborhood N[u] of u (SelfLoop marker)
            self.marked.mark_all_with(graph.closed_neighbors_of(u), u);

            // Check whether N[v] is a subset of N[u]
            for v in self.inv_mappings[u as usize].drain(..) {
                if graph
                    .closed_neighbors_of(v)
                    .all(|x| self.marked.is_marked_with(x, u) || cov.get_bit(x))
                {
                    self.parent.mark_with(v, u);
                    self.type2_nodes.set_bit(v);
                }
            }
        }

        // We drained inv_mappings earlier completely, so we can now reuse it
        debug_assert!(self.inv_mappings.iter().all(|vec| vec.is_empty()));

        // (2) Compute second mapping from list of candidate-pairs
        for u in self.type2_nodes.iter_set_bits() {
            for v in graph.closed_neighbors_of(u) {
                // Only process each node once
                if self.processed.set_bit(v) {
                    continue;
                }

                // Mark closed neighborhood N[v] of v (SelfLoop marker)
                self.marked.mark_all_with(graph.closed_neighbors_of(v), v);

                // Find minimum dominating node of neighbors in neighborhood of v
                if let Some((_, min_node)) = graph
                    .closed_neighbors_of(v)
                    .filter_map(|x| {
                        let pt = self.parent.get_mark(x);
                        (pt != NOT_SET && pt != v && self.marked.is_marked_with(pt, v))
                            .then(|| (self.non_perm_degree[pt as usize], pt))
                    })
                    .min()
                {
                    // We drained inv_mappings earlier completely, so we can now reuse it
                    self.inv_mappings[min_node as usize].push(v);
                }
            }
        }

        self.parent.reset();

        // (3) Mark candidates as possible Type2-Nodes if their neighborhoods are subsets
        for u in graph.vertices() {
            if self.inv_mappings[u as usize].is_empty() {
                continue;
            }

            // Mark closed neighborhood N[u] of u (SelfLoop marker)
            self.marked.mark_all_with(graph.closed_neighbors_of(u), u);

            for &v in &self.inv_mappings[u as usize] {
                if graph
                    .closed_neighbors_of(v)
                    .all(|x| self.marked.is_marked_with(x, u) || cov.get_bit(x))
                {
                    self.parent.mark_with(v, u);
                }
            }

            for v in self.inv_mappings[u as usize].drain(..) {
                if cov.get_bit(v) {
                    continue;
                }
                if graph
                    .closed_neighbors_of(v)
                    .all(|x| self.parent.is_marked_with(x, u) || x == u)
                {
                    sol.add_node(u);
                    cov.set_bits(graph.closed_neighbors_of(u));
                    break;
                }
            }
        }
    }

    /// Apply the linear removal rule
    pub fn linear_removal(&mut self, graph: &mut Graph, sol: &DominatingSet, cov: &BitSet) {
        // Used as Marker for nodes that can be safely deleted
        self.processed.clear_all();

        for u in sol.iter() {
            self.processed.set_bit(u);
        }

        for u in cov.iter_set_bits() {
            if !self.processed.get_bit(u) && graph.neighbors_of(u).all(|v| cov.get_bit(v)) {
                self.processed.set_bit(u);
            }
        }

        graph.remove_edges_at_nodes(&self.processed);
    }

    /// Apply the plus removal rule
    pub fn plus_removal(&mut self, graph: &mut Graph, sol: &mut DominatingSet, cov: &mut BitSet) {
        // Used as Marker for nodes that can be safely deleted
        self.processed.clear_all();

        for u in sol.iter() {
            self.processed.set_bit(u);
        }

        for u in cov.iter_set_bits() {
            if !self.processed.get_bit(u)
                && graph.neighbors_of(u).filter(|&v| cov.get_bit(v)).count() <= 1
            {
                self.processed.set_bit(u);
            }
        }

        graph.remove_edges_at_nodes(&self.processed);

        // Check if new singletons were created by the above step
        cov.update_cleared_bits(|u| {
            let is_singleton = graph.degree_of(u) == 0;
            if is_singleton {
                sol.add_node(u);
            }
            is_singleton
        });
    }

    /// Apply the extra removal rule
    pub fn extra_removal(&mut self, graph: &mut Graph, sol: &mut DominatingSet, cov: &mut BitSet) {
        // Remove all covered edges (this includes at least all linear/naive removals)
        graph.remove_edges_between(cov);

        // Additional Plus-Removals
        for u in cov.iter_set_bits() {
            // Neighbor must be uncovered
            if graph.degree_of(u) == 1 {
                debug_assert!(!cov.get_bit(graph.ith_neighbor(u, 0)));
                graph.remove_edges_at_node(u);
            }
        }

        // Check if new singletons were created by the above step
        cov.update_cleared_bits(|u| {
            let is_singleton = graph.degree_of(u) == 0;
            if is_singleton {
                sol.add_node(u);
            }
            is_singleton
        });
    }
}
