use smallvec::SmallVec;

use crate::{domset::DominatingSet, graph::*};

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
pub struct Rule1;

const NOT_SET: Node = Node::MAX;

impl Rule1 {
    pub fn apply_naive(graph: &mut Graph, sol: &mut DominatingSet) -> BitSet {
        let mut marked = BitSet::new(graph.len());
        let mut type2_nodes = BitSet::new(graph.len());

        let mut redundant = BitSet::new(graph.len());
        for u in graph.vertices() {
            if graph.degree_of(u) == 0 && !redundant.get_bit(u) {
                sol.add_node(u);
                continue;
            }

            if redundant.get_bit(u) {
                continue;
            }

            marked.clear_all();
            type2_nodes.clear_all();
            for v in graph.closed_neighbors_of(u) {
                marked.set_bit(v);
            }

            for v in graph.neighbors_of(u) {
                if graph.closed_neighbors_of(v).all(|x| marked.get_bit(x)) {
                    if graph.degree_of(v) == graph.degree_of(u) && u < v {
                        continue;
                    }

                    type2_nodes.set_bit(v);
                }
            }

            let mut type3 = false;
            for v in graph.neighbors_of(u) {
                type3 = type3
                    || graph
                        .closed_neighbors_of(v)
                        .all(|x| x == u || type2_nodes.get_bit(x));
            }

            if type3 {
                sol.add_node(u);
                for v in type2_nodes.iter_set_bits() {
                    graph.remove_edges_at_node(v);
                    redundant.set_bit(v);
                }

                // Add gadget
                let first_bit = type2_nodes.get_first_set_index_atleast(0).unwrap();
                graph.add_edge(u, first_bit);
            }
        }

        redundant
    }

    pub fn apply_linear(graph: &mut Graph, sol: &mut DominatingSet) -> BitSet {
        let n = graph.len();

        // Inverse mappings of step (1) and (2)
        let mut inv_mappings: Vec<SmallVec<[Node; 4]>> = vec![Default::default(); n as usize];

        // Parent[u] = v if (u,v) is a possible candidate
        let mut parent: Vec<Node> = vec![NOT_SET; n as usize];

        // Used for confirming whether neighborhoods of nodes are subsets of other neighborhoods.
        let mut marked: Vec<Node> = vec![NOT_SET; n as usize];

        // List of all possible candidate-pairs
        let mut potential_type3_node: Vec<(Node, Node)> = Vec::new();

        // BitSet indicating that a node is a Type(2 or 3)-Candidate (not confirmed yet)
        let mut type2_nodes: BitSet = BitSet::new(n);

        // Helper-BitSet to ensure we only process each node once later
        let mut processed: BitSet = BitSet::new(n);

        // BitSet of removed nodes at the end
        let mut redundant = BitSet::new(n);

        // (1) Compute first mapping and fix possible singletons
        for u in graph.vertices() {
            let max_neighbor = graph
                .closed_neighbors_of(u)
                .map(|u| (graph.degree_of(u), u))
                .max()
                .map(|(_, u)| u)
                .unwrap();
            if max_neighbor != u {
                inv_mappings[max_neighbor as usize].push(u);
            } else if graph.degree_of(u) == 0 {
                sol.add_node(u);
            }
        }

        // (1) Compute list of candidate-pairs based on mapping
        for u in graph.vertices() {
            // Mark closed neighborhood N[u] of u (SelfLoop marker)
            for v in graph.closed_neighbors_of(u) {
                marked[v as usize] = u;
            }

            // Check whether N[v] is a subset of N[u]
            for v in inv_mappings[u as usize].drain(..) {
                if graph
                    .closed_neighbors_of(v)
                    .all(|x| marked[x as usize] == u)
                {
                    parent[v as usize] = u;
                    potential_type3_node.push((v, u));

                    type2_nodes.set_bit(v);
                }
            }
        }

        // We drained inv_mappings earlier completely, so we can now reuse it
        debug_assert!(inv_mappings.iter().all(|vec| vec.is_empty()));

        // (2) Compute second mapping from list of candidate-pairs
        for &(u, _) in &potential_type3_node {
            for v in graph.closed_neighbors_of(u) {
                // Only process each node once
                if processed.set_bit(v) {
                    continue;
                }

                // Mark closed neighborhood N[v] of v (SelfLoop marker)
                for x in graph.closed_neighbors_of(v) {
                    marked[x as usize] = v;
                }

                // Find minimum dominating node of neighbors in neighborhood of v
                if let Some((_, min_node)) = graph
                    .closed_neighbors_of(v)
                    .filter_map(|x| {
                        let pt = parent[x as usize];
                        (pt != NOT_SET && pt != v && marked[pt as usize] == v)
                            .then(|| (graph.degree_of(pt), pt))
                    })
                    .min()
                {
                    // We drained inv_mappings earlier completely, so we can now reuse it
                    inv_mappings[min_node as usize].push(v);
                }
            }
        }

        parent = vec![NOT_SET; n as usize];

        // (3) Mark candidates as possible Type2-Nodes if their neighborhoods are subsets
        for u in graph.vertices() {
            // Mark closed neighborhood N[u] of u (SelfLoop marker)
            for v in graph.closed_neighbors_of(u) {
                marked[v as usize] = u;
            }

            for &v in &inv_mappings[u as usize] {
                if graph
                    .closed_neighbors_of(v)
                    .all(|x| marked[x as usize] == u)
                {
                    parent[v as usize] = u;
                }
            }

            for v in inv_mappings[u as usize].drain(..) {
                if graph
                    .closed_neighbors_of(v)
                    .all(|x| parent[x as usize] == u || x == u)
                {
                    sol.add_node(u);
                    break;
                }
            }
        }

        parent = vec![NOT_SET; n as usize];
        for u in sol.iter() {
            parent[u as usize] = u;
            for v in graph.neighbors_of(u) {
                if parent[v as usize] == NOT_SET {
                    parent[v as usize] = u;
                }
            }
        }

        for u in graph.vertices() {
            if parent[u as usize] != u
                && graph
                    .closed_neighbors_of(u)
                    .all(|v| parent[v as usize] != NOT_SET)
            {
                redundant.set_bit(u);
            }
        }

        for u in redundant.iter_set_bits() {
            graph.remove_edges_at_node(u);
        }

        if let Some(first_bit) = redundant.get_first_set_index_atleast(0) {
            graph.add_edge(parent[first_bit as usize], first_bit);
        }

        redundant
    }

    pub fn apply_plus(graph: &mut Graph, sol: &mut DominatingSet) -> BitSet {
        let n = graph.len();

        // Inverse mappings of step (1) and (2)
        let mut inv_mappings: Vec<SmallVec<[Node; 4]>> = vec![Default::default(); n as usize];

        // Parent[u] = v if (u,v) is a possible candidate
        let mut parent: Vec<Node> = vec![NOT_SET; n as usize];

        // Used for confirming whether neighborhoods of nodes are subsets of other neighborhoods.
        let mut marked: Vec<Node> = vec![NOT_SET; n as usize];

        // List of all possible candidate-pairs
        let mut potential_type3_node: Vec<(Node, Node)> = Vec::new();

        // BitSet indicating that a node is a Type(2 or 3)-Candidate (not confirmed yet)
        let mut type2_nodes: BitSet = BitSet::new(n);

        // Helper-BitSet to ensure we only process each node once later
        let mut processed: BitSet = BitSet::new(n);

        // BitSet of removed nodes at the end
        let mut redundant = BitSet::new(n);

        // (1) Compute first mapping and fix possible singletons
        for u in graph.vertices() {
            let max_neighbor = graph
                .closed_neighbors_of(u)
                .map(|u| (graph.degree_of(u), u))
                .max()
                .map(|(_, u)| u)
                .unwrap();
            if max_neighbor != u {
                inv_mappings[max_neighbor as usize].push(u);
            } else if graph.degree_of(u) == 0 {
                sol.add_node(u);
            }
        }

        // (1) Compute list of candidate-pairs based on mapping
        for u in graph.vertices() {
            // Mark closed neighborhood N[u] of u (SelfLoop marker)
            for v in graph.closed_neighbors_of(u) {
                marked[v as usize] = u;
            }

            // Check whether N[v] is a subset of N[u]
            for v in inv_mappings[u as usize].drain(..) {
                if graph
                    .closed_neighbors_of(v)
                    .all(|x| marked[x as usize] == u)
                {
                    parent[v as usize] = u;
                    potential_type3_node.push((v, u));

                    type2_nodes.set_bit(v);
                }
            }
        }

        // We drained inv_mappings earlier completely, so we can now reuse it
        debug_assert!(inv_mappings.iter().all(|vec| vec.is_empty()));

        // (2) Compute second mapping from list of candidate-pairs
        for &(u, _) in &potential_type3_node {
            for v in graph.closed_neighbors_of(u) {
                // Only process each node once
                if processed.set_bit(v) {
                    continue;
                }

                // Mark closed neighborhood N[v] of v (SelfLoop marker)
                for x in graph.closed_neighbors_of(v) {
                    marked[x as usize] = v;
                }

                // Find minimum dominating node of neighbors in neighborhood of v
                if let Some((_, min_node)) = graph
                    .closed_neighbors_of(v)
                    .filter_map(|x| {
                        let pt = parent[x as usize];
                        (pt != NOT_SET && pt != v && marked[pt as usize] == v)
                            .then(|| (graph.degree_of(pt), pt))
                    })
                    .min()
                {
                    // We drained inv_mappings earlier completely, so we can now reuse it
                    inv_mappings[min_node as usize].push(v);
                }
            }
        }

        parent = vec![NOT_SET; n as usize];

        // (3) Mark candidates as possible Type2-Nodes if their neighborhoods are subsets
        for u in graph.vertices() {
            // Mark closed neighborhood N[u] of u (SelfLoop marker)
            for v in graph.closed_neighbors_of(u) {
                marked[v as usize] = u;
            }

            for &v in &inv_mappings[u as usize] {
                if graph
                    .closed_neighbors_of(v)
                    .all(|x| marked[x as usize] == u)
                {
                    parent[v as usize] = u;
                }
            }

            for v in inv_mappings[u as usize].drain(..) {
                if graph
                    .closed_neighbors_of(v)
                    .all(|x| parent[x as usize] == u || x == u)
                {
                    sol.add_node(u);
                    break;
                }
            }
        }

        parent = vec![NOT_SET; n as usize];
        for u in sol.iter() {
            parent[u as usize] = u;
            for v in graph.neighbors_of(u) {
                if parent[v as usize] == NOT_SET {
                    parent[v as usize] = u;
                }
            }
        }

        for u in graph.vertices() {
            if parent[u as usize] != u
                && parent[u as usize] != NOT_SET
                && graph
                    .neighbors_of(u)
                    .filter(|&v| parent[v as usize] != NOT_SET)
                    .count()
                    <= 1
            {
                redundant.set_bit(u);
            }
        }

        for u in redundant.iter_set_bits() {
            graph.remove_edges_at_node(u);
        }

        if let Some(first_bit) = redundant.get_first_set_index_atleast(0) {
            graph.add_edge(parent[first_bit as usize], first_bit);
        }

        redundant
    }

    pub fn apply_extra(graph: &mut Graph, sol: &mut DominatingSet) -> BitSet {
        let n = graph.len();

        // Inverse mappings of step (1) and (2)
        let mut inv_mappings: Vec<SmallVec<[Node; 4]>> = vec![Default::default(); n as usize];

        // Parent[u] = v if (u,v) is a possible candidate
        let mut parent: Vec<Node> = vec![NOT_SET; n as usize];

        // Used for confirming whether neighborhoods of nodes are subsets of other neighborhoods.
        let mut marked: Vec<Node> = vec![NOT_SET; n as usize];

        // List of all possible candidate-pairs
        let mut potential_type3_node: Vec<(Node, Node)> = Vec::new();

        // BitSet indicating that a node is a Type(2 or 3)-Candidate (not confirmed yet)
        let mut type2_nodes: BitSet = BitSet::new(n);

        // Helper-BitSet to ensure we only process each node once later
        let mut processed: BitSet = BitSet::new(n);

        // BitSet of removed nodes at the end
        let mut redundant = BitSet::new(n);

        // (1) Compute first mapping and fix possible singletons
        for u in graph.vertices() {
            let max_neighbor = graph
                .closed_neighbors_of(u)
                .map(|u| (graph.degree_of(u), u))
                .max()
                .map(|(_, u)| u)
                .unwrap();
            if max_neighbor != u {
                inv_mappings[max_neighbor as usize].push(u);
            } else if graph.degree_of(u) == 0 {
                sol.add_node(u);
            }
        }

        // (1) Compute list of candidate-pairs based on mapping
        for u in graph.vertices() {
            // Mark closed neighborhood N[u] of u (SelfLoop marker)
            for v in graph.closed_neighbors_of(u) {
                marked[v as usize] = u;
            }

            // Check whether N[v] is a subset of N[u]
            for v in inv_mappings[u as usize].drain(..) {
                if graph
                    .closed_neighbors_of(v)
                    .all(|x| marked[x as usize] == u)
                {
                    parent[v as usize] = u;
                    potential_type3_node.push((v, u));

                    type2_nodes.set_bit(v);
                }
            }
        }

        // We drained inv_mappings earlier completely, so we can now reuse it
        debug_assert!(inv_mappings.iter().all(|vec| vec.is_empty()));

        // (2) Compute second mapping from list of candidate-pairs
        for &(u, _) in &potential_type3_node {
            for v in graph.closed_neighbors_of(u) {
                // Only process each node once
                if processed.set_bit(v) {
                    continue;
                }

                // Mark closed neighborhood N[v] of v (SelfLoop marker)
                for x in graph.closed_neighbors_of(v) {
                    marked[x as usize] = v;
                }

                // Find minimum dominating node of neighbors in neighborhood of v
                if let Some((_, min_node)) = graph
                    .closed_neighbors_of(v)
                    .filter_map(|x| {
                        let pt = parent[x as usize];
                        (pt != NOT_SET && pt != v && marked[pt as usize] == v)
                            .then(|| (graph.degree_of(pt), pt))
                    })
                    .min()
                {
                    // We drained inv_mappings earlier completely, so we can now reuse it
                    inv_mappings[min_node as usize].push(v);
                }
            }
        }

        parent = vec![NOT_SET; n as usize];

        // (3) Mark candidates as possible Type2-Nodes if their neighborhoods are subsets
        for u in graph.vertices() {
            // Mark closed neighborhood N[u] of u (SelfLoop marker)
            for v in graph.closed_neighbors_of(u) {
                marked[v as usize] = u;
            }

            for &v in &inv_mappings[u as usize] {
                if graph
                    .closed_neighbors_of(v)
                    .all(|x| marked[x as usize] == u)
                {
                    parent[v as usize] = u;
                }
            }

            for v in inv_mappings[u as usize].drain(..) {
                if graph
                    .closed_neighbors_of(v)
                    .all(|x| parent[x as usize] == u || x == u)
                {
                    sol.add_node(u);
                    break;
                }
            }
        }

        parent = vec![NOT_SET; n as usize];
        for u in sol.iter() {
            parent[u as usize] = u;
            for v in graph.neighbors_of(u) {
                if parent[v as usize] == NOT_SET {
                    parent[v as usize] = u;
                }
            }
        }

        for u in graph.vertices() {
            if parent[u as usize] != u
                && parent[u as usize] != NOT_SET
                && graph
                    .neighbors_of(u)
                    .filter(|&v| parent[v as usize] != NOT_SET)
                    .count()
                    <= 1
            {
                redundant.set_bit(u);
            }
        }

        for u in redundant.iter_set_bits() {
            graph.remove_edges_at_node(u);
        }

        graph.remove_edges_between(|u| parent[u as usize] != NOT_SET);

        if let Some(first_bit) = redundant.get_first_set_index_atleast(0) {
            graph.add_edge(parent[first_bit as usize], first_bit);
        }

        redundant
    }
}
