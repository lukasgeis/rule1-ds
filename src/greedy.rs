use crate::{domset::DominatingSet, graph::*, radix::NodeHeap};

/// # The basic Greedy-Approximation
///
/// 1. Greedily adds nodes neighbored to the highest number of uncovered nodes to DomSet until
///    covered
/// 2. Remove nodes that now redundant, ie. do not cover any nodes that no other DomSet-Node
///    covers
pub struct Greedy;

impl Greedy {
    pub fn compute(graph: &Graph, sol: &mut DominatingSet, cov: &BitSet) {
        let prev_dom_nodes = sol.len();

        // Compute how many neighbors in the DomSet every node has
        let mut num_covered = vec![0usize; graph.len() as usize];
        cov.iter_set_bits()
            .for_each(|u| num_covered[u as usize] = 1);
        let mut total_covered = cov.cardinality();

        let max_degree = graph.max_degree();

        // Compute scores for nodes that have neighbors
        // (every other node is either in the DomSet or has its entire neighborhood covered at this point)
        let mut heap = NodeHeap::new(graph.len() as usize, 0);
        for u in graph.vertices_with_neighbors() {
            let mut node_score = max_degree + 1;
            for v in graph.closed_neighbors_of(u) {
                node_score -= (num_covered[v as usize] == 0) as NumNodes;
            }

            if node_score == max_degree + 1 {
                continue;
            }

            heap.push(node_score, u);
        }

        // Compute rest of DomSet via GreedyAlgorithm
        while total_covered < graph.len() {
            let (_, node) = heap.pop().unwrap();
            sol.add_node(node);

            for u in graph.closed_neighbors_of(node) {
                num_covered[u as usize] += 1;
                if num_covered[u as usize] == 1 {
                    total_covered += 1;
                    for v in graph.closed_neighbors_of(u) {
                        if v != node {
                            let current_score = heap.remove(v);
                            heap.push(current_score + 1, v);
                        }
                    }
                }
            }
        }

        // Remove redundant nodes from DomSet
        let mut index = prev_dom_nodes;
        while index < sol.len() {
            let node = sol.ith_node(index);
            if graph
                .closed_neighbors_of(node)
                .all(|u| num_covered[u as usize] > 1)
            {
                for u in graph.closed_neighbors_of(node) {
                    num_covered[u as usize] -= 1;
                }
                sol.remove_node(node);
                continue;
            }
            index += 1;
        }
    }
}
