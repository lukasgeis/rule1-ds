use std::time::Instant;

use itertools::Itertools;
use rand::Rng;

use crate::{domset::DominatingSet, graph::*, radix::NodeHeap};

/// # The basic Greedy-Approximation
///
/// 1. Fixes nodes in complete subgraphs where at most one node has neighbors outside the subgraph
///     - Singletons
///     - Nodes that are the only neighbors of some other node
///     - Nodes part of a triangle where the other two nodes are only incident to the triangle
/// 2. Greedily adds nodes neighbored to the highest number of uncovered nodes to DomSet until
///    covered
/// 3. Remove nodes that now redundant, ie. do not cover any nodes that not other DomSet-Node
///    covers
///
/// Returns the solution
pub struct IterativeGreedy<'a, R> {
    rng: &'a mut R,
    graph: &'a Graph,

    max_degree: NumNodes,
    total_covered: NumNodes,
    num_covered: Vec<NumNodes>,

    base_solution: &'a DominatingSet,

    rand_bits: u32,
    rand_mask: NumNodes,
}

#[derive(Debug, Clone, Copy)]
pub enum GreedyStrategy {
    UnitValue,
    DegreeValue,
}

impl<'a, R: Rng> IterativeGreedy<'a, R> {
    pub fn new(rng: &'a mut R, graph: &'a Graph, sol: &'a DominatingSet, cov: &'a BitSet) -> Self {
        let mut num_covered = vec![0; graph.len() as usize];

        cov.iter_set_bits()
            .for_each(|u| num_covered[u as usize] = 1);
        let total_covered = cov.cardinality();

        let max_degree = graph.max_degree();
        let rand_bits = (max_degree + 1).leading_zeros();
        Self {
            rng,
            graph,
            max_degree,
            total_covered,
            num_covered,
            base_solution: sol,
            rand_bits,
            rand_mask: (1 << rand_bits) - 1,
        }
    }

    pub fn run(&mut self, strategy: GreedyStrategy) -> DominatingSet {
        let mut solution = self.base_solution.clone();

        const MAX_RAND: u32 = 64;
        let max_value = u32::MAX / (1 + self.max_degree) - MAX_RAND;

        let mut node_values = match strategy {
            GreedyStrategy::UnitValue => self.num_covered.clone(),
            GreedyStrategy::DegreeValue => self
                .graph
                .vertices()
                .map(|u| {
                    if self.num_covered[u as usize] > 0 {
                        0
                    } else {
                        let coverage_candidated =
                            self.graph.closed_neighbors_of(u).count() as NumNodes;

                        let random = self.rng.random_range(0..MAX_RAND);
                        (max_value / (1 + coverage_candidated)).max(1) + random
                    }
                })
                .collect_vec(),
        };

        // Compute scores for non-fixed nodes
        let mut heap = NodeHeap::new(self.graph.len() as usize, 0);
        for u in self.graph.vertices() {
            match strategy {
                GreedyStrategy::UnitValue => {
                    let mut node_score =
                        self.max_degree + (node_values[u as usize] > 0) as NumNodes;
                    for v in self.graph.neighbors_of(u) {
                        node_score -= (node_values[v as usize] == 0) as NumNodes;
                    }

                    if node_score <= self.max_degree {
                        let rand_word: u32 = self.rng.random();
                        let rand_score =
                            (node_score << self.rand_bits) | (rand_word & self.rand_mask);

                        heap.push(rand_score, u);
                    }
                }
                GreedyStrategy::DegreeValue => {
                    let value: u32 = self
                        .graph
                        .closed_neighbors_of(u)
                        .map(|v| node_values[v as usize])
                        .sum();

                    if value > 0 {
                        heap.push(u32::MAX - value, u);
                    }
                }
            }
        }

        let mut total_covered = self.total_covered;

        // Compute rest of DomSet via GreedyAlgorithm
        while total_covered < self.graph.len() {
            let (_, node) = heap.pop().unwrap();
            solution.add_node(node);

            match strategy {
                GreedyStrategy::UnitValue => {
                    for u in self.graph.closed_neighbors_of(node) {
                        node_values[u as usize] += 1;
                        if node_values[u as usize] == 1 {
                            total_covered += 1;
                            for v in self.graph.closed_neighbors_of(u) {
                                if v != node {
                                    let current_score = heap.remove(v);
                                    heap.push(current_score + 1 + self.rand_mask, v);
                                }
                            }
                        }
                    }
                }
                GreedyStrategy::DegreeValue => {
                    for u in self.graph.closed_neighbors_of(node) {
                        let prev_value = node_values[u as usize];
                        if prev_value == 0 {
                            continue;
                        }
                        node_values[u as usize] = 0;
                        total_covered += 1;

                        for v in self.graph.closed_neighbors_of(u) {
                            if v != node {
                                let current_score = heap.remove(v) + prev_value;
                                if current_score < u32::MAX {
                                    heap.push(current_score, v);
                                }
                            }
                        }
                    }
                }
            }
        }

        let mut index = 0;
        while index < solution.len() {
            let node = solution.ith_node(index);
            if self
                .graph
                .closed_neighbors_of(node)
                .all(|u| node_values[u as usize] > 1)
            {
                for u in self.graph.closed_neighbors_of(node) {
                    node_values[u as usize] -= 1;
                }
                solution.remove_node(node);
                continue;
            }
            index += 1;
        }

        solution
    }
}

pub fn run_greedy<R: Rng>(
    graph: &Graph,
    rng: &mut R,
    sol: &DominatingSet,
    cov: &BitSet,
    num_iter: u128,
) -> (DominatingSet, u128) {
    let mut timer = Instant::now();
    let mut iter_greedy = IterativeGreedy::new(rng, graph, sol, cov);
    let init_time = timer.elapsed().as_nanos();

    let mut best_sol: Option<DominatingSet> = None;

    let mut total_time = 0u128;

    for i in 0..num_iter {
        let strategy = if i & 1 == 0 {
            GreedyStrategy::UnitValue
        } else {
            GreedyStrategy::DegreeValue
        };

        timer = Instant::now();
        let sol = iter_greedy.run(strategy);
        total_time += timer.elapsed().as_nanos();

        if let Some(bsol) = best_sol.as_ref() {
            if sol.len() < bsol.len() {
                best_sol = Some(sol);
            }
        } else {
            best_sol = Some(sol);
        }
    }

    (best_sol.unwrap(), init_time + (total_time / num_iter))
}
