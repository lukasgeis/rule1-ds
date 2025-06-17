use std::ops::Range;

use stream_bitset::bitset::BitSetImpl;

pub type Node = u32;
pub type Edge = (Node, Node);

pub type BitSet = BitSetImpl<Node>;

/// Minimal Graph-Representation using AdjacencyLists
pub struct Graph {
    nbs: Vec<Vec<Node>>,
    m: usize,
}

impl Graph {
    /// Creates a new graph from an iterator of edges
    pub fn new<I: Iterator<Item = Edge>>(n: Node, edges: I) -> Self {
        let mut m = 0;
        let mut neighborhoods = vec![Vec::new(); n as usize];
        for (u, v) in edges {
            m += 1;
            neighborhoods[u as usize].push(v);
            neighborhoods[v as usize].push(u);
        }

        Self {
            nbs: neighborhoods,
            m,
        }
    }

    /// Returns the number of nodes in the graph
    pub fn len(&self) -> Node {
        self.nbs.len() as Node
    }

    /// Returns the number of edges in the graph
    pub fn num_edges(&self) -> usize {
        self.m
    }

    /// Provides a range over all nodes in the graph
    pub fn vertices(&self) -> Range<Node> {
        0..self.len()
    }

    /// Returns the degree of a node
    pub fn degree_of(&self, u: Node) -> Node {
        self.nbs[u as usize].len() as Node
    }

    /// Provides an iterator over all neighbors of a node
    pub fn neighbors_of(&self, u: Node) -> impl Iterator<Item = Node> {
        self.nbs[u as usize].iter().copied()
    }

    /// Provides an iterator over all neighbors of a node and the node itself
    pub fn closed_neighbors_of(&self, u: Node) -> impl Iterator<Item = Node> {
        std::iter::once(u).chain(self.neighbors_of(u))
    }

    /// Returns an iterator over all edges in the graph
    pub fn edges(&self) -> impl Iterator<Item = Edge> {
        self.vertices().flat_map(|u| {
            self.neighbors_of(u)
                .filter_map(move |v| (u < v).then_some((u, v)))
        })
    }

    /// Removes all edges at a given node
    pub fn remove_edges_at_node(&mut self, u: Node) {
        self.m -= self.degree_of(u) as usize;
        let nbs = std::mem::take(&mut self.nbs[u as usize]);

        for v in nbs {
            let i = self.neighbors_of(v).position(|x| x == u).unwrap();
            self.nbs[v as usize].swap_remove(i);
        }
    }

    /// Removes all edges (u, v) where u and v both return true for a closure f
    pub fn remove_edges_between<F: Fn(Node) -> bool>(&mut self, f: F) {
        let mut m = 0;
        for u in self.vertices() {
            if !f(u) {
                continue;
            }
            let nbs = &mut self.nbs[u as usize];
            for i in (0..nbs.len()).rev() {
                if f(nbs[i]) {
                    nbs.swap_remove(i);
                    m += 1;
                }
            }
        }

        self.m -= m / 2;
    }
}
