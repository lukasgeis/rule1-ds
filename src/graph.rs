use std::ops::Range;

use stream_bitset::bitset::BitSetImpl;

pub type Node = u32;
pub type Edge = (Node, Node);

pub type NumNodes = Node;

pub type BitSet = BitSetImpl<Node>;

/// Minimal Graph-Representation using AdjacencyLists
#[derive(Clone)]
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
    pub fn len(&self) -> NumNodes {
        self.nbs.len() as NumNodes
    }

    /// Returns the number of edges in the graph
    pub fn num_edges(&self) -> usize {
        self.m
    }

    /// Provides a range over all nodes in the graph
    pub fn vertices(&self) -> Range<Node> {
        0..self.len()
    }

    /// Provides a range over all nodes in the graph that have at least one incident edge
    pub fn vertices_with_neighbors(&self) -> impl Iterator<Item = Node> {
        self.vertices()
            .filter(|&u| !self.nbs[u as usize].is_empty())
    }

    /// Returns the degree of a node
    pub fn degree_of(&self, u: Node) -> NumNodes {
        self.nbs[u as usize].len() as NumNodes
    }

    /// Returns the maximum degree in the graph
    pub fn max_degree(&self) -> NumNodes {
        self.vertices().map(|u| self.degree_of(u)).max().unwrap()
    }

    /// Provides an iterator over all neighbors of a node
    pub fn neighbors_of(&self, u: Node) -> impl Iterator<Item = Node> {
        self.nbs[u as usize].iter().copied()
    }

    /// Provides the ith neighbor of u.
    /// Panics if there is no such neighbor.
    pub fn ith_neighbor(&self, u: Node, i: NumNodes) -> Node {
        self.nbs[u as usize][i as usize]
    }

    /// Provides an iterator over all neighbors of a node and the node itself
    pub fn closed_neighbors_of(&self, u: Node) -> impl Iterator<Item = Node> {
        std::iter::once(u).chain(self.neighbors_of(u))
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

    /// Removes all edges at a set of nodes
    pub fn remove_edges_at_nodes(&mut self, nodes: &BitSet) {
        let mut removed_half_edges = 0;

        for u in self.vertices() {
            let nbs = &mut self.nbs[u as usize];
            if nodes.get_bit(u) {
                removed_half_edges += nbs.len();
                nbs.clear();
                continue;
            }

            for i in (0..nbs.len()).rev() {
                if nodes.get_bit(nbs[i]) {
                    nbs.swap_remove(i);
                    removed_half_edges += 1;
                }
            }
        }

        self.m -= removed_half_edges / 2;
    }

    /// Removes all edges (u, v) where u and v both return true for a closure f
    pub fn remove_edges_between(&mut self, nodes: &BitSet) {
        let mut m = 0;
        for u in nodes.iter_set_bits() {
            let nbs = &mut self.nbs[u as usize];
            for i in (0..nbs.len()).rev() {
                if nodes.get_bit(nbs[i]) {
                    nbs.swap_remove(i);
                    m += 1;
                }
            }
        }

        self.m -= m / 2;
    }
}
