mod domset;
mod graph;
mod io;
mod rule1;

use smallvec::SmallVec;
use stream_bitset::prelude::{BitmaskStreamConsumer, ToBitmaskStream};
use structopt::StructOpt;

use crate::{domset::DominatingSet, graph::*, rule1::Rule1};

use std::{
    collections::HashMap,
    io::Write,
};

#[derive(StructOpt)]
struct Args {
    /// By default, the reduced graph with the produced DominatingSet is written to `stdout`.
    /// Setting this flag, disables this behaviour
    #[structopt(short = "d", long)]
    deny_graph_output: bool,
}

fn main() -> std::io::Result<()> {
    let args = Args::from_args();

    let mut stderr = std::io::stderr();

    // Read graph
    let mut graph = Graph::try_read_pace(std::io::stdin().lock())?;
    let mut domset = DominatingSet::new();
    
    // Clone graph for later
    let org_graph = graph.clone();

    writeln!(
        stderr,
        "Graph loaded n={:7} m={:8}",
        graph.len(),
        graph.num_edges()
    )?;

    // Run Rule
    Rule1::apply_rule(&graph, &mut domset);

    // Reduce graph in several steps --- this is inefficient as we want to log multiple levels of reduction
    let mut markers: Vec<SmallVec<[Node; 2]>> = vec![Default::default(); graph.len() as usize];
    for u in domset.iter() {
        for v in graph.closed_neighbors_of(u) {
            markers[v as usize].push(u);
        }
    }

    let mut naive_nodes = BitSet::new(graph.len());
    let mut linear_nodes = BitSet::new(graph.len());

    let mut node_markers: HashMap<Node, Node> = HashMap::new();
    'outer: for u in graph.vertices() {
        // Ignore DomSet-Nodes itself for deletion
        if markers[u as usize].is_empty() || markers[u as usize].contains(&u) {
            continue;
        }

        node_markers.clear();
        for &m in &markers[u as usize] {
            node_markers.insert(m, 0);
        }

        for v in graph.neighbors_of(u) {
            if markers[v as usize].is_empty() {
                continue 'outer;
            }

            for &m in &markers[v as usize] {
                node_markers.entry(m).and_modify(|num| *num += 1);
            }
        }

        for &m in &markers[u as usize] {
            if *node_markers.get(&m).unwrap() == graph.degree_of(u) {
                naive_nodes.set_bit(u);
                break;
            }
        }

        linear_nodes.set_bit(u);
    }

    let m = graph.num_edges();

    for u in naive_nodes.iter_set_bits() {
        graph.remove_edges_at_node(u);
    }

    writeln!(
        stderr,
        "{:20} n -= {:7}, m -= {:7}, |D| += {:7}",
        "Rule1-Naive",
        naive_nodes.cardinality() - 1,
        m - graph.num_edges() - 1,
        domset.len()
    )?;

    for u in (linear_nodes.bitmask_stream() - &naive_nodes).iter_set_bits() {
        graph.remove_edges_at_node(u);
    }

    writeln!(
        stderr,
        "{:20} n -= {:7}, m -= {:7}, |D| += {:7}",
        "Rule1-Linear",
        linear_nodes.cardinality() - 1,
        m - graph.num_edges() - 1,
        domset.len()
    )?;

    for u in graph.vertices() {
        if markers[u as usize].is_empty() {
            continue;
        }

        // Rule1-Plus
        if graph
            .neighbors_of(u)
            .filter(|&v| markers[v as usize].is_empty())
            .count()
            == 1
        {
            linear_nodes.set_bit(u);
            graph.remove_edges_at_node(u);
        }
    }

    writeln!(
        stderr,
        "{:20} n -= {:7}, m -= {:7}, |D| += {:7}",
        "Rule1-Plus",
        linear_nodes.cardinality() - 1,
        m - graph.num_edges() - 1,
        domset.len()
    )?;

    graph.remove_edges_between(|u| !markers[u as usize].is_empty());

    writeln!(
        stderr,
        "{:20} n -= {:7}, m -= {:7}, |D| += {:7}",
        "Rule1-Extra",
        linear_nodes.cardinality() - 1,
        m - graph.num_edges() - 1,
        domset.len()
    )?;

    // Print reduced graph and domset
    if !args.deny_graph_output {
        // Add gadgets back in 
        for u in domset.iter() {
            let nb = org_graph.neighbors_of(u).find(|&v| linear_nodes.get_bit(v)).unwrap();
            graph.add_edge(u, nb);
        }

        let mut stdout = std::io::stdout();
        writeln!(
            stdout,
            "c DomSet: {:?}",
            domset.iter().map(|u| u + 1).collect::<Vec<Node>>()
        )?;
        writeln!(stdout, "p ds {} {}", graph.len(), graph.num_edges())?;
        for (u, v) in graph.edges() {
            writeln!(stdout, "{} {}", u + 1, v + 1)?;
        }
    }

    Ok(())
}
