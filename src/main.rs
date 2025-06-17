mod domset;
mod graph;
mod io;
mod rule1;

use structopt::StructOpt;

use crate::{domset::DominatingSet, graph::*, rule1::Rule1};

use std::{io::Write, time::Instant};

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
    let org_graph = Graph::try_read_pace(std::io::stdin().lock())?;

    writeln!(
        stderr,
        "Graph loaded n={:7} m={:8}",
        org_graph.len(),
        org_graph.num_edges()
    )?;

    // Run Rules

    let mut graph = org_graph.clone();
    let mut timer = Instant::now();
    let mut domset = DominatingSet::new();
    let mut rm = Rule1::apply_naive(&mut graph, &mut domset);
    let mut time = timer.elapsed().as_nanos();

    writeln!(
        stderr,
        "{:20} n -= {:7}, m -= {:7}, |D| += {:7}, Time = {:10}",
        "Rule1-Naive",
        rm.cardinality(),
        org_graph.num_edges() - graph.num_edges(),
        domset.len(),
        time
    )?;

    graph = org_graph.clone();
    timer = Instant::now();
    domset = DominatingSet::new();
    rm = Rule1::apply_linear(&mut graph, &mut domset);
    time = timer.elapsed().as_nanos();

    writeln!(
        stderr,
        "{:20} n -= {:7}, m -= {:7}, |D| += {:7}, Time = {:10}",
        "Rule1-Linear",
        rm.cardinality(),
        org_graph.num_edges() - graph.num_edges(),
        domset.len(),
        time
    )?;

    graph = org_graph.clone();
    timer = Instant::now();
    domset = DominatingSet::new();
    rm = Rule1::apply_plus(&mut graph, &mut domset);
    time = timer.elapsed().as_nanos();

    writeln!(
        stderr,
        "{:20} n -= {:7}, m -= {:7}, |D| += {:7}, Time = {:10}",
        "Rule1-Plus",
        rm.cardinality(),
        org_graph.num_edges() - graph.num_edges(),
        domset.len(),
        time
    )?;

    graph = org_graph.clone();
    timer = Instant::now();
    domset = DominatingSet::new();
    rm = Rule1::apply_extra(&mut graph, &mut domset);
    time = timer.elapsed().as_nanos();

    writeln!(
        stderr,
        "{:20} n -= {:7}, m -= {:7}, |D| += {:7}, Time = {:10}",
        "Rule1-Extra",
        rm.cardinality(),
        org_graph.num_edges() - graph.num_edges(),
        domset.len(),
        time
    )?;

    // Print reduced graph and domset
    if !args.deny_graph_output {
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
