mod domset;
mod graph;
mod io;
mod iter_greedy;
mod marker;
mod radix;
mod rule1;

use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;

use crate::{
    domset::DominatingSet,
    graph::*,
    iter_greedy::run_greedy,
    rule1::{Rule1, Rule1Naive},
};

use std::{
    io::{Read, Write},
    time::Instant,
};

macro_rules! time {
    ($code:block) => {
        {
            let timer = Instant::now();
            $code
            timer.elapsed().as_nanos()
        }
    };
}

#[allow(clippy::too_many_arguments)]
fn report(
    name: &str,
    n: usize,
    m: usize,
    d: usize,
    c: NumNodes,
    t: u128,
    gd: usize,
    gt: u128,
) -> std::io::Result<()> {
    writeln!(
        std::io::stderr(),
        "{name:25} n -= {n:10}, m -= {m:10}, |D| += {d:10}, |covered| += {c:10}, Time[ns] = {t:12}, |Greedy-D| = {gd:10}, Greedy-Time[ns] = {gt:12}"
    )
}

fn main() -> std::io::Result<()> {
    let mut stderr = std::io::stderr();

    let greedy_iter = 10u128;
    let mut rng = Pcg64Mcg::seed_from_u64(3);

    // Read Graph

    // Load into buffer to prevent external factors in measurement
    let mut buffer = vec![];
    std::io::stdin().lock().read_to_end(&mut buffer)?;

    let timer = Instant::now();
    let org_graph = Graph::try_read_pace(std::io::Cursor::new(buffer))?;
    let read_time = timer.elapsed().as_nanos();

    writeln!(
        stderr,
        "Graph loaded n={:7} m={:8} in {:12}ns",
        org_graph.len(),
        org_graph.num_edges(),
        read_time
    )?;

    // Fix Singletons

    let mut graph = org_graph.clone();
    let mut domset_st = DominatingSet::new(graph.len());
    let mut covered_st = BitSet::new(graph.len());

    let singleton_time = time!({
        // Fix Singletons
        graph.vertices().for_each(|u| {
            if graph.degree_of(u) == 0 {
                domset_st.add_node(u);
                covered_st.set_bit(u);
            }
        });
    });

    // Run Greedy without Reduction

    let (greedy_ds, greedy_time) =
        run_greedy(&graph, &mut rng, &domset_st, &covered_st, greedy_iter);

    assert!(greedy_ds.is_valid(&org_graph));

    report(
        "Singletons",
        0,
        0,
        domset_st.len(),
        covered_st.cardinality(),
        singleton_time,
        greedy_ds.len(),
        greedy_time,
    )?;

    // Run Naive Rule1 (and Greedy)

    let mut naive_ds = domset_st.clone();
    let mut naive_cov = covered_st.clone();

    let before_nodes = graph.vertices_with_neighbors().count();
    let before_edges = graph.num_edges();
    let before_in_domset = naive_ds.len();
    let before_covered = naive_cov.cardinality();

    let finished_in_time;

    let naive_time = time!({
        finished_in_time = Rule1Naive::apply(&mut graph, &mut naive_ds, &mut naive_cov);
    });

    if finished_in_time {
        let after_nodes = graph.vertices_with_neighbors().count();
        let after_edges = graph.num_edges();
        let after_in_domset = naive_ds.len();
        let after_covered = naive_cov.cardinality();
    
        for u in naive_ds.iter() {
            graph.remove_edges_at_node(u);
        }

        let (greedy_ds, greedy_time) =
            run_greedy(&graph, &mut rng, &naive_ds, &naive_cov, greedy_iter);
        
        assert!(greedy_ds.is_valid(&org_graph));

        report(
            "Naive",
            before_nodes - after_nodes,
            before_edges - after_edges,
            after_in_domset - before_in_domset,
            after_covered - before_covered,
            naive_time,
            greedy_ds.len(),
            greedy_time,
        )?;
    } else {
        writeln!(stderr, "Naive did not finish in time!")?;
    }

    // Initialize Rule1 Datastructures

    let timer = Instant::now();
    let mut rule1 = Rule1::new(org_graph.len());
    let init_time = timer.elapsed().as_nanos();

    // Run Rule1

    let graph = org_graph.clone();

    let mut ds = domset_st.clone();
    let mut cov = covered_st.clone();

    let before_nodes = graph.vertices_with_neighbors().count();
    let before_edges = graph.num_edges();
    let before_in_domset = ds.len();
    let before_covered = cov.cardinality();

    let rule_time = time!({
        rule1.apply_rule::<false>(&graph, &mut ds, &mut cov);
    });

    // Apply Linear-Removal

    let mut linear_graph = graph.clone();
    let linear_ds = ds.clone();
    let linear_cov = cov.clone();

    let linear_time = time!({
        rule1.linear_removal(&mut linear_graph, &linear_ds, &linear_cov);
    });

    let after_nodes = linear_graph.vertices_with_neighbors().count();
    let after_edges = linear_graph.num_edges();
    let after_in_domset = linear_ds.len();
    let after_covered = linear_cov.cardinality();

    let (greedy_ds, greedy_time) = run_greedy(
        &linear_graph,
        &mut rng,
        &linear_ds,
        &linear_cov,
        greedy_iter,
    );
    assert!(greedy_ds.is_valid(&org_graph));

    report(
        "Linear",
        before_nodes - after_nodes,
        before_edges - after_edges,
        after_in_domset - before_in_domset,
        after_covered - before_covered,
        linear_time + rule_time + init_time,
        greedy_ds.len(),
        greedy_time,
    )?;

    // Apply Plus-Removal

    let mut plus_graph = graph.clone();
    let mut plus_ds = ds.clone();
    let mut plus_cov = cov.clone();

    let plus_time = time!({
        rule1.plus_removal(&mut plus_graph, &mut plus_ds, &mut plus_cov);
    });

    let after_nodes = plus_graph.vertices_with_neighbors().count();
    let after_edges = plus_graph.num_edges();
    let after_in_domset = plus_ds.len();
    let after_covered = plus_cov.cardinality();

    let (greedy_ds, greedy_time) =
        run_greedy(&plus_graph, &mut rng, &plus_ds, &plus_cov, greedy_iter);
    assert!(greedy_ds.is_valid(&org_graph));

    report(
        "Plus",
        before_nodes - after_nodes,
        before_edges - after_edges,
        after_in_domset - before_in_domset,
        after_covered - before_covered,
        plus_time + rule_time + init_time,
        greedy_ds.len(),
        greedy_time,
    )?;

    // Apply Extra-Removal

    let mut extra_graph = graph.clone();
    let mut extra_ds = ds.clone();
    let mut extra_cov = cov.clone();

    let extra_time = time!({
        rule1.extra_removal(&mut extra_graph, &mut extra_ds, &mut extra_cov);
    });

    let after_nodes = extra_graph.vertices_with_neighbors().count();
    let after_edges = extra_graph.num_edges();
    let after_in_domset = extra_ds.len();
    let after_covered = extra_cov.cardinality();

    let (greedy_ds, greedy_time) =
        run_greedy(&extra_graph, &mut rng, &extra_ds, &extra_cov, greedy_iter);
    assert!(greedy_ds.is_valid(&org_graph));

    report(
        "Extra[1]",
        before_nodes - after_nodes,
        before_edges - after_edges,
        after_in_domset - before_in_domset,
        after_covered - before_covered,
        extra_time + rule_time + init_time,
        greedy_ds.len(),
        greedy_time,
    )?;

    // Apply Extra exhaustively

    for i in 2.. {
        let before_nodes = extra_graph.vertices_with_neighbors().count();
        let before_edges = extra_graph.num_edges();
        let before_in_domset = extra_ds.len();
        let before_covered = extra_cov.cardinality();

        let extra_time = time!({
            rule1.apply_rule::<true>(&extra_graph, &mut extra_ds, &mut extra_cov);
            rule1.extra_removal(&mut extra_graph, &mut extra_ds, &mut extra_cov);
        });

        let after_nodes = extra_graph.vertices_with_neighbors().count();
        let after_edges = extra_graph.num_edges();
        let after_in_domset = extra_ds.len();
        let after_covered = extra_cov.cardinality();

        // No Change happened => abort
        if before_in_domset == after_in_domset {
            break;
        }

        let (greedy_ds, greedy_time) =
            run_greedy(&extra_graph, &mut rng, &extra_ds, &extra_cov, greedy_iter);
        assert!(greedy_ds.is_valid(&org_graph));

        report(
            format!("Extra[{i}]").as_str(),
            before_nodes - after_nodes,
            before_edges - after_edges,
            after_in_domset - before_in_domset,
            after_covered - before_covered,
            extra_time,
            greedy_ds.len(),
            greedy_time,
        )?;
    }

    Ok(())
}
