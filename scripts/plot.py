#!/usr/bin/env python3
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms
from matplotlib.ticker import LogLocator, ScalarFormatter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse as cli
import os
import re

parser = cli.ArgumentParser()
parser.add_argument("datadir")
parser.add_argument("outpath")

args = parser.parse_args()

data = {
    "n": [],
    "m": [],
    "rule_name": [],
    "rule_nodes_removed": [],
    "rule_edges_removed": [],
    "rule_fixed": [],
    "time": [],
    "naive_nodes": [],
    "naive_edges": [],
    "naive_time": [],
}

load_pattern = re.compile(r"Graph loaded n=\s*(\d+) m=\s*(\d+) in\s*(\d+)ns")
rule_pattern = re.compile(
    r"(\w+)\s* n\s*-=\s*(\d+), m\s*-=\s*(\d+), \|D\|\s*\+=\s*(\d+), \|covered\|\s*\+=\s*(\d+), Time\[ns\]\s*=\s*(\d+), \|Greedy-D\|\s*=\s*(\d+), Greedy-Time\[ns\]\s*=\s*(\d+)"
)
extra_pattern = re.compile(
    r"Extra\[(\d+)\]\s* n\s*-=\s*(\d+), m\s*-=\s*(\d+), \|D\|\s*\+=\s*(\d+), \|covered\|\s*\+=\s*(\d+), Time\[ns\]\s*=\s*(\d+), \|Greedy-D\|\s*=\s*(\d+), Greedy-Time\[ns\]\s*=\s*(\d+)"
)


def parse_file(path):
    data = {"rules": {}, "extra": []}

    with open(path, "r") as f:
        for line in f:
            p = re.search(load_pattern, line)
            if p is not None:
                data["n"] = int(p.group(1))
                data["m"] = int(p.group(2))
                data["read_time"] = int(p.group(3))
                continue

            p = re.search(extra_pattern, line)
            if p is not None:
                data["extra"].append(
                    {
                        "iter": int(p.group(1)),
                        "n": int(p.group(2)),
                        "m": int(p.group(3)),
                        "D": int(p.group(4)),
                        "covered": int(p.group(5)),
                        "time": int(p.group(6)),
                        "greedy_domset": int(p.group(7)),
                        "greedy_time": int(p.group(8)),
                    }
                )

                # Insert into Rules if it is the first Extra-Iteration
                if int(p.group(1)) == 1:
                    data["rules"]["Extra"] = {
                        "n": int(p.group(2)),
                        "m": int(p.group(3)),
                        "D": int(p.group(4)),
                        "covered": int(p.group(5)),
                        "time": int(p.group(6)),
                        "greedy_domset": int(p.group(7)),
                        "greedy_time": int(p.group(8)),
                    }
                continue

            p = re.search(rule_pattern, line)
            if p is not None:
                data["rules"][p.group(1)] = {
                    "n": int(p.group(2)),
                    "m": int(p.group(3)),
                    "D": int(p.group(4)),
                    "covered": int(p.group(5)),
                    "time": int(p.group(6)),
                    "greedy_domset": int(p.group(7)),
                    "greedy_time": int(p.group(8)),
                }
                continue
    return data


data = {
    "n": [],
    "m": [],
    "iter": [],
    "read_time": [],
    "rule_name": [],
    "rule_n": [],
    "rule_m": [],
    "rule_D": [],
    "rule_covered": [],
    "rule_time": [],
    "greedy_D": [],
    "greedy_time": [],
    # Speedups to Naive
    "su_n": [],
    "su_m": [],
    "su_D": [],
    "su_covered": [],
    "su_time": [],
    "su_gD": [],
    "su_gtD": [],
    "su_gt": [],
    "su_tT": [],
    # Speedups to Singletons
    "st_D": [],
    "st_t": [],
    "st_T": [],
    "st_tD": [],
}

extra_stats = {
    "m": [],
    "num_iters_faster_than_naive": [],
    "num_iters_faster_than_empty": [],
}

no_naive_times = {"Linear": [], "Plus": [], "Extra": []}

for file in os.listdir(args.datadir):
    out = parse_file(f"{args.datadir}/{file}")

    # Something went wrong here and the file does not include a full dataset
    if any(
        rule not in out["rules"]
        for rule in ["Singletons", "Naive", "Linear", "Plus", "Extra"]
    ):
        # Check if only Naive was not run
        if all(rule in out["rules"] for rule in ["Linear", "Plus", "Extra"]):
            for rule in ["Linear", "Plus", "Extra"]:
                no_naive_times[rule].append(out["rules"][rule]["time"])
        continue

    empty = out["rules"]["Singletons"]
    naive = out["rules"]["Naive"]

    # No Type3 was found, disregard this entry
    if naive["D"] == 0:
        continue

    for rule_name, rule in out["rules"].items():
        # Only used for speedups
        if rule_name == "Singletons":
            continue

        data["n"].append(out["n"])
        data["m"].append(out["m"])
        data["iter"].append(1)
        data["read_time"].append(out["read_time"])
        data["rule_name"].append(rule_name)

        data["rule_n"].append(rule["n"])
        data["rule_m"].append(rule["m"])
        data["rule_D"].append(rule["D"])
        data["rule_covered"].append(rule["covered"])
        data["rule_time"].append(rule["time"])
        data["greedy_D"].append(rule["greedy_domset"])
        data["greedy_time"].append(rule["greedy_time"])

        data["su_n"].append(rule["n"] / naive["n"])
        data["su_m"].append(rule["m"] / naive["m"])
        data["su_D"].append(rule["D"] / naive["D"])
        data["su_covered"].append(rule["covered"] / naive["covered"])
        data["su_time"].append(naive["time"] / rule["time"])
        data["su_gD"].append(naive["greedy_domset"] / rule["greedy_domset"])
        data["su_gtD"].append(naive["greedy_domset"] - rule["greedy_domset"])
        data["su_gt"].append(naive["greedy_time"] / rule["greedy_time"])
        data["su_tT"].append(naive["time"] - rule["time"])

        data["st_D"].append(empty["greedy_domset"] / rule["greedy_domset"])
        data["st_t"].append(empty["greedy_time"] / rule["greedy_time"])
        data["st_T"].append(empty["greedy_time"] / (rule["time"] + rule["greedy_time"]))
        data["st_tD"].append(empty["greedy_domset"] - rule["greedy_domset"])

    times = {}
    for rule in out["extra"]:
        times[rule["iter"]] = (rule["time"], rule["greedy_time"])

        if rule["iter"] == 1:
            continue

        data["n"].append(out["n"])
        data["m"].append(out["m"])
        data["iter"].append(rule["iter"])
        data["read_time"].append(out["read_time"])
        data["rule_name"].append("Extra")

        data["rule_n"].append(rule["n"])
        data["rule_m"].append(rule["m"])
        data["rule_D"].append(rule["D"])
        data["rule_covered"].append(rule["covered"])
        data["rule_time"].append(rule["time"])
        data["greedy_D"].append(rule["greedy_domset"])
        data["greedy_time"].append(rule["greedy_time"])

        data["su_n"].append(rule["n"] / naive["n"])
        data["su_m"].append(rule["m"] / naive["m"])
        data["su_D"].append(rule["D"] / naive["D"])
        data["su_covered"].append(rule["covered"] / naive["covered"])
        data["su_time"].append(naive["time"] / rule["time"])
        data["su_gD"].append(naive["greedy_domset"] / rule["greedy_domset"])
        data["su_gtD"].append(naive["greedy_domset"] - rule["greedy_domset"])
        data["su_gt"].append(naive["greedy_time"] / rule["greedy_time"])
        data["su_tT"].append(naive["time"] - rule["time"])

        data["st_D"].append(empty["greedy_domset"] / rule["greedy_domset"])
        data["st_t"].append(empty["greedy_time"] / rule["greedy_time"])
        data["st_T"].append(empty["greedy_time"] / (rule["time"] + rule["greedy_time"]))
        data["st_tD"].append(empty["greedy_domset"] - rule["greedy_domset"])

    iter = 1

    iter_naive = None
    iter_empty = None
    cum_rule_time = 0
    while True:
        if iter not in times or (iter_naive is not None and iter_empty is not None):
            if iter_naive is None:
                iter_naive = iter

            if iter_empty is None:
                iter_empty = iter

            break
        (rt, gt) = times[iter]

        cum_rule_time += rt

        if iter_naive is None and cum_rule_time > naive["time"]:
            iter_naive = iter - 1

        if iter_empty is None and cum_rule_time + gt > empty["greedy_time"]:
            iter_empty = iter - 1

        iter += 1

    extra_stats["m"].append(out["m"])
    extra_stats["num_iters_faster_than_naive"].append(iter_naive)
    extra_stats["num_iters_faster_than_empty"].append(iter_empty)


data = pd.DataFrame.from_dict(data)
data = data[data.m > 10000]

data["density"] = (2 * data["m"]) / (data["n"] * (data["n"] - 1))
data["frac_n"] = data["rule_n"] / data["n"]
data["frac_m"] = data["rule_m"] / data["m"]
data["frac_D"] = data["rule_D"] / data["n"]
data["frac_covered"] = data["rule_covered"] / data["n"]

print(data["m"].max())

extra_data = data[data.rule_name == "Extra"]
data = data[data.iter == 1]


def print_mean(data, rule, stat, label):
    print(
        "- {} (m > 1e4): {}\n- {} (m > 1e6): {}".format(
            label,
            data[data.rule_name == rule][stat].mean(),
            label,
            data[(data.rule_name == rule) & (data.m > 1000000)][stat].mean(),
        )
    )


def print_median(data, rule, stat, label):
    print(
        "- {} (m > 1e4) (Med): {}\n- {} (m > 1e6) (Med): {}".format(
            label,
            data[data.rule_name == rule][stat].median(),
            label,
            data[(data.rule_name == rule) & (data.m > 1000000)][stat].median(),
        )
    )


for rule in ["Linear", "Plus", "Extra"]:
    print(f"Average {rule} Speedups")
    for stat, label in [
        ("su_time", "Time[Naive]"),
        ("su_n", "RemovedNodes[Naive]"),
        ("su_m", "RemovedEdges[Naive]"),
        ("su_gD", "GreedyDomset[Naive]"),
        ("su_gtD", "TotalGreedyDomset[Naive]"),
        ("st_D", "GreedyDomset[NoRule]"),
        ("st_T", "TotalTime[NoRule]"),
        ("st_tD", "TotalGreedyDomset[NoRule]"),
    ]:
        print_mean(data, rule, stat, label)
        print_median(data, rule, stat, label)
    print()


no_naive_times = pd.DataFrame.from_dict(no_naive_times)
print("In ", len(no_naive_times), " Instances, Naive did not finish in 1 hour, but")
for rule in ["Linear", "Plus", "Extra"]:
    print(
        "- ",
        rule,
        "finished on average in ",
        no_naive_times[rule].mean() / 1000000000,
        " seconds!",
    )
print()

extra_stats = pd.DataFrame.from_dict(extra_stats)
print(
    "Extra-Stats",
    "\n- Average Number of Iterations until Naive was faster (m > 1e4): ",
    extra_stats[extra_stats.m > 10000]["num_iters_faster_than_naive"].mean(),
    "\n- Average Number of Iterations until Naive was faster (m > 1e6): ",
    extra_stats[extra_stats.m > 1000000]["num_iters_faster_than_naive"].mean(),
    "\n- Average Number of Iterations until Empty was faster (m > 1e4): ",
    extra_stats[extra_stats.m > 10000]["num_iters_faster_than_empty"].mean(),
    "\n- Average Number of Iterations until Empty was faster (m > 1e6): ",
    extra_stats[extra_stats.m > 1000000]["num_iters_faster_than_empty"].mean(),
)

print(
    "\nNaive was faster than Extra by a total of ",
    -data[data.su_time < 1.0]["su_tT"].mean() / 1000000000,
    "s (avg) and ",
    -data[data.su_time < 1.0]["su_tT"].min() / 1000000000,
    "s (max)",
)

sns_palette = sns.color_palette("colorblind")
sns_palette = [sns_palette[0], sns_palette[1], sns_palette[4]]


def plot_distr(data, x, x_label, extra, name):
    plt.clf()

    sns.set_theme(style="whitegrid")
    sns.set_palette("colorblind")
    plt.rcParams["text.usetex"] = True
    plt.rcParams["figure.figsize"] = 7, 4

    filtered = data[data.rule_name != "Naive"]
    if not extra:
        filtered = filtered[filtered.rule_name != "Extra"]
    bins = np.logspace(np.log10(1), np.log10(data[x].max()), 50)

    hue_order = ["Linear", "Plus"]
    if extra:
        hue_order += ["Extra"]

    len = 2 + int(extra)
    palette = sns_palette[:len]

    plot = sns.histplot(
        data=filtered,
        x=x,
        hue="rule_name",
        hue_order=hue_order,
        palette=palette,
        bins=bins,
        element="step",
        stat="count",
        common_norm=False,
    )

    plt.xscale("log")
    plt.yscale("log")

    plot.set(
        ylabel=r"\textsc{Number of Instances}",
        xlabel=x_label,
    )

    handles = [
        Line2D([0], [0], color=sns_palette[0], lw=2, label="Linear"),
        Line2D([0], [0], color=sns_palette[1], lw=2, label="Plus"),
    ]
    labels = [r"\textsc{Linear}", r"\textsc{Plus}"]

    if extra:
        handles += [Line2D([0], [0], color=sns_palette[2], lw=2, label="Extra")]
        labels += [r"\textsc{Extra}"]

    plt.legend(handles[::-1], labels[::-1], title=r"\textsc{Rule1}")

    plt.savefig(f"{args.outpath}/{name}.pdf", format="pdf", bbox_inches="tight")


plot_distr(
    data,
    "su_n",
    r"\textsc{RemovedNodes[Rule1] / RemovedNodes[Naive]}",
    False,
    "nodes",
)

plot_distr(
    data,
    "su_m",
    r"\textsc{RemovedEdges[Rule1] / RemovedEdges[Naive]}",
    True,
    "edges",
)

plot_distr(
    data,
    "st_tD",
    r"\textsc{GreedyBefore[Rule1] - GreedyAfter[Rule1]}",
    True,
    "empty",
)


plt.clf()

sns.set_theme(style="whitegrid")
sns.set_palette("colorblind")
plt.rcParams["text.usetex"] = True
plt.rcParams["figure.figsize"] = 7, 4

plot = sns.scatterplot(
    data[data.rule_name != "Naive"],
    x="m",
    y="su_time",
    hue="rule_name",
    hue_order=["Linear", "Plus", "Extra"],
    palette=sns_palette,
    style="rule_name",
    markers=["o", "v", "X"],
    alpha=0.7,
)

plt.xscale("log")
plt.yscale("log")

plot.set(
    xlabel=r"\textsc{Input: Number of Edges}",
    ylabel=r"\textsc{Speedup to Naive}",
)

handles, labels = plot.get_legend_handles_labels()
labels = [r"\textsc{Linear}", r"\textsc{Plus}", r"\textsc{Extra}"]

plt.legend(handles[::-1], labels[::-1], title=r"\textsc{Rule1}")

plt.savefig(f"{args.outpath}/time.pdf", format="pdf", bbox_inches="tight")


plt.clf()

sns.set_theme(style="whitegrid")
sns.set_palette("colorblind")
plt.rcParams["text.usetex"] = True
plt.rcParams["figure.figsize"] = 7, 4

plot = sns.lineplot(
    data=extra_data,
    x="iter",
    y="rule_D",
    estimator=np.median,
    label=r"\textsc{Median}",
)

sns.lineplot(
    data=extra_data, x="iter", y="rule_D", estimator=np.mean, label=r"\textsc{Mean}"
)

sns.lineplot(
    data=extra_data,
    x="iter",
    y="rule_D",
    estimator=np.max,
    label=r"\textsc{Max}",
)

plt.xscale("log")
plt.yscale("log")

plot.set(
    xlabel=r"\textsc{Iteration of Rule1-Extra}",
    ylabel=r"\textsc{Total Increase in $|D|$}",
)

plt.savefig(f"{args.outpath}/extra_iters.pdf", format="pdf", bbox_inches="tight")
