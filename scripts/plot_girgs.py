import numpy as np
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
    # Speedups to Singletons
    "st_D": [],
    "st_t": [],
    "st_T": [],
    "st_tD": [],
}


for file in os.listdir(args.datadir):
    out = parse_file(f"{args.datadir}/{file}")

    # Something went wrong here and the file does not include a full dataset
    if any(
        rule not in out["rules"]
        for rule in ["Singletons", "Naive", "Linear", "Plus", "Extra"]
    ):
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

        data["st_D"].append(empty["greedy_domset"] / rule["greedy_domset"])
        data["st_t"].append(empty["greedy_time"] / rule["greedy_time"])
        data["st_T"].append(empty["greedy_time"] / (rule["time"] + rule["greedy_time"]))
        data["st_tD"].append(empty["greedy_domset"] - rule["greedy_domset"])


data = pd.DataFrame.from_dict(data)

data["frac_n"] = data["rule_n"] / data["n"]
data["frac_m"] = data["rule_m"] / data["m"]
data["frac_D"] = data["rule_D"] / data["n"]
data["frac_covered"] = data["rule_covered"] / data["n"]


def print_stat(data, rule, stat, label):
    print(
        "- {} (m > 1e4): {}\n- {} (m > 1e6): {}".format(
            label,
            data[data.rule_name == rule][stat].mean(),
            label,
            data[(data.rule_name == rule) & (data.m > 1000000)][stat].mean(),
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
        print_stat(data, rule, stat, label)
    print()

plt.clf()

sns.set_theme(style="whitegrid")
sns.set_palette("colorblind")
plt.rcParams["text.usetex"] = True
plt.rcParams["figure.figsize"] = 7, 4

sns_palette = sns.color_palette("colorblind")
sns_palette = [sns_palette[0], sns_palette[1], sns_palette[4]]

plot = sns.scatterplot(
    data[data.rule_name != "Naive"],
    x="m",
    y="su_time",
    hue="rule_name",
    hue_order=["Linear", "Plus", "Extra"],
    palette=sns_palette,
    style="rule_name",
    markers=["o", "v", "X"],
)

plt.xscale("log")
plt.yscale("log")

plot.set(
    xlabel=r"\textsc{Input: Number of Edges}",
    ylabel=r"\textsc{Speedup to Naive}",
)

handles, labels = plot.get_legend_handles_labels()
labels = [r"\textsc{Linear}", r"\textsc{Plus}", r"\textsc{Extra}"]

legend = plt.legend(handles[::-1], labels[::-1], title=r"\textsc{Rule1}", loc=2)

x_vals = [0.5 * 10**6, 10**8]
(l1,) = plt.plot(x_vals, [0.00015 * (x) ** 0.72 for x in x_vals], color="blue")
handles, labels = plot.get_legend_handles_labels()
plt.legend([l1], [r"$f(x)=0.00015 x^{0.72}$"], loc=4)

plt.savefig(f"{args.outpath}/time_girgs.pdf", format="pdf", bbox_inches="tight")
