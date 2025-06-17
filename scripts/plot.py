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
    "rule_fixed": [],
    "rule_nodes_removed": [],
    "rule_edges_removed": []
}

load_pattern = re.compile(r"Graph loaded n=\s*(\d+) m=\s*(\d+)")
rule_pattern = re.compile(
    r"(\w+)\s* n\s*-=\s*(\d+), m\s*-=\s*(\d+), \|D\|\s*\+=\s*(\d+)"
)

for file in os.listdir(args.datadir):
    with open(f"{args.datadir}/{file}", "r") as f:
        n, m = 0, 0 
        rules = []
        for line in f:
            p = re.search(load_pattern, line)
            if p is not None:
                n = int(p.group(1))
                m = int(p.group(2))

            p = re.search(rule_pattern, line)
            if p is not None:
                rules.append((
                    p.group(1),
                    int(p.group(2)),
                    int(p.group(3)),
                    int(p.group(4))
                ))

        for rule in rules:
            data["n"].append(n)
            data["m"].append(m)
            data["rule_name"].append(rule[0])
            data["rule_fixed"].append(rule[3])
            data["rule_nodes_removed"].append(rule[1])
            data["rule_edges_removed"].append(rule[2])

data = pd.DataFrame.from_dict(data)

print(data)

data = data[data.rule_fixed > 0]
data["frac_fixed"] = data["rule_fixed"] / data["n"]
data["frac_nodes"] = data["rule_nodes_removed"] / data["n"]
data["frac_edges"] = data["rule_edges_removed"] / data["m"]

sns.set_theme(style="whitegrid")
sns.set_palette("colorblind")
plt.rcParams["text.usetex"] = True
plt.rcParams["figure.figsize"] = 7, 4

plot = sns.scatterplot(
    data[data.rule_name != "Extra"],
    x="n",
    y="frac_nodes",
    hue="rule_name",
    palette="colorblind",
    style='rule_name',
    markers=["o", "v", "X"]
)

plot.set(
    xlabel=r"\textsc{Input: Number of Nodes}",
    ylabel=r"\textsc{Fraction of Removed Nodes}"
)

handles, labels = plot.get_legend_handles_labels()
labels = [
    r"\textsc{Naive}",
    r"\textsc{Linear}",
    r"\textsc{Plus}"
]

plt.legend(handles[::-1], labels[::-1], title=r"\textsc{Rule1}")

plt.savefig(
    f"{args.outpath}/nodes.pdf",
    format="pdf",
    bbox_inches="tight"
)

plt.clf()

plot = sns.scatterplot(
    data,
    x="m",
    y="frac_edges",
    hue="rule_name",
    palette="colorblind",
    style='rule_name',
    markers=["o", "v", "X", "p"]
)


plot.set(
    xlabel=r"\textsc{Input: Number of Edges}",
    ylabel=r"\textsc{Fraction of Removed Edges}"
)

handles, labels = plot.get_legend_handles_labels()
labels = [
    r"\textsc{Naive}",
    r"\textsc{Linear}",
    r"\textsc{Plus}",
    r"\textsc{Extra}"
]

plt.legend(handles[::-1], labels[::-1], title=r"\textsc{Rule1}")

plt.savefig(
    f"{args.outpath}/edges.pdf",
    format="pdf",
    bbox_inches="tight"
)
