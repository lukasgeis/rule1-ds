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
    "su_gt": [],
    # Speedups to Singletons
    "st_D": [],
    "st_t": [],
    "st_T": [],
    "st_tD": []
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
        data["su_gt"].append(naive["greedy_time"] / rule["greedy_time"])

        data["st_D"].append(empty["greedy_domset"] / rule["greedy_domset"])
        data["st_t"].append(empty["greedy_time"] / rule["greedy_time"])
        data["st_T"].append(empty["greedy_time"] / (rule["time"] + rule["greedy_time"]))
        data["st_tD"].append(empty["greedy_domset"] - rule["greedy_domset"])

    for rule in out["extra"]:
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
        data["su_gt"].append(naive["greedy_time"] / rule["greedy_time"])

        data["st_D"].append(empty["greedy_domset"] / rule["greedy_domset"])
        data["st_t"].append(empty["greedy_time"] / rule["greedy_time"])
        data["st_T"].append(empty["greedy_time"] / (rule["time"] + rule["greedy_time"]))
        data["st_tD"].append(empty["greedy_domset"] - rule["greedy_domset"])


data = pd.DataFrame.from_dict(data)
data = data[data.m > 10000]

data["frac_n"] = data["rule_n"] / data["n"]
data["frac_m"] = data["rule_m"] / data["m"]
data["frac_D"] = data["rule_D"] / data["n"]
data["frac_covered"] = data["rule_covered"] / data["n"]

extra_data = data[data.rule_name == "Extra"]
data = data[data.iter == 1]

print("Speedups:")
for rule in ["Linear", "Plus", "Extra"]:
    print(
        rule,
        "\n- time[Naive]: ",
        data[data.rule_name == rule]["su_time"].mean(),
        "\n- n[Naive]: ",
        data[data.rule_name == rule]["su_n"].mean(),
        "\n- m[Naive]: ",
        data[data.rule_name == rule]["su_m"].mean(),
        "\n- D[Naive]: ",
        data[data.rule_name == rule]["su_D"].mean(),
        "\n- covered[Naive]: ",
        data[data.rule_name == rule]["su_covered"].mean(),
        "\n- GreedyD[Naive]: ",
        data[data.rule_name == rule]["su_gD"].mean(),
        "\n- GreedyTime[Naive]: ",
        data[data.rule_name == rule]["su_gt"].mean(),
        "\n- GreedyD[Empty]: ",
        data[data.rule_name == rule]["st_D"].mean(),
        "\n- GreedyTime[Empty]: ",
        data[data.rule_name == rule]["st_t"].mean(),
        "\n- Time[Empty]: ",
        data[data.rule_name == rule]["st_T"].mean(),
        "\n",
    )


class LogBelowCompressAboveScale(mscale.ScaleBase):
    name = "log_below_10"

    def __init__(self, axis, **kwargs):
        super().__init__(axis)
        self.threshold = 10
        self.compression = 0.2

    def get_transform(self):
        return self.LogBelowCompressAboveTransform(self.threshold, self.compression)

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(LogLocator(base=10.0))
        axis.set_major_formatter(ScalarFormatter())

    def limit_range_for_scale(self, vmin, vmax, minpos):
        return max(vmin, 1e-3), vmax

    class LogBelowCompressAboveTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def __init__(self, threshold, compression):
            super().__init__()
            self.threshold = threshold
            self.compression = compression

        def transform_non_affine(self, y):
            y = np.asarray(y)
            with np.errstate(divide="ignore"):
                return np.where(
                    y <= self.threshold,
                    np.log10(y),
                    np.log10(self.threshold)
                    + np.log10(y / self.threshold) * self.compression,
                )

        def inverted(self):
            return LogBelowCompressAboveScale.InvertedLogBelowCompressAboveTransform(
                self.threshold, self.compression
            )

    class InvertedLogBelowCompressAboveTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def __init__(self, threshold, compression):
            super().__init__()
            self.threshold = threshold
            self.compression = compression

        def transform_non_affine(self, y):
            y = np.asarray(y)
            return np.where(
                y <= np.log10(self.threshold),
                10**y,
                self.threshold
                * 10 ** ((y - np.log10(self.threshold)) / self.compression),
            )

        def inverted(self):
            return LogBelowCompressAboveScale.LogBelowCompressAboveTransform(
                self.threshold, self.compression
            )


# Register the custom scale
mscale.register_scale(LogBelowCompressAboveScale)

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
        stat="density",
        common_norm=False,
    )

    plt.xscale("log")
    plt.yscale("log")

    plot.set(
        ylabel=r"\textsc{Density among Instances}",
        xlabel=x_label,
    )

    handles = [
        Line2D(
            [0], [0], color=sns_palette[0], lw=2, label="Linear"
        ),
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
    r"\textsc{Increase in Removed Nodes to Naive}",
    False,
    "nodes",
)

plot_distr(
    data,
    "su_m",
    r"\textsc{Increase in Removed Edges to Naive}",
    True,
    "edges",
)

plot_distr(
    data,
    "st_tD",
    r"\textsc{Total Increase in Dominating Nodes to No Rule1}",
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

plot = sns.scatterplot(
    extra_data,
    x="iter",
    y="rule_D",
    alpha=0.7,
)

plt.xscale("log")
plt.yscale("log")

plot.set(
    xlabel=r"\textsc{Iteration of Rule1-Extra}",
    ylabel=r"\textsc{Total Increase in $|D|$}",
)

plt.savefig(f"{args.outpath}/extra_iters.pdf", format="pdf", bbox_inches="tight")
