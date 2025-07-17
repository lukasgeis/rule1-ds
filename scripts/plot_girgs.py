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
    "su_gt": [],
    # Speedups to Singletons
    "st_D": [],
    "st_t": [],
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
        data["su_gt"].append(naive["greedy_time"] / rule["greedy_time"])

        data["st_D"].append(empty["greedy_domset"] / rule["greedy_domset"])
        data["st_t"].append(empty["greedy_time"] / rule["greedy_time"])


data = pd.DataFrame.from_dict(data)

data["frac_n"] = data["rule_n"] / data["n"]
data["frac_m"] = data["rule_m"] / data["m"]
data["frac_D"] = data["rule_D"] / data["n"]
data["frac_covered"] = data["rule_covered"] / data["n"]

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


def plot_node_distr(data, y, y_label, y_scale, name):
    plt.clf()

    sns.set_theme(style="whitegrid")
    sns.set_palette("colorblind")
    plt.rcParams["text.usetex"] = True
    plt.rcParams["figure.figsize"] = 7, 4

    plot = sns.scatterplot(
        data[(data.rule_name != "Naive") & (data.rule_name != "Extra")],
        x="m",
        y=y,
        hue="rule_name",
        hue_order=["Linear", "Plus"],
        palette="colorblind",
        style="rule_name",
        markers=["o", "v"],
    )

    plt.xscale("log")
    plt.yscale(y_scale)

    plot.set(
        xlabel=r"\textsc{Input: Number of Edges}",
        ylabel=y_label,
    )

    handles, labels = plot.get_legend_handles_labels()
    labels = [r"\textsc{Linear}", r"\textsc{Plus}"]

    plt.legend(handles[::-1], labels[::-1], title=r"\textsc{Rule1}")

    plt.savefig(f"{args.outpath}/{name}.pdf", format="pdf", bbox_inches="tight")


plot_node_distr(
    data,
    "su_n",
    r"\textsc{Increase in Removed Nodes to Naive}",
    "linear",
    "nodes",
)
plot_node_distr(
    data,
    "su_D",
    r"\textsc{Increase in Dominating Nodes to Naive}",
    "linear",
    "domset",
)
plot_node_distr(
    data,
    "su_covered",
    r"\textsc{Increase in Covered Nodes to Naive}",
    "linear",
    "covered",
)


plt.clf()

sns.set_theme(style="whitegrid")
sns.set_palette("colorblind")
plt.rcParams["text.usetex"] = True
plt.rcParams["figure.figsize"] = 7, 4

plot = sns.scatterplot(
    data[data.rule_name != "Naive"],
    x="m",
    y="su_m",
    hue="rule_name",
    hue_order=["Linear", "Plus", "Extra"],
    palette="colorblind",
    style="rule_name",
    markers=["o", "v", "X"],
)

plt.xscale("log")
plt.yscale("linear")

plot.set(
    xlabel=r"\textsc{Input: Number of Edges}",
    ylabel=r"\textsc{Increase in Removed Edges to Naive}",
)

handles, labels = plot.get_legend_handles_labels()
labels = [r"\textsc{Linear}", r"\textsc{Plus}", r"\textsc{Extra}"]

plt.legend(handles[::-1], labels[::-1], title=r"\textsc{Rule1}")

plt.savefig(f"{args.outpath}/edges.pdf", format="pdf", bbox_inches="tight")


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
    palette="colorblind",
    style="rule_name",
    markers=["o", "v", "X"],
)

plt.xscale("log")
plt.yscale("linear")

plot.set(
    xlabel=r"\textsc{Input: Number of Edges}",
    ylabel=r"\textsc{Speedup to Naive}",
)

handles, labels = plot.get_legend_handles_labels()
labels = [r"\textsc{Linear}", r"\textsc{Plus}", r"\textsc{Extra}"]

plt.legend(handles[::-1], labels[::-1], title=r"\textsc{Rule1}")

plt.savefig(f"{args.outpath}/time.pdf", format="pdf", bbox_inches="tight")
