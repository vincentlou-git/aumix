"""
urmp_plot.py

Visualise BSS evaluation results on the stereo URMP dataset, separated using ADRess.

@author: Chan Wai Lou
"""

import pandas as pd
import os
from datetime import datetime
import scipy.stats as stats
import matplotlib.pyplot as pl

from aumix.plot.fig_data import *


#
# Parameters
#
data_folder = "D:\\Year 3\\COMP3931 Individual Project\\repo\examples\\eval\\separation\\bss_metrics"
mean_filename = "mean_urmp.txt"
pm_filename = "pm_urmp.txt"
raw_filename = "raw_urmp.txt"
std_filename = "std_urmp.txt"


#
# Functions
#
def double_std(array):
    return np.std(array) * 2


#
# Computation
#

# Load
mean_df = pd.read_csv(os.path.join(data_folder, mean_filename), sep="\t", header=0)
pm_df = pd.read_csv(os.path.join(data_folder, pm_filename), sep="\t", header=0)
raw_df = pd.read_csv(os.path.join(data_folder, raw_filename), sep="\t", header=0)
std_df = pd.read_csv(os.path.join(data_folder, std_filename), sep="\t", header=0)


# ---Comparison of the performance of ADRess on increasing numbers of mixed instruments---
decline_fig = pl.figure(figsize=(5.5, 2.5), dpi=300)
table = raw_df.groupby("n_sources").agg([np.mean, double_std, stats.sem])

# Common fig values
x = np.arange(2, 6)
linestyles = ["solid", "solid", (0, (10, 5)), "solid", (5, (10, 5))]
alphas = [1, 1, 0.5, 1, 0.5]
colors = ["tab:purple", "tab:red", "tab:orange", "tab:blue", "green"]

# Get all metric dataframes
headers = mean_df.columns.values[:-2]
for i, header in enumerate(headers):
    metric_df = table[header]

    # Plot its errorbar
    pl.errorbar(x, table[header, "mean"].to_numpy(),
                yerr=table[header, "sem"].to_numpy(),
                label=header, linestyle=linestyles[i], alpha=alphas[i], color=colors[i])

pl.title("Evaluation of ADRess on the URMP Dataset")
pl.xticks(x)
pl.xlabel("Number of sources")
pl.ylabel("Score")
pl.grid(alpha=0.5)
pl.legend(bbox_to_anchor=(1, 1), loc="upper left")

timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
pl.savefig(f"figures/{timestamp}-ADRess-eval-urmp-decline", bbox_inches='tight')
pl.show()


# ---box-plot for each piece---
# Common fig values
x = np.arange(1, 45)
box_colors = ["w", "w", "firebrick", "olive", "seagreen", "steelblue", "indigo"]

# For each metric, create a boxplot of all pieces
headers = mean_df.columns.values[:-2]
for header in headers:
    data = [raw_df.query(f"idx == {i}")[header].to_numpy() for i in x]
    min_data = np.min([np.min(row) for row in data])
    max_data = np.max([np.max(row) for row in data])
    ns = [raw_df.query(f"idx == {i}")["n_sources"].iloc[0] for i in x] + [6]

    # Plot
    box_fig = pl.figure(figsize=(10.5, 4), dpi=300)
    for n in range(2, 6):
        pl.boxplot(data[ns.index(n):ns.index(n+1)],
                   positions=range(ns.index(n)+1, ns.index(n+1)+1),
                   flierprops=dict(markerfacecolor='r', marker='d'),
                   boxprops=dict(color=box_colors[n]),
                   whiskerprops=dict(color=box_colors[n]),
                   capprops=dict(color=box_colors[n]))

    pl.title(header)
    pl.xticks(x)
    pl.xlabel("Piece Number")
    pl.ylabel("Score")
    pl.grid(alpha=0.5, axis="y")

    ax = pl.gca()
    ax.set_ylim((min_data - (max_data - min_data) * 0.125, None))
    legend = ax.legend((2, 3, 4, 5), title="# Sources",
                       loc="lower left")
    for i, n in enumerate(range(2, 6)):
        legend.legendHandles[i].set_color(box_colors[n])

    timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
    pl.savefig(f"figures/{timestamp}-ADRess-eval-urmp-{header}", bbox_inches='tight')
    pl.show()
