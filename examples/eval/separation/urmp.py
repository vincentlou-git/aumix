"""
urmp.py

Evaluation of ADRess separation on the synthetic melodic progression
using standard metrics implemented in `mir_eval` and the scale-related SDR variants.

@author: Chan Wai Lou
"""

import re
import os
import librosa
import pandas as pd
from scipy import stats

from aumix.plot.fig_data import *
import aumix.analysis.eval as aeval


#
# Parameters
#
truth_root = "../../../data/URMP/Dataset/"
est_root = "audio"

samp_rate = 48000


#
# Functions
#
def comp_stats(table):
    arr = table.iloc[:, :-2].to_numpy()

    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)
    pm = (np.max(arr, axis=0) - np.min(arr, axis=0)) / 2

    # TODO: t-test
    # stats.ttest_rel()
    return arr, mean, std, pm


#
# Load and Evaluate
#
results_df = pd.DataFrame()

true_pattern = re.compile("AuSep(.*?).wav")   # Pattern for truth file names
est_pattern = re.compile("(.*?).wav")   # Pattern for separated file names

# Find all separated file names
est_root_abs = os.path.abspath(est_root)
est_filenames = [f for f in os.listdir(est_root_abs) if est_pattern.match(f)]

# Put the est file names in a dictionary and sort them in lists
est_file_dict = dict()
for i in range(1, 45):
    est_part_pattern = re.compile(f"{i}-d=(.*?).wav")
    matches = [f for f in est_filenames if est_part_pattern.match(f)]

    # Sort according to position
    matches.sort(key=lambda f: int(f.split("d=")[1].split("-")[0]))
    est_file_dict[i] = matches

for dir_name, subdir_list, files in os.walk(truth_root):
    # Inside directory: dir_name

    # Find all the parts
    part_names = list()
    for filename in files:
        if true_pattern.match(filename) is not None:
            part_names.append(filename)

    # Skip the folder if there are less than 2 parts
    n = len(part_names)
    if n <= 1:
        continue

    # id is the first 2 characters of the dir name
    idx = int(dir_name.split("/")[-1][:2])

    if idx != 35:
        continue

    # Load the true parts
    true_signals = np.array([librosa.load(f"{dir_name}/{part}", sr=samp_rate, mono=True)[0] for part in part_names])

    # Load the reconstructed parts
    est_signals = np.array([librosa.load(f"{est_root_abs}/{part}", sr=samp_rate, mono=True)[0] for part in est_file_dict[idx]])

    # Evaluate
    df = aeval.bss_eval_df(true_signals,
                           est_signals[:, :true_signals.shape[1]],
                           compute_permutation=False)
    # add index and number of instruments
    df = df.assign(idx=idx, n_sources=n)

    # Concatenate results to the existing table
    results_df = pd.concat((results_df, df))


# Calculate statistics
columns = ["SI-SDR", "SD-SDR", "SDR", "SIR", "SAR"]
ids = np.concatenate((np.array(["All"]), np.arange(1, 45), np.array(["-"] * 4)))
n_sources = np.array(["All"] + [results_df.query(f"idx == {i}").n_sources.iloc[0] for i in range(1, 45)] + [2, 3, 4, 5])

mean = np.empty((len(ids), 5))
std = np.empty((len(ids), 5))
pm = np.empty((len(ids), 5))
_, mean[0], std[0], pm[0] = comp_stats(results_df)   # overall

# Stats across each piece
for i in range(1, 45):
    result = results_df.query(f"idx == {i}")

    if result.empty:
        continue

    _, mean[i], std[i], pm[i] = comp_stats(results_df)

# Stats across each number of sources
for i in range(4):
    result = results_df.query(f"n_sources == {i+2}")

    if result.empty:
        continue

    idx = len(ids) - 4 + i
    _, mean[idx], std[idx], pm[idx] = comp_stats(results_df)

# Concatenate mean and plus minus
mean_pm = np.core.defchararray.add(np.core.defchararray.add(mean.astype(str), " $\\pm$ "), pm.astype(str))

# Put all statistical results in a table
mean_pm_df = pd.DataFrame(
    data=mean_pm,
    columns=columns
)
std_df = pd.DataFrame(
    data=std,
    columns=columns
)
mean_pm_df = mean_pm_df.assign(idx=ids, n_sources=n_sources)
std_df = std_df.assign(idx=ids, n_sources=n_sources)

# Reorder raw eval results
results_df = results_df[["idx", "n_sources", "SI-SDR", "SD-SDR", "SDR", "SIR", "SAR", "Perm"]]

#
# Write file
#
if not os.path.exists("bss_metrics"):
    os.mkdir("bss_metrics")

mean_pm_df.to_csv("bss_metrics/mean_pm_urmp.txt", header=True, index=False, sep=" ", mode="w")
std_df.to_csv("bss_metrics/std_urmp.txt", header=True, index=False, sep=" ", mode="w")
results_df.to_csv("bss_metrics/raw_urmp.txt", header=True, index=False, sep=" ", mode="w")
