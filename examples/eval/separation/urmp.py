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
import aumix.io.filename as af
import aumix.analysis.eval as aeval


#
# Parameters
#
truth_root = "../../../data/URMP/Dataset/"
est_root = "adress_audio"

samp_rate = 48000

ids_to_test = range(1, 45)   # Which pieces to test for?
ns = [2, 3, 4, 5]   # Possible numbers of sources


#
# Load and Evaluate
#
if not os.path.exists("bss_metrics"):
    os.mkdir("bss_metrics")

results_df = pd.DataFrame()


# Put the xml file names in a dictionary and sort them in lists
est_root_abs = os.path.abspath(est_root)
est_file_dict = af.id_filename_dict(est_root_abs,
                                    ids=ids_to_test,
                                    regexes=[f"{i}-d=(.*?).wav" for i in ids_to_test],
                                    sort_function=lambda f: int(f.split("d=")[1].split("-")[0]))

true_pattern = re.compile("AuSep(.*?).wav")   # Pattern for truth file names
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

    if idx not in ids_to_test:
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

    # Write it in case things crash
    df.to_csv("bss_metrics/raw_urmp.txt", header=False, index=False, sep="\t", mode="a")

# Calculate statistics
columns = ["SI-SDR", "SD-SDR", "SDR", "SIR", "SAR"]
ids = np.concatenate((np.array(["All"]), ids_to_test, np.array(["-"] * len(ns))))
n_sources = np.array(["All"] + [results_df.query(f"idx == {i}").n_sources.iloc[0] for i in ids_to_test] + ns)

mean = np.empty((len(ids), 5))
std = np.empty((len(ids), 5))
pm = np.empty((len(ids), 5))
_, mean[0], std[0], pm[0] = aeval.comp_stats(results_df)   # overall

# Stats across each piece
for i, idx in enumerate(ids_to_test):
    result = results_df.query(f"idx == {idx}")

    if result.empty:
        continue

    _, mean[i+1], std[i+1], pm[i+1] = aeval.comp_stats(result)

# Stats across each number of sources
for i, n in enumerate(ns):
    result = results_df.query(f"n_sources == {n}")

    if result.empty:
        continue

    idx = len(ids) - len(ns) + i
    _, mean[idx], std[idx], pm[idx] = aeval.comp_stats(result)

# Concatenate mean and plus minus
mean_pm = np.core.defchararray.add(np.core.defchararray.add(mean.astype(str), " $\\pm$ "), pm.astype(str))

# Put all statistical results in a table
mean_pm_df = pd.DataFrame(data=mean_pm, columns=columns)
mean_df = pd.DataFrame(data=mean, columns=columns)
pm_df = pd.DataFrame(data=pm, columns=columns)
std_df = pd.DataFrame(data=std, columns=columns)

mean_pm_df = mean_pm_df.assign(idx=ids, n_sources=n_sources)
mean_df = mean_df.assign(idx=ids, n_sources=n_sources)
pm_df = pm_df.assign(idx=ids, n_sources=n_sources)
std_df = std_df.assign(idx=ids, n_sources=n_sources)

#
# Write files
#

mean_pm_df.to_csv("bss_metrics/mean_pm_urmp.txt", header=True, index=False, sep="\t", mode="w")
mean_df.to_csv("bss_metrics/mean_urmp.txt", header=True, index=False, sep="\t", mode="w")
pm_df.to_csv("bss_metrics/pm_urmp.txt", header=True, index=False, sep="\t", mode="w")
std_df.to_csv("bss_metrics/std_urmp.txt", header=True, index=False, sep="\t", mode="w")
