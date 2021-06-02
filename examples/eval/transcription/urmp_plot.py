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
import seaborn as sns

from aumix.plot.fig_data import *


#
# Parameters
#
data_folder = "D:\\Year 3\\COMP3931 Individual Project\\repo\examples\\eval\\transcription"



#
# Functions
#
def double_std(array):
    return np.std(array) * 2


#
# Computation
#
headers = ["idx", "Comparison", "Precision", "Recall", "fmeasure", "overlap"]

# Load
raw_df = pd.read_csv(os.path.join(data_folder, "raw.txt"), sep="\t", header=None, names=headers)


# ---bar-plot for each piece---
sns.set_theme(style="whitegrid")
sns.set_palette("tab10")

for title, col in [("F-score", "fmeasure"), ("Average Overlap Ratio", "overlap")]:
    ax = sns.catplot(x="idx", y=col, hue="Comparison", data=raw_df, kind="bar",
                     height=4, aspect=2.1)
    pl.title(title)
    pl.xlabel("Piece Number")
    pl.ylabel("Score")

    timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
    pl.savefig(f"figures/{timestamp}-ADRess-eval-urmp-{col}", bbox_inches='tight', dpi=300)
    pl.show()
