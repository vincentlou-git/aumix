# -*- coding: utf-8 -*-
"""
periodic_composite_signal_plot.py

@author: Chan Wai Lou / Vincent Lou
"""

import aumix.signal.stationary_signal as pcs
import aumix.plot.plot as aplot
from aumix.plot.fig_data import *

import numpy as np

# parameters
duration = 0.01
samp_rate = 44100

cl_f = 466.16 / 2  # written pitch C4, actual pitch is Bb4
cl_pitch_label = "C4"

signal_params = {
    "duration": duration,
    "samp_rate": samp_rate,
    "freq": cl_f,
    "options": {
        "normalize": True
    }
}

# Generate clarinet signal using predefined amplitudes
cl_signal = pcs.ClarinetApproxSignal(**signal_params)

# Figure options
cl_fig = FigData(xs=cl_signal.samp_nums,
                 ys=[cl_signal.data],
                 line_options=[{"label": f"f0 = {cl_f}Hz"}],
                 xlabel="Time (seconds",
                 ylabel="Amplitude",
                 title=f"Synthesized clarinet (written pitch {cl_pitch_label})",
                 figsize=(5, 3)
                 )

# Plot signals
aplot.single_plot(cl_fig)
