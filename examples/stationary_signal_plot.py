# -*- coding: utf-8 -*-
"""
periodic_composite_signal_plot.py

@author: Chan Wai Lou / Vincent Lou
"""

import aumix.signal.stationary_signal as pcs
import aumix.plot.plot as aplot

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
general_options = {"xlabel": "time (seconds)",
                   "ylabel": "Amplitude"}

cl_line_options = [{"label": f"f0 = {cl_f}Hz"}]

# Plot signals
aplot.single_plot(cl_signal.samp_nums,
                  [cl_signal.data],
                  line_options=cl_line_options,
                  title=f"Synthesized clarinet (written pitch {cl_pitch_label})",
                  **general_options)
