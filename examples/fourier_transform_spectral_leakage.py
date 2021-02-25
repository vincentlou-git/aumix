# -*- coding: utf-8 -*-
"""
fourier_transform_spectral_leakage.py

Experiments in the spectral leakage effect in Fourier Transform,
and how to mitigate this undesired effect.

@author: Chan Wai Lou / Vincent Lou
"""

import aumix.signal.fourier_series as fs
import aumix.signal.simple_signal as ss
import aumix.signal.stationary_signal as sts
import aumix.plot.plot as aplot
from aumix.plot.fig_data import *

import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import blackman

#
# Parameters
#
duration = 1
samp_rate = 44100

cl_freq = 100
cl_pitch_label = "?"

signal_params = {
    "duration": duration,
    "samp_rate": samp_rate,
    # "freq": cl_freq,
    "options": {
        "normalize": True
    }
}

#
# Generate signals
#

# Generate clarinet signal using predefined amplitudes
cl_signal = sts.ClarinetApproxSignal(freq=cl_freq, **signal_params)

# Windowing the clarinet signal
w = blackman(44100)
cl_window = cl_signal.data * w


# Generate clarinet Fourier Transform
cl_fft = fft(cl_signal.data)
cl_window_fft = fft(cl_window)

# Only consider positive frequencies (since the spectrum is symmetric)
cl_slice_num = cl_signal.samp_nums.shape[0]//2
# Nyquist: max discernable freq is sampling rate / 2
cl_fft_x = fftfreq(cl_signal.samp_nums.shape[0], 1/cl_signal.samp_rate)


#
# Figure options
#

cl_signal_title = "Approximated Clarinet Signal"
cl_window_title = cl_signal_title + " w/ Blackman Window"
cl_fft_title = "Fourier Transform"

signal_axis_labels = {"xlabel": "time (seconds)",
                      "ylabel": "Amplitude"}
fft_axis_labels = {"xlabel": "Frequency (Hz)",
                   "ylabel": "Amplitude"}

#
# Encapsulate signals data
#

cl_signal_fig = FigData(xs=cl_signal.samp_nums,
                        ys=[cl_signal.data],
                        title=cl_signal_title,
                        **signal_axis_labels)

cl_window_fig = FigData(xs=cl_signal.samp_nums,
                        ys=cl_window,
                        title=cl_window_title,
                        **signal_axis_labels)

cl_fft_fig = FigData(xs=cl_fft_x[:cl_slice_num],
                     ys=[np.abs(cl_fft[:cl_slice_num]),
                         np.abs(cl_window_fft[:cl_slice_num])],
                     title=cl_fft_title,
                     line_options=[{"label": cl_signal_title},
                                   {"label": cl_window_title}],
                     options=["grid"],
                     plot_type="semilogy",
                     **fft_axis_labels)

# cl_window_fft_fig = FigData(xs=cl_fft_x[:cl_slice_num],
#                             ys=np.abs(cl_window_fft[:cl_slice_num]),
#                             title=cl_window_fft_title,
#                             options=["grid"],
#                             **fft_axis_labels)

#
# Plot
#

aplot.single_subplots(grid_size=(2, 2),
                      fig_data={(0, 0): cl_signal_fig,
                                (0, 1): cl_window_fig,
                                (1, 0, 1, 2): cl_fft_fig},
                      individual_figsize=(6, 4))