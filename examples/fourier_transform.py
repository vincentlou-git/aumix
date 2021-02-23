# -*- coding: utf-8 -*-
"""
fourier_transform.py

Experiments in using Fourier Transform.

@author: Chan Wai Lou / Vincent Lou
"""

import aumix.signal.fourier_series as fs
import aumix.signal.simple_signal as ss
import aumix.signal.stationary_signal as sts
import aumix.plot.plot as aplot
from aumix.plot.fig_data import *

import numpy as np
from scipy.fft import fft, ifft, fftfreq


#
# Parameters
#
duration = 0.75
samp_rate = 800

freq = 440
cl_pitch_label = "?"

signal_params = {
    "duration": duration,
    "samp_rate": samp_rate,
    # "freq": freq,
    "options": {
        "normalize": True
    }
}

#
# Generate signals
#

# Generate clarinet signal using predefined amplitudes
cl_signal = sts.ClarinetApproxSignal(**signal_params)

# Generate simple signals
sine_signal1 = ss.SineSignal(**signal_params, freq=50)
sine_signal2 = ss.SineSignal(**signal_params, freq=80)
sine_sum_signal = sine_signal1.data + sine_signal2.data * 0.5


# Generate clarinet Fourier Transform
cl_fft = fft(cl_signal.data)
cl_recon = ifft(cl_fft)

cl_slice_num = cl_signal.samp_nums.shape[0]//2
cl_fft_x = fftfreq(cl_signal.samp_nums.shape[0], 1/cl_signal.samp_rate)

# Generate sine sum Fourier Transform
sine_fft = fft(sine_sum_signal.data)
sine_recon = ifft(sine_fft)

sine_slice_num = sine_signal1.samp_nums.shape[0]//2
sine_fft_x = fftfreq(sine_signal1.samp_nums.shape[0], 1/sine_signal1.samp_rate)


#
# Figure options
#

cl_signal_title = "Approximated Clarinet Signal"
cl_fft_title = "Fourier Transform " + cl_signal_title

sine_sum_signal_title = f"Sum of two sine signals at frequencies {sine_signal1.freq}Hz and {sine_signal2.freq}Hz"
sine_sum_fft_title = "Fourier Transform of the " + sine_sum_signal_title

general_options = {"xlabel": "time (seconds)",
                   "ylabel": "Amplitude"}

# line_options = [{"label": f"f0 = {freq}Hz"}]

#
# Encapsulate signals data
#
cl_signal_fig = FigData(xs=cl_signal.samp_nums,
                        ys=[cl_signal.data],
                        title=cl_signal_title,
                        **general_options)

cl_fft_fig = FigData(xs=cl_fft_x[:cl_slice_num],
                     ys=np.abs(cl_fft[:cl_slice_num]),
                     title=cl_fft_title,
                     options=["grid"],
                     **general_options)

sine_signal_fig = FigData(xs=sine_signal1.samp_nums,
                          ys=[sine_sum_signal.data],
                          title=sine_sum_signal_title,
                          **general_options)

sine_fft_fig = FigData(xs=sine_fft_x[:sine_slice_num],
                       ys=np.abs(sine_fft[:sine_slice_num]),
                       title=sine_sum_fft_title,
                       options=["grid"],
                       **general_options)

#
# Plot
#

aplot.single_subplots({(2, 1, 1): cl_signal_fig,
                       (2, 1, 2): cl_fft_fig},
                      individual_figsize=(12, 6))

aplot.single_subplots({(2, 1, 1): sine_signal_fig,
                       (2, 1, 2): sine_fft_fig},
                      individual_figsize=(12, 6))
