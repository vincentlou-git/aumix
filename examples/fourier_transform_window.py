# -*- coding: utf-8 -*-
"""
fourier_transform_window.py

Experiments in the effect of the Fourier Transform of the convolution of different window functions
and the original signal. The idea of window functions are to redistribute the "spectral leakage" effect,
which can be beneficial depending on the application.

@author: Chan Wai Lou / Vincent Lou
"""

import aumix.signal.stationary_signal as sts
import aumix.plot.plot as aplot
from aumix.plot.fig_data import *

import numpy as np
from scipy.fft import fft, ifft, fftfreq
import scipy.signal.windows as windows

#
# Parameters
#
duration = 0.1
samp_rate = 44100

window = windows.blackmanharris
window_params = {}

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
w = window(cl_signal.samp_nums.shape[0], **window_params)
cl_window = cl_signal.data * w


# Generate clarinet Fourier Transform
cl_fft = fft(cl_signal.data)
cl_window_fft = fft(cl_window)

# Reconstructed signals from FFT
cl_ifft = ifft(cl_fft)
cl_window_ifft = ifft(cl_window_fft)

# Only consider positive frequencies (since the spectrum is symmetric)
cl_slice_num = cl_signal.samp_nums.shape[0]//2
# Nyquist: max discernible freq is sampling rate / 2
cl_fft_x = fftfreq(cl_signal.samp_nums.shape[0], 1/cl_signal.samp_rate)


#
# Figure options
#

# Zoom
cl_time_for_two_cycles = min(duration, 1/cl_signal.freq*2)
cl_xlim = (-0.05*cl_time_for_two_cycles, 1.05*cl_time_for_two_cycles)

cl_signal_title = "Approximated Clarinet Signal"
cl_window_title = cl_signal_title + f" * {window.__name__}"
cl_fft_title = "Fourier Transform"

signal_axis_labels = {"xlabel": "time (seconds)",
                      "ylabel": "Amplitude"}
fft_axis_labels = {"xlabel": "Frequency (Hz)",
                   "ylabel": "Amplitude"}

#
# Encapsulate signals data
#

cl_signal_fig = FigData(xs=cl_signal.samp_nums,
                        ys=[cl_signal.data,
                            cl_ifft],
                        title=cl_signal_title,
                        line_options=[{"label": "Original"},
                                      {"label": "Reconstructed from FFT"}],
                        # xlim=cl_xlim,
                        **signal_axis_labels)

cl_window_fig = FigData(xs=cl_signal.samp_nums,
                        ys=[cl_window,
                            cl_window_ifft,
                            w],
                        title=cl_window_title,
                        line_options=[{"label": "Original"},
                                      {"label": "Reconstructed from FFT"},
                                      {"label": window.__name__}],
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

#
# Plot
#

aplot.single_subplots(grid_size=(2, 2),
                      fig_data={(0, 0): cl_signal_fig,
                                (0, 1): cl_window_fig,
                                (1, 0, 1, 2): cl_fft_fig},
                      individual_figsize=(6, 4),
                      savefig_path=f"Clarinet_FFT_Full_{window.__name__}")
