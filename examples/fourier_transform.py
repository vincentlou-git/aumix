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
duration = 1

cl_freq = 440
cl_pitch_label = "?"
cl_samp_rate = 44100

sine_samp_rate = 800

signal_params = {
    "duration": duration,
    "options": {
        "normalize": True
    }
}

#
# Generate signals
#

# Clarinet signal using predefined amplitudes
cl_signal = sts.ClarinetApproxSignal(freq=cl_freq, samp_rate=cl_samp_rate, **signal_params)

# Superposition of two sines
sine_signal = sts.StationarySignal(sin_freqs=[25, 80], sin_coeffs=[1, 0.5], samp_rate=sine_samp_rate, **signal_params)


# FFT of clarinet
cl_fft = fft(cl_signal.data)
cl_recon = ifft(cl_fft)

# Only consider positive frequencies (since the spectrum is symmetric)
cl_slice_num = cl_signal.samp_nums.shape[0]//2
# Nyquist: max discernable freq is sampling rate / 2
cl_fft_x = fftfreq(cl_signal.samp_nums.shape[0], 1/cl_signal.samp_rate)

# FFT of sine sum
sine_fft = fft(sine_signal.data)
sine_recon = ifft(sine_fft)

sine_slice_num = sine_signal.samp_nums.shape[0]//2
sine_fft_x = fftfreq(sine_signal.samp_nums.shape[0], 1/sine_signal.samp_rate)


#
# Figure options
#

# Zoom
cl_time_for_two_cycles = min(duration, 1/cl_signal.freq*2)
cl_xlim = (-0.05*cl_time_for_two_cycles, 1.05*cl_time_for_two_cycles)

sine_time_for_two_cycles = min(duration, 1/min(sine_signal.sin_freqs)*2)
sine_xlim = (-0.05*sine_time_for_two_cycles, 1.05*sine_time_for_two_cycles)

# Axis labels
signal_axis_labels = {"xlabel": "time (seconds)",
                      "ylabel": "Amplitude"}
fft_axis_labels = {"xlabel": "Frequency (Hz)",
                   "ylabel": "Amplitude per Hz"}

# Titles
cl_signal_title = f"Synthetic Clarinet Signal. Sampling Rate = {cl_signal.samp_rate}Hz"
cl_raw_title = "(a) Constituent waves of the " + cl_signal_title
cl_fft_title = "(c) Fourier Transform of the " + cl_signal_title.split(".")[0]

sine_raw_title = f"Pure Sine Waves. Sampling Rate = {sine_signal.samp_rate}Hz"
sine_sum_signal_title = f"Sum of two sine signals at frequencies { sine_signal.sin_freqs[0] }Hz and { sine_signal.sin_freqs[1] }Hz"
sine_sum_fft_title = "Fourier Transform of the " + sine_sum_signal_title

# Line options
cl_raw_line_options = [{"label": f"(Sine) f0 = {cl_freq}Hz"}]
cl_raw_line_options += [{"label": f"(Sine) {amp:.2f} * { ((i+1)*2 + 1) * cl_freq }Hz"} for i, amp in enumerate(cl_signal.sin_coeffs[2::2])]

sine_raw_line_options = [{"label": f"{amp:.2f} * { sine_signal.sin_freqs[i] }Hz"} for i, amp in enumerate(sine_signal.sin_coeffs)]

#
# Encapsulate signals data
#
cl_raw_fig = FigData(xs=cl_signal.samp_nums,
                     ys=cl_signal._sine_components()[::2],
                     title=cl_raw_title,
                     line_options=cl_raw_line_options,
                     xlim=cl_xlim,
                     **signal_axis_labels)

cl_signal_fig = FigData(xs=cl_signal.samp_nums,
                        ys=[cl_signal.data,
                            cl_recon],
                        title=f"(b) {cl_signal_title}",
                        line_options=[{"label": "Original",
                                       "linestyle": "solid"},
                                      {"label": "Reconstructed from FFT",
                                       "linestyle": "dashed"}],
                        xlim=cl_xlim,
                        **signal_axis_labels)

cl_fft_fig = FigData(xs=cl_fft_x[:cl_slice_num],
                     ys=np.abs(cl_fft[:cl_slice_num]),
                     title=cl_fft_title,
                     options=["grid"],
                     **fft_axis_labels)


sine_raw_fig = FigData(xs=sine_signal.samp_nums,
                       ys=sine_signal._sine_components(),
                       title=sine_raw_title,
                       line_options=sine_raw_line_options,
                       xlim=sine_xlim,
                       **signal_axis_labels)

sine_signal_fig = FigData(xs=sine_signal.samp_nums,
                          ys=[sine_signal.data,
                              sine_recon],
                          title=sine_sum_signal_title,
                          line_options=[{"label": "Original"},
                                        {"label": "Reconstructed from FFT"}],
                          xlim=sine_xlim,
                          **signal_axis_labels)

sine_fft_fig = FigData(xs=sine_fft_x[:sine_slice_num],
                       ys=np.abs(sine_fft[:sine_slice_num]),
                       title=sine_sum_fft_title,
                       options=["grid"],
                       **fft_axis_labels)

#
# Plot
#

aplot.single_subplots(grid_size=(3, 1),
                      fig_data={(0, 0): cl_raw_fig,
                                (1, 0): cl_signal_fig,
                                (2, 0): cl_fft_fig},
                      individual_figsize=(9, 3),
                      savefig_path="Clarinet_FFT"
                      )

aplot.single_subplots(grid_size=(3, 1),
                      fig_data={(0, 0): sine_raw_fig,
                                (1, 0): sine_signal_fig,
                                (2, 0): sine_fft_fig},
                      individual_figsize=(12, 3),
                      savefig_path=f"{str(sine_signal.sin_freqs).replace(' ', '')}Hz_Sine_Sum_FFT"
                      )
