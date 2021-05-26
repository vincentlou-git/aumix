# -*- coding: utf-8 -*-
"""
spectral_leakage.py

Demonstration of spectral leakage.

@author: Chan Wai Lou / Vincent Lou
"""

import aumix.signal.fourier_series as fs
import aumix.signal.simple_signal as ss
import aumix.signal.stationary_signal as sts
import aumix.plot.plot as aplot
from aumix.plot.fig_data import *

import numpy as np
from scipy.fft import fft, fftfreq, fftshift
import scipy.signal as signal

#
# Parameters
#
durations = [3, 3.01]

freq = 2
samp_rate = 10


#
# Functions
#
def positive_response(sig, normalize=False):
    ft = fft(sig)
    res = np.abs(ft) / (np.max(np.abs(ft)) if normalize else 1)
    return res


#
# Generate signals
#
signals = [ss.SineSignal(freq=freq, samp_rate=samp_rate, duration=d) for d in durations]
true_signal = ss.SineSignal(freq=freq, samp_rate=1000, duration=10)

rect = signal.get_window("rect", signals[1].samp_nums.shape[0])
blackman = signal.get_window("blackman", signals[1].samp_nums.shape[0])
ws = signals[1].data * blackman

# FFT
freqs = [fftfreq(s.samp_nums.shape[0], 1 / samp_rate) for s in signals]
psns = [np.argmax(freqs[i]) for i in range(len(signals))]  # positive slice numbers
max_response_freq = np.max(fftfreq(2048, 1 / samp_rate))

responses = [positive_response(s.data) for s in signals]
zp_responses = [positive_response(np.concatenate((s.data, np.zeros(2048 - len(s.data))))) for s in signals]

ws_response = positive_response(signals[1].data * blackman)
wszp_response = positive_response(np.concatenate((signals[1].data * blackman, np.zeros(2048 - len(signals[1].data)))))

#
# Figure options
#

# Axis labels
signal_axis_labels = {"xlabel": "time (seconds)",
                      "ylabel": "Amplitude"}
fft_axis_labels = {"xlabel": "Frequency (Hz)",
                   "ylabel": "Amplitude of FFT"}
fft_line_options = [{"label": "DTFT",
                     "color": "tab:orange"},
                    {"label": "DFT",
                     "color": "tab:blue",
                     "marker": "x"}]

#
# Encapsulate signals data
#
fig_labels = [chr(ord('a') + i) for i in range(6)]
sig_figs = [FigData(xs=signals[1].samp_nums,
                    ys=[s.data, rect],
                    title=f"({fig_labels[i * 2]}) {freq * s.duration} cycles",
                    line_options=[{"label": "Sinusoid Input"},
                                  {"label": "Rectangular window"}],
                    legend_options={"loc": 3},
                    **signal_axis_labels)
            for i, s in enumerate(signals)]
ws_fig = FigData(xs=signals[1].samp_nums,
                 ys=[ws, blackman],
                 title=f"(e) {freq * signals[1].duration} cycles",
                 line_options=[{"label": "Sinusoid Input"},
                               {"label": "Hann window"}],
                 legend_options={"loc": 3},
                 **signal_axis_labels)

fft_figs = [FigData(xs=[np.linspace(0, max_response_freq, 1024), x[:psns[i]+1]],
                    ys=[zp_responses[i][:1024], responses[i][:psns[i]+1]],
                    title=f"({fig_labels[i * 2 + 1]}) Magnitude of DFT ({freq * signals[i].duration} cycles)",
                    options=["grid", "xres"],
                    line_options=fft_line_options,
                    **fft_axis_labels)
            for i, x in enumerate(freqs)]
ws_fft_fig = FigData(xs=[np.linspace(0, max_response_freq, 1024), freqs[1][:psns[1]+1]],
                     ys=[wszp_response[:1024], ws_response[:psns[1]+1]],
                     title=f"(f) Magnitude of DFT (Windowed, {freq * signals[1].duration} cycles)",
                     options=["grid", "xres"],
                     line_options=fft_line_options,
                     **fft_axis_labels)

#
# Plot
#
aplot.single_subplots(grid_size=(3, 2),
                      fig_data={(0, 0): sig_figs[0],
                                (0, 1): fft_figs[0],
                                (1, 0): sig_figs[1],
                                (1, 1): fft_figs[1],
                                (2, 0): ws_fig,
                                (2, 1): ws_fft_fig},
                      individual_figsize=(5.5, 3),
                      title=f"{freq}Hz Sinusoid sampled at {samp_rate}Hz",
                      savefig_path="Spectral_Leakage"
                      )

