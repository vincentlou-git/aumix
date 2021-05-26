# -*- coding: utf-8 -*-
"""
fourier_series_plot.py

Plots signals approximated with fourier series.

@author: Chan Wai Lou / Vincent Lou
"""

import aumix.signal.fourier_series as fs
import aumix.signal.simple_signal as ss
import aumix.plot.plot as aplot
from aumix.plot.fig_data import *

import numpy as np


    
# parameters
duration = 4
samp_rate = 1000
freq = 1
signal_params = {
    "duration": duration,
    "samp_rate": samp_rate,
    "freq": freq
}

Ns = [1, 3, 5, 15, 101]
ts = np.arange(duration * samp_rate) / samp_rate



# Generate sawtooth signal
sawtooths = [
    fs.FourierSawtoothSignal(n=n, 
                             **signal_params).data 
    for n in Ns]
sawtooths.append( ss.SawtoothSignal(**signal_params).data )

# Generate square signal
squares = [
    fs.FourierSquareSignal(n=n, 
                           **signal_params).data 
    for n in Ns]
squares.append( ss.SquareSignal(**signal_params).data )


# Figure options
general_options = {"xlabel": "time (seconds)",
                   "ylabel": "Amplitude"}

sawtooth_line_options = [{"label": f"N = {n}"} for n in Ns]
sawtooth_line_options.append({"label": "True signal",
                              "linewidth": 2.5,
                              "color": "black"})
sq_line_options = [{"label": f"N = {n}"} for n in Ns]
sq_line_options.append({"label": "True signal",
                              "linewidth": 2,
                              "color": "black"})

# Encapsulate and Plot signals
sawtooth_fig = FigData(ts,
                       sawtooths,
                       line_options=sawtooth_line_options,
                       title="Sawtooth wave approximated with a Fourier series",
                       **general_options)
square_fig = FigData(ts,
                     squares,
                     line_options=sq_line_options,
                     title="Square wave approximated with a Fourier series",
                     **general_options)

aplot.single_plot(sawtooth_fig)
aplot.single_plot(square_fig)
