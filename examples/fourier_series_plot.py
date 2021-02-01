# -*- coding: utf-8 -*-
"""
fourier_series_plot.py

Plots signals approximated with fourier series.

@author: DoraMemo
"""

# import sys
# if "../" not in sys.path:
#     sys.path.append("../")
# if "../aumix/signal" not in sys.path:
#     sys.path.append("../aumix/signal")

import aumix.signal.fourier_series as fs
import aumix.signal.simple_signal as ss
import aumix.plot.plot as aplot

import numpy as np



if __name__ == "__main__":
    
    # parameters
    duration = 4
    samp_rate = 1000
    freq = 1
    sawtooth_params = {
        "duration": duration,
        "samp_rate": samp_rate,
        "freq": freq
    }
    
    Ns = [1, 3, 5, 15, 101]
    ts = np.arange(duration * samp_rate) / samp_rate
    
    
    
    # Generate signals
    fourier_sawtooths = [fs.FourierSawtoothSignal(n=n, 
                                                  **sawtooth_params).data 
                         for n in Ns]
    true_sawtooth = ss.SawtoothSignal(**sawtooth_params).data
    
    fourier_squares = [fs.FourierSquareSignal(n=n, 
                                              **sawtooth_params).data 
                       for n in Ns]
    true_square = ss.SquareSignal(**sawtooth_params).data
    
    
    
    # Figure labels
    flabels = [f"N = {n}" for n in Ns]
    
    # Plot signals
    aplot.fourier_plot(ts, fourier_sawtooths, true_sawtooth, 
                       fsignal_labels=flabels, 
                       title="Sawtooth wave approximated with Fourier series")
    aplot.fourier_plot(ts, fourier_squares, true_square, 
                       fsignal_labels=flabels, 
                       title="Square wave approximated with Fourier series")
    