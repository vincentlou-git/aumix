# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 20:10:38 2021

@author: Chan Wai Lou / Vincent Lou
"""

import aumix.signal.fourier_series as fs
import aumix.signal.simple_signal as ss
import aumix.plot.plot as aplot

import numpy as np


    
# parameters
duration = 0.01
samp_rate = 44100

cl_f = 466.16 / 2   # written pitch C4, actual pitch is Bb4
cl_pitch_label = "C4"

signal_params = {
    "duration": duration,
    "samp_rate": samp_rate,
    "freq": cl_f,
    "options": {
            "normalize": True
        }
}



# Approximate clarinet signal
cl_odd_amplitudes = [1, 0.75, 0.50, 0.14, 0.50, 0.12, 0.17]
n = len(cl_odd_amplitudes)*2

cl_sin_amplitudes = [0
                     if i % 2 == 0 
                     else 
                     cl_odd_amplitudes[int((i-1)/2)] 
                     for i in range(1, n+1)]
cl_cos_amplitudes = [0 for i in range(n)]
print(cl_sin_amplitudes)

# Generate clarinet signal
cl_signal = fs.FourierSeriesSignal(n=n, 
                                   k=0,
                                   cos_coeffs = cl_cos_amplitudes,
                                   sin_coeffs = cl_sin_amplitudes,
                                   **signal_params)

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
