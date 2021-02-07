# -*- coding: utf-8 -*-
"""
fourier_transform.py

Experiments in using Fourier Transform.

@author: Chan Wai Lou / Vincent Lou
"""

import aumix.signal.fourier_series as fs
import aumix.signal.periodic_composite_signal as pcs
import aumix.plot.plot as aplot

import numpy as np
from scipy.fft import fft, ifft, fftfreq


# Create a complex signal



# Fourier Transform
sine_fft = fft(sine_signal)
print(sine_fft)

sine_recon = ifft(sine_fft)
print(sine_recon)

slice_num = int(n//2 * 0.05)
sine_fft_x = fftfreq(n, 1/samp_rate)
pl.plot(sine_fft_x[:slice_num], 2.0/n * np.abs(sine_fft[:slice_num]))
pl.grid()
pl.show()