# -*- coding: utf-8 -*-
"""
periodic_composite_signal.py

Periodic signals that is a superposition of multiple frequencies.

@author: DoraMemo
"""

import numpy as np

import aumix.signal.fourier_series as fs



class PeriodicCompositeSignal(fs.FourierSeriesSignal):
    """
    A representation of a generic, non-piecewise-smooth signal of mulitple frequencies.
    
    Parameters
    ----------
    See `FourierSeriesSignal`.
    """
    
    def __init__(self, cos_freqs, sin_freqs, **kwargs):
        self.cos_freqs = cos_freqs
        self.sin_freqs = sin_freqs
        super().__init__(**kwargs)
        
        
    def __cosine_sum(self):
        cosine_components = [self.cos_coeffs[n-1] * np.cos(2 * np.pi * n * self.cos_freqs[n-1] * self.samp_nums)
                             for n in self.ns]
        return np.sum(cosine_components, axis=0)
        
    
    def __sine_sum(self):
        sine_components = [self.sin_coeffs[n-1] * np.cos(2 * np.pi * n * self.sin_freqs[n-1] * self.samp_nums)
                           for n in self.ns]
        return np.sum(sine_components, axis=0)
        
        


class ClarinetApproxSignal(fs.FourierSeriesSignal):
    """
    An approximation of a clarinet sound.
    
    TODO: Amplitude values taken from ...
    """
    
    def __init__(self, **kwargs):
        super().__init__(n=0, cos_coeffs=None, sin_coeffs=None, **kwargs)
        
    
    def gen_data(self):
        odd_amplitudes = [1, 0.75, 0.50, 0.14, 0.50, 0.12, 0.17]
        self.n = len(odd_amplitudes)*2
        self.ns = range(1, self.n+1)
        
        self.sin_coeffs = [0
                           if i % 2 == 0 
                           else 
                           odd_amplitudes[int((i-1)/2)] 
                           for i in range(1, self.n+1)]
        self.cos_coeffs = [0 for i in range(self.n)]
        
        # Generate clarinet signal
        super().gen_data()
