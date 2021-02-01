# -*- coding: utf-8 -*-
"""
fourier_series.py

Generates signals approximated with Fourier series.

@author: Chan Wai Lou / Vincent Lou
"""

import numpy as np

import simple_signal as ss



class FourierSignal(ss.Signal):
    
    def __init__(self, n, **kwargs):
        self.n = n
        super().__init__(**kwargs)



class FourierSquareSignal(FourierSignal):
    """Represents a square signal approximated with Fourier series."""
    
    def gen_data(self):
        
        n_odds = range(1, self.n+1, 2)
        self.data = []
        
        for t in self.samp_nums:
            sum_terms = [1/n * np.sin(2 * n * np.pi * self.freq * t) for n in n_odds]
    
            y = 4/np.pi * sum(sum_terms)
            self.data.append(y)
            
        self.data = np.array(self.data)



class FourierSawtoothSignal(FourierSignal):
    """Represents a sawtooth signal approximated with Fourier series."""
    
    def gen_data(self):
        
        ns = range(1, self.n+1)
        self.data = []
        
        for t in self.samp_nums:
            sum_terms = [1/n * np.sin(2 * n * np.pi * self.freq * t) for n in ns]
    
            y = 1/2 - 1/np.pi * sum(sum_terms)
            self.data.append(y)
        
        self.data = np.array(self.data)



class FourierSeriesSignal(FourierSignal):
    """
    Represent a general Fourier series approximated function.
    
    .. math::
        0.5 k + \sum_{n=1}^{N} a_{n} cos(2\pi nf) + b_{n} sin(2\pi nf)
    
    """
    
    def __init__(self, k, cos_coeffs, sin_coeffs, **kwargs):
        self.k = k
        self.cos_coeffs = cos_coeffs
        self.sin_coeffs = sin_coeffs
        super().__init__(**kwargs)
        
    
    def gen_data(self):
        
        ns = range(1, self.n+1)
        
        # cosines
        cosine_components = [self.cos_coeffs[n-1] * np.cos(2 * np.pi * n * self.freq * self.samp_nums)
                             for n in ns]
        cosine_sum = np.sum(cosine_components, axis=0)
        
        # sines
        sine_components = [self.sin_coeffs[n-1] * np.sin(2 * np.pi * n * self.freq * self.samp_nums)
                           for n in ns]
        sine_sum = np.sum(sine_components, axis=0)
        
        # add everything up
        self.data = 0.5 * self.k + cosine_sum + sine_sum
        
        if self.options.get("normalize", False):
            max_amp = max(self.data)
            self.data /= max_amp
        
        self.data = np.array(self.data)

        
        
       
# Example of using the FourierSignal class 
if __name__ == "__main__":
    print(FourierSawtoothSignal(5).data)