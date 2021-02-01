# -*- coding: utf-8 -*-
"""
fourier_series.py

Generates signals approximated with Fourier series.

@author: Chan Wai Lou
"""

import numpy as np

import simple_signal as ss



class FourierSignal(ss.Signal):
    
    def __init__(self, n, **kwargs):
        self.n = n
        super().__init__(**kwargs)



class FourierSquareSignal(FourierSignal):
    
    def gen_data(self):
        
        n_odds = range(1, self.n+1, 2)
        self.data = []
        
        for t in self.samp_nums:
            sum_terms = [1/n * np.sin(2 * n * np.pi * self.freq * t) for n in n_odds]
    
            y = 4/np.pi * sum(sum_terms)
            self.data.append(y)
            
        self.data = np.array(self.data)



class FourierSawtoothSignal(FourierSignal):
    
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
        \frac{1}{2} k + \sum_{i=1}^{N} a_{i} cos(nf)
    
    """
    
    def gen_data(self):
        
        ns = range(1, self.n+1)
        self.data = []
        
        self.data = np.array(self.data)

        
        
        
if __name__ == "__main__":
    print(FourierSawtoothSignal(5).data)