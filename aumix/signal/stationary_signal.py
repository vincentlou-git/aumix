# -*- coding: utf-8 -*-
"""
periodic_composite_signal.py

Periodic signals that is a superposition of multiple frequencies.

@author: DoraMemo
"""

import numpy as np

import aumix.signal.fourier_series as fs


class StationarySignal(fs.FourierSeriesSignal):
    """
    A representation of a generic, non-piecewise-smooth signal of multiple frequencies.
    
    Notice Fourier Series states that any such function can be represented
    as a sum of sines and cosines, hence we don't need to specify any
    coefficients other than for sin and cos.

    This is different to FourierSeriesSignal in that the individual frequencies can be specified,
    rather than specifying 1 fundamental frequency (f0) then adding multiples of the fundamental frequency (k*f0).
    
    Parameters
    ----------
    See `FourierSeriesSignal`.
    """

    def __init__(self, cos_freqs=None, sin_freqs=None, **kwargs):
        """
        Initialization

        Parameters
        ----------
        cos_freqs: list
            Frequencies of cosine components.

        sin_freqs: list
            Frequencies of sine components.

        kwargs : dict
            Other keyword arguments.
        """
        n = max(0 if cos_freqs is None else len(cos_freqs), 0 if sin_freqs is None else len(sin_freqs))

        self.freq = None
        self.cos_freqs = [0 for _ in range(n)] if cos_freqs is None else cos_freqs
        self.sin_freqs = [0 for _ in range(n)] if sin_freqs is None else sin_freqs
        super().__init__(n=n, **kwargs)

    # @overrides
    def _cosine_sum(self):
        self.cosine_components = [self.cos_coeffs[n - 1] * np.cos(2 * np.pi * n * self.cos_freqs[n - 1] * self.samp_nums)
                                  for n in self.ns]
        return np.sum(self.cosine_components, axis=0)

    # @overrides
    def _sine_sum(self):
        self.sine_components = [self.sin_coeffs[n - 1] * np.cos(2 * np.pi * n * self.sin_freqs[n - 1] * self.samp_nums)
                                for n in self.ns]
        return np.sum(self.sine_components, axis=0)


class ClarinetApproxSignal(fs.FourierSeriesSignal):
    """
    An approximation of a clarinet sound.
    
    TODO: Amplitude values taken from ...
    """

    def __init__(self, **kwargs):
        super().__init__(n=0, cos_coeffs=None, sin_coeffs=None, **kwargs)

    def gen_data(self):
        odd_amplitudes = [1, 0.75, 0.50, 0.14, 0.50, 0.12, 0.17]
        self.n = len(odd_amplitudes) * 2
        self.ns = range(1, self.n + 1)

        self.sin_coeffs = [0
                           if i % 2 == 0
                           else
                           odd_amplitudes[int((i - 1) / 2)]
                           for i in range(1, self.n + 1)]
        self.cos_coeffs = [0 for _ in range(self.n)]

        # Generate clarinet signal
        super().gen_data()
