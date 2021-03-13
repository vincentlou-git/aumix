"""
non_stationary_signal.py

Piecewise smooth non stationary signals, each chunk being a signal.

@author: DoraMemo
"""

import aumix.signal.stationary_signal as sts
import aumix.signal.simple_signal as ss

import numpy as np


class NonStationarySignal(ss.Signal):
    """
    ...
    """

    def __init__(self, signals: list = None, **kwargs):

        self.signals = [] if signals is None else signals

        base_params = {}
        if len(self.signals) > 0:
            base_params["samp_rate"] = self.signals[0].samp_rate
            base_params["duration"] = sum([s.duration for s in self.signals])

        super().__init__(**base_params, **kwargs)

    def gen_data(self):
        self.data = np.concatenate([sig.data for sig in self.signals])
