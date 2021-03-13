# -*- coding: utf-8 -*-
"""
simple_signal.py

Generates simple signals such as sine waves or square waves.

@author: Chan Wai Lou / Vincent Lou
"""

from abc import ABC, abstractmethod

import numpy as np
from scipy import signal


class Signal(ABC):

    def __init__(self, freq=440, samp_rate=44100, duration=1, chop_range=None, options=None):
        """

        Parameters
        ----------
        freq : int, default: 440
            Frequency of the signal. The default 440 Hz corresponds to
            the note "A".
        samp_rate : int, default: 44100
            Sampling rate of the signal. This along with duration determines the
            number of sample points. The default is the standard mp4 sampling rate of
            44.1kHz.
        duration : number, default: 1
            Duration of the signal in seconds.
        chop_range : tuple, default: None

        options : dict, default: None
            Options for the signal. Available options depend on the implemented class.
        """
        self.samp_rate = samp_rate
        self.duration = duration
        self.freq = freq
        self.samp_nums = np.arange(duration * samp_rate) / samp_rate
        self.datalen = len(self.samp_nums)
        self.chop_range = chop_range

        lower, upper = self._chop_tuple()
        self.samp_nums = self.samp_nums[lower:upper]

        self.options = {} if options is None else options

        self.gen_data()

    @abstractmethod
    def gen_data(self):
        pass

    def _chop_tuple(self):
        lower = self._chop_lower()
        upper = self._chop_upper()

        if lower > upper:
            print("Lower chop range cannot be larger than upper chop range. "
                  "Falling back to not chopping.")
            return 0, self.datalen

        return lower, upper

    def _chop_lower(self):
        lower = 0

        if type(self.chop_range) is tuple and len(self.chop_range) >= 1:

            # Sample number is specified
            if type(self.chop_range[0]) is int:
                lower = min(max(self.chop_range[0], 0), self.datalen)

            # Fraction is specified
            elif type(self.chop_range[0]) is float and abs(self.chop_range[0]) <= 1:
                lower = max(self.chop_range[0] * self.datalen, 0)

            else:
                print(f"Invalid lower chop_range {self.chop_range[0]}.")

        return lower

    def _chop_upper(self):
        upper = self.datalen

        if type(self.chop_range) is tuple and len(self.chop_range) >= 2:

            # Sample number is specified
            if type(self.chop_range[1]) is int:
                upper = min(self.chop_range[1], self.datalen)

            # Fraction is specified
            elif type(self.chop_range[1]) is float and abs(self.chop_range[1]) <= 1:
                upper = max(self.chop_range[1] * self.datalen, 0)

            else:
                print(f"Invalid upper chop_range {self.chop_range[1]}.")

        return upper

    def __str__(self):
        return f"Signal abstract class with freq={self.freq}, samp_rate={self.samp_rate}, duration={self.duration}"

    def __repr__(self):
        return f"Signal(freq={self.freq},samp_rate={self.samp_rate},duration={self.duration})"


class SineSignal(Signal):

    def gen_data(self):
        self.data = np.sin(2 * np.pi * self.freq * self.samp_nums)


class CosineSignal(Signal):

    def gen_data(self):
        self.data = np.cos(2 * np.pi * self.freq * self.samp_nums)


class SquareSignal(Signal):

    def gen_data(self):
        self.data = signal.square(2 * np.pi * self.freq * self.samp_nums)


class SawtoothSignal(Signal):

    def gen_data(self):
        self.data = (signal.sawtooth(2 * np.pi * self.freq * self.samp_nums) + 1) / 2


# Example of using the Signal class
if __name__ == "__main__":
    print(SineSignal(freq=880).data)
