"""
wav.py

Module to read / write .wav files from numerical data.

@author: Chan Wai Lou / Vincent Lou
"""

import numpy as np
from scipy.io import wavfile
import os

import aumix.signal.simple_signal as ss


def write(filename, signal, samp_rate=None, dtype=np.int32):
    """
    Convert from numerical data to .wav.

    Parameters
    ----------
    filename : str
        Filename of the output WITHOUT ".wav" appended.

    signal : np.ndarray, or aumix.signal.simple_signal.Signal
        An array containing the numerical data, or a Signal class with the numerical data
        encapsulated inside.

    samp_rate : int, optional
        Sampling rate.
        If `signal` is an array, this needs to be specified.
        If `signal` is a Signal class, its samp_rate field should be specified.

    dtype
        Data type of the output .wav file. 4 resolution are supported as follows:

        np.uint8 : 8-bit PCM
        np.int16 : 16-bit PCM
        np.int32 : 32-bit PCM
        np.float32 : 32-bit floating point

    Returns
    -------
    None
    """

    data = signal

    # If signal is a Signal class, then retrieve data and sampling rate from it
    if isinstance(signal, ss.Signal):
        data = signal.data
        samp_rate = signal.samp_rate

    # If sampling rate is undefined, we don't have enough info to output a .wav file
    if samp_rate is None:
        raise ValueError("Sampling rate is undefined.")

    # Find out the maximum size of the specified type
    amplitude = np.iinfo(dtype).max if dtype != np.float32 else 1

    # Treat 8-bit PCM as a special case, since the numbers are unsigned.
    if dtype == np.uint8:
        unsigned_data = data - np.min(data)
        out_data = unsigned_data * amplitude / np.max(unsigned_data)
    else:
        out_data = data * amplitude / np.max(data)

    # Create folder if it doesn't exist
    folder = "/".join(filename.split("/")[:-1])
    if folder != "" and not os.path.exists(folder):
        os.makedirs(folder)

    # Output file
    wavfile.write(f"{filename}.wav", samp_rate, out_data.astype(dtype))
