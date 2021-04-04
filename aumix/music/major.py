"""
major.py

Module for generating major scale frequencies.

@author: DoraMemo
"""

import aumix.music.note as note

import numpy as np

semitone_steps = [2, 2, 1, 2, 2, 2, 1]


def maj_freqs(scale_name, n_notes=8):

    n_octaves = int(np.ceil(n_notes / 8))

    fund_freq = note.notename2freq(scale_name)   # Root note for the scale
    return np.array([note.note2freq(sum((semitone_steps*n_octaves)[:i]), f0=fund_freq) for i in range(n_notes)])
