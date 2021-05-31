"""
notes.py

Module for generating scales and chords.

@author: Chan Wai Lou
"""

import aumix.music.freq as freq

import music21 as m21
import numpy as np


def __output_helper(notes, output):

    output = output.lower()

    if output == "note":
        return notes

    elif output == "stream":
        return m21.stream.Stream(notes)

    elif output == "chord":
        return m21.chord.Chord(notes)

    elif output == "freq":
        return np.array([note.pitch.frequency for note in notes])

    else:
        # Attempt to grab the specified attribute of each note. Only 1 level supported
        # i.e. note.pitch.frequency will not work.
        return [getattr(note, output) for note in notes]


def __notes_freqs(root_note, n_notes, step_list):
    """Deprecated"""

    n_octaves = int(np.ceil(n_notes / len(step_list)))
    full_range = step_list * n_octaves

    fund_freq = freq.notename2freq(root_note)
    return np.array([freq.note2freq(sum(full_range[:i]), f0=fund_freq) for i in range(n_notes)])


def __notes(root_note, n_notes, step_list):

    n_octaves = int(np.ceil(n_notes / len(step_list)))
    full_range = step_list * n_octaves

    note = m21.note.Note(root_note)
    return [note.transpose(sum(full_range[:i])) for i in range(n_notes)]


#
# Scales
#
def major_scale(scale_name, n_notes=8, output="note"):
    """Major scale"""

    notes = __notes(scale_name, n_notes, [2, 2, 1, 2, 2, 2, 1])
    return __output_helper(notes, output)


def minor_nat_scale(scale_name, n_notes=8, output="note"):
    """Natural minor scale"""

    notes = __notes(scale_name, n_notes, [2, 1, 2, 2, 1, 2, 2])
    return __output_helper(notes, output)


def minor_har_scale(scale_name, n_notes=8, output="note"):
    """Harmonic minor scale"""

    notes = __notes(scale_name, n_notes, [2, 1, 2, 2, 1, 3, 1])
    return __output_helper(notes, output)


def minor_mel_scale(scale_name, n_notes=8, output="note"):
    """Melodic minor scale"""

    notes = __notes(scale_name, n_notes, [2, 1, 2, 2, 2, 2, 1])
    return __output_helper(notes, output)


#
# Chords
#
def major_chord(scale_name, n_notes=8, output="note"):

    notes = __notes(scale_name, n_notes, [4, 3, 5])
    return __output_helper(notes, output)


def minor_chord(scale_name, n_notes=8, output="note"):

    notes = __notes(scale_name, n_notes, [3, 4, 5])
    return __output_helper(notes, output)


def dim_chord(scale_name, n_notes=8, output="note"):

    notes = __notes(scale_name, n_notes, [3, 3, 6])
    return __output_helper(notes, output)


def aug_chord(scale_name, n_notes=8, output="note"):

    notes = __notes(scale_name, n_notes, [4, 4, 4])
    return __output_helper(notes, output)
