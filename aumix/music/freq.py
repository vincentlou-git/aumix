"""
freq.py

Module for converting letter notes to frequencies and vice versa.

@author: Chan Wai Lou
"""

notes = [
    "C0", "C#0", "D0", "D#0", "E0", "F0", "F#0", "G0", "G#0", "A0", "A#0", "B0",
    "C1", "C#1", "D1", "D#1", "E1", "F1", "F#1", "G1", "G#1", "A1", "A#1", "B1",
    "C2", "C#2", "D2", "D#2", "E2", "F2", "F#2", "G2", "G#2", "A2", "A#2", "B2",
    "C3", "C#3", "D3", "D#3", "E3", "F3", "F#3", "G3", "G#3", "A3", "A#3", "B3",
    "C4", "C#4", "D4", "D#4", "E4", "F4", "F#4", "G4", "G#4", "A4", "A#4", "B4",
    "C5", "C#5", "D5", "D#5", "E5", "F5", "F#5", "G5", "G#5", "A5", "A#5", "B5",
    "C6", "C#6", "D6", "D#6", "E6", "F6", "F#6", "G6", "G#6", "A6", "A#6", "B6",
    "C7", "C#7", "D7", "D#7", "E7", "F7", "F#7", "G7", "G#7", "A7", "A#7", "B7",
    "C8", "C#8", "D8", "D#8", "E8", "F8", "F#8", "G8", "G#8", "A8", "A#8", "B8",
    "C9", "C#9", "D9", "D#9", "E9", "F9", "F#9", "G9", "G#9", "A9", "A#9", "B9"]


def note2freq(n, octave=0, f0=440):
    """
    Compute the freq of a note using the equal tempered scale.

    fn = f0 * 2^(n/12)
    https://pages.mtu.edu/~suits/NoteFreqCalcs.html

    Parameters
    ----------
    f0: float, default = 440
        Fundamental freq. Commonly this is chosen to be the A above middle C (A4),
        at f0 = 440 Hz.

    octave: int
        Number of octaves above or below the note.

    n: int
        Number of semitones (half steps) away from the fixed note at f0.

    Returns
    -------
    fn: float
        freq of the note "octave" octaves above f0 and n half steps away.

    Examples
    --------
    note2freq(440, 1, 0)   # 880. This is A5
    note2freq(440, 0, -9)   # 261.63. This is C4
    """

    return f0 * 2 ** ((12 * octave + n) / 12)


def notename2freq(name, a4=440):
    """
    Returns the frequency of a letter note.

    Parameters
    ----------
    name: str
        Letter note.

    a4: int, default = 440
        Frequency of A4.

    Returns
    -------
    Frequency of the specified letter note.

    """
    half_steps = notes.index(name.upper()) - notes.index("A4")
    return note2freq(n=half_steps, octave=0, f0=a4)


def freq2note(fn, a4=440, ret_cents=False):
    """
    Frequency to letter note name.
    
    Code taken from https://newt.phys.unsw.edu.au/music/note/,
    translated from JavaScript to Python.
    
    Parameters
    ----------
    fn: float
        Frequency to be converted.
        
    a4: int, default = 440
        Frequency of A4. This can be changed to facilitate tuning.
    
    ret_cents: bool, default = False
        Return cents or not.

    Returns
    -------
    (note_name, cents) if ret_cents is True, otherwise note_name.

    Examples
    --------
    freq2note(440)   # "A4"
    freq2note(261.63)   # "C4"
    freq2note(441, ret_cents=True)   # ("A4", 4)
    """

    if (fn < 27.5) or (fn > 14080):
        raise ValueError("Frequency must lie between 27.5Hz (A0) and 14080Hz (A9)")

    r = 2 ** (1 / 12)
    cent = 2 ** (1 / 1200)
    freq = a4
    r_index = 0
    cent_index = 0

    if fn >= freq:
        while fn >= r * freq:
            freq = r * freq
            r_index += 1
        while fn > cent * freq:
            freq = cent * freq
            cent_index += 1
        if (cent * freq - fn) < (fn - freq):
            cent_index += 1
        if cent_index > 50:
            r_index += 1
            cent_index = 100 - cent_index
            side = -1 if cent_index != 0 else 1
        else:
            side = 1

    else:
        while fn <= freq / r:
            freq = freq / r
            r_index -= 1
        while fn < freq / cent:
            freq = freq / cent
            cent_index += 1
        if (fn - freq / cent) < (freq - fn):
            cent_index += 1
        if cent_index >= 50:
            r_index -= 1
            cent_index = 100 - cent_index
            side = 1
        else:
            side = -1 if cent_index != 0 else 1

    return (notes[notes.index("A4") + r_index], side * cent_index) if ret_cents else notes[
        notes.index("A4") + r_index]
