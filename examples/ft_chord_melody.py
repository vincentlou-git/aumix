# -*- coding: utf-8 -*-
"""
ft_chord_melody.py

Comparison of the Fourier transform of a chord (stationary) and a melody (non-stationary).

@author: Chan Wai Lou / Vincent Lou
"""

import aumix.signal.simple_signal as ss
import aumix.signal.stationary_signal as sts
import aumix.signal.non_stationary_signal as nsts
import aumix.plot.plot as aplot
from aumix.plot.fig_data import *
import aumix.music.notes as anotes
import aumix.io.wav as wav

import numpy as np
from scipy.fft import fft, fftfreq

#
# Parameters
#
scale_name = "C2"
n_notes = 3
samp_rate = 44100
durations = [1] * n_notes

#
# Generate signals
#
freqs = anotes.major_chord(scale_name=scale_name, n_notes=n_notes, output="freq")
note_names = " ".join([n.nameWithOctave for n in
                       anotes.major_chord(scale_name=scale_name, n_notes=n_notes,
                                          output="note")])
underline_note_names = note_names.replace(" ", "_")
duration = sum(durations)

signals = [ss.SineSignal(freq=freqs[i],
                         duration=durations[i],
                         samp_rate=samp_rate)
           for i in range(len(freqs))]
melody = nsts.NonStationarySignal(signals)
chord = sts.StationarySignal(sin_freqs=freqs,
                             sin_coeffs=[1] * n_notes,
                             duration=duration,
                             samp_rate=samp_rate)

# Only consider positive frequencies (since the spectrum is symmetric)
melody_fft_x = fftfreq(melody.samp_nums.shape[0], 1 / melody.samp_rate)
chord_fft_x = fftfreq(chord.samp_nums.shape[0], 1 / chord.samp_rate)
slice_num = np.where(melody_fft_x > freqs[-1] * 2)[0][0]

# FFT of both
melody_fft = fft(melody.data)[:slice_num] / (melody.duration * samp_rate)
chord_fft = fft(chord.data)[:slice_num] / (chord.duration * samp_rate)

#
# Figures
#

label = ord('a')

sig_figs = [None, None]
fft_figs = [None, None]

for i, (sig, ft, fft_x, name) in enumerate([(chord, chord_fft, chord_fft_x, "Chord"),
                                            (melody, melody_fft, melody_fft_x, "Melody")]):
    # Encapsulate signals data
    sig_figs[i] = FigData(xs=sig.samp_nums,
                          ys=[sig.data],
                          title=f"({chr(label)}) {name} of {note_names}",
                          plot_type="plot",
                          xlabel="Time (s)",
                          ylabel="Amplitude")
    label += 1

    fft_figs[i] = FigData(xs=fft_x[:slice_num],
                          ys=np.abs(ft) ** 2,
                          title=f"({chr(label)}) Fourier Transform Magnitude of {name}",
                          options=["grid"],
                          xlabel="Frequency (Hz)",
                          ylabel="Normalised Magnitude")

    label += 1

# Plot
aplot.single_subplots(grid_size=(2, 2),
                      fig_data={(0, 0): sig_figs[0],
                                (0, 1): sig_figs[1],
                                (1, 0): fft_figs[0],
                                (1, 1): fft_figs[1]},
                      individual_figsize=(5, 3),
                      auto_timestamp=True,
                      savefig_path=f"{underline_note_names}_{round(duration*1000)}ms_{name}_FFT.png"
                      )

# Output wav
wav.write(f"audio/{underline_note_names}_{round(duration*1000)}ms_melody", melody)
wav.write(f"audio/{underline_note_names}_{round(duration*1000)}ms_chord", chord)
