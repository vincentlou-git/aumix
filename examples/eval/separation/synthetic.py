"""
synthetic.py

Evaluation of ADRess separation on the synthetic melodic progression
using standard metrics implemented in `mir_eval` and the scale-related SDR variants.

@author: Chan Wai Lou
"""

import music21 as m21
import scipy.signal as signal
import mir_eval as eval
import pandas as pd

import aumix.analysis.adress as adress
import aumix.signal.stationary_signal as sts
import aumix.signal.non_stationary_signal as nst
from aumix.plot.fig_data import *
import aumix.plot.preset as apreset
import aumix.plot.plot as aplot
import aumix.io.wav as wav
import aumix.analysis.eval as aeval


#
# Parameters
#

# Audio creation parameters
soprano = ["E4", "C5", "B4", "G4"]  # melody
alto = ["C4", "A4", "E4", "E4"]  # chord w/ bass
tenor = ["G3", "E4", "B3", "C4"]  # melody
bass = ["E3", "C4", "G3", "G3"]  # chord w/ alto
left_intensities = np.array([0.5, 0.25, 0.5, 0.75])
note_duration = 1
samp_rate = 44100

# STFT parameters
window = "hann"
nperseg = 4096  # FFT window length
noverlap = 3072  # nperseg - Step size

# ADRess parameters
ds = [100, 30, 165]
Hs = [10, 20, 25]

beta = 200  # Azimuth resolution

# Output parameters
freq_absence_tol = 5e-2  # Frequency magnitude to tell that "this frequency has nothing"
sep_names = ["Melody (Soprano & Tenor)", "Alto (Chord w/ Bass)", "Bass (Chord w/ Alto)"]
name = f"melody-chord-cl-stereo={str(left_intensities).replace(' ', ',')}_sr={samp_rate}"
tech_name = f"{name}_{window}_{nperseg}_{noverlap}"


#
# Computation
#

# Format names
short_sep_names = [n.split(" ")[0] for n in sep_names]

# Generate signal
chords = [m21.chord.Chord((soprano[i], alto[i], tenor[i], bass[i])) for i in range(len(soprano))]

# row = chord, column = melody
chord_freqs = np.array([[pitch.frequency for pitch in chord.pitches] for chord in chords])


note_sigs = [[sts.ClarinetApproxSignal(freq=freq,
                                       duration=note_duration,
                                       samp_rate=samp_rate)
             for freq in chord_freqs.T[i]]
             for i in range(len(chord_freqs))]

melody_data = np.array([nst.NonStationarySignal(ns).data for ns in note_sigs])

left_signal = np.sum((left_intensities * melody_data.T).T, axis=0)
right_signal = np.sum(((1 - left_intensities) * melody_data.T).T, axis=0)
src = np.array((left_signal, right_signal))

# Compute the audio at each position
true_signals = {}
for li in set(left_intensities):
    part_args = np.where(left_intensities == li)   # Which parts have the same intensity?
    true_signals[li] = np.sum(melody_data[part_args], axis=0)
    true_length = true_signals[li].shape[0]

# Perform stereo ADRess
_, recons, extra = adress.adress_stereo(left_signal=left_signal,
                                        right_signal=right_signal,
                                        samp_rate=samp_rate,
                                        ds=ds,
                                        Hs=Hs,
                                        beta=beta,
                                        window=window,
                                        nperseg=nperseg,
                                        noverlap=noverlap,
                                        print_options=["progress"],
                                        more_outputs=["stfts", "stft_f", "stft_t"])

t = extra["stft_t"]
f = extra["stft_f"]
left_stft = extra["stfts"]["left"]
right_stft = extra["stfts"]["right"]
recon_stfts = extra["stfts"]["recons"]


#
# Evaluate
#

# Find out the values for perfect reconstruction (theoretical upper bound)
df = aeval.bss_eval_df(np.arange(1, 2),
                       np.arange(1, 2),
                       compute_permutation=True)
print("Theoretical Upper Bound (Integer True VS. Integer True)")
print(df)

# Practical upper bound
df = aeval.bss_eval_df(np.array(list(true_signals.values())),
                       np.array(list(true_signals.values())),
                       compute_permutation=True)
print("Practical Upper Bound under Numerical noise (Float True VS. Float True)")
print(df)

# True vs. Reconstructed
df = aeval.bss_eval_df(np.array(list(true_signals.values())),
                       np.array(recons)[:, :true_length],
                       compute_permutation=True)
print("True VS. Reconstructed")
print(df)

