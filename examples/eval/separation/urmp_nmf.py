"""
urmp_nmf.py

Separation of the stereo recordings from the URMP dataset using NMF.
This is a failed attempt.

@author: Chan Wai Lou
"""

import os
import re

import librosa

import aumix.analysis.adress as adress
from aumix.plot.fig_data import *
import aumix.plot.plot as aplot
import aumix.io.wav as wav


#
# Parameters
#

root = "../../../data/URMP/Dataset/"
samp_rate = 48000

n_sources = {
    range(1, 12): 2,
    range(12, 24): 3,
    range(24, 38): 4,
    range(38, 44): 5,
}

whitelist = range(1, 2)

#
# Computation
#
pattern = re.compile("AuMix(.*?).wav")

for dir_name, subdir_list, files in os.walk(root):
    # Inside directory: dir_name

    # Find the mixed file
    mono_filename = None
    for filename in files:
        if pattern.match(filename) is not None:
            mono_filename = filename

    # Did not find a file with this pattern. probably in a wrong directory
    if mono_filename is None:
        continue

    # id is the first 2 characters of the dir name
    idx = int(dir_name.split("/")[-1][:2])

    # number of sources
    n = -1
    for k, v in n_sources.items():
        if idx in k:
            n = v

    # Out of range or index not in whitelist, move on
    if n == -1 or idx not in whitelist:
        continue

    # Load the mono file
    mono = librosa.load(f"{dir_name}/{mono_filename}", sr=samp_rate, mono=True)[0]

    # Decompose the spectrogram with NMF
    S = np.abs(librosa.stft(mono))**2
    comps, acts = librosa.decompose.decompose(S, n_components=n)

    # Reconstruct using each basis
    for i in range(n):
        ind_comp = np.zeros(comps.shape)
        ind_comp[i, :] = comps[i, :]
        ind_act = np.zeros(acts.shape)
        ind_act[:, i] = ind_act[:, i]

        recon_spectrogram = comps @ acts
        recon = librosa.istft(recon_spectrogram)

        # Write file
        wav.write(f"nmf_audio/{idx}-{i}.wav", recon, samp_rate=samp_rate,
                  # auto_timestamp=True
                  )
