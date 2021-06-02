"""
urmp_ica_gen.py

ICA decomposition on stereo URMP.

@author: Chan Wai Lou
"""

from sklearn.decomposition import FastICA
import os
import librosa

import aumix.io.filename as af
import aumix.io.wav as awav

#
# Parameters
#
truth_root = "../../../data/URMP/Dataset/"
source_root = "adress_audio"
audio_out_folder = "ica_audio"

samp_rate = 48000

ids_to_test = range(3, 4)  # 1-11 are 2-instrument

#
# Computation
#

if not os.path.exists(audio_out_folder):
    os.mkdir(audio_out_folder)

# Load stereo audios
source_root_abs = os.path.abspath(source_root)
source_file_dict = af.id_filename_dict(source_root_abs,
                                       ids=ids_to_test,
                                       regexes=[f"{i}-stereo.wav" for i in ids_to_test],
                                       sort_function=lambda f: f)

# Load truth files
for idx in ids_to_test:
    print("working on", idx)

    # Load stereo recording
    stereo_signal = librosa.load(f"{source_root}/{source_file_dict[idx][0]}", sr=samp_rate, mono=False)[0]

    # Decompose with ICA
    transformer = FastICA(n_components=2,
                          random_state=0)

    est_signals = transformer.fit_transform(stereo_signal.T).T

    # Write to file
    for i in range(est_signals.shape[0]):
        awav.write(f"ica_audio/{idx}-ica-{i}", est_signals[i], samp_rate=samp_rate)
