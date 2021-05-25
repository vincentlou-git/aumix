"""
urmp_gen.py

Generation of stereo recordings and ADRess reconstruction using the URMP dataset.

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

# STFT parameters
window = "hann"
nperseg = 4096  # FFT window length
noverlap = 3072  # nperseg - Step size

# ADRess parameters
beta = 200  # Azimuth resolution
# number of sources: azimuth position
positions = {
    2: [25, 120],
    3: [25, 76, 150],
    4: [22, 63, 123, 163],
    5: [20, 53, 90, 138, 174]
}
# number of sources: width
widths = {
    2: [20] * 2,               # 1 - 11
    3: [14, 12, 14],           # 12 - 23
    4: [14, 12, 12, 14],       # 24 - 37
    5: [16, 12, 12, 12, 16],   # 38 - 44
}
# Specify the position and width for specific recordings.
# This is useful for adjusting the d and H parameters for only specific reconstructions.
overwrite_ds = {8: [25],  9: [25],  12: [76],
                14: [75], 17: [76], 18: [76, 150],
                19: [25, 76],       24: [38],
                25: [38],           26: [120, 120],
                27: [60, 62],       31: [63, 63],
                32: [63, 63, 118, 118],  33: [61, 65, 116, 120],
                35: [63, 65, 116, 120],  36: [61, 65, 116, 120],
                37: [116, 120, 124],     39: [25, 50, 56, 130, 135, 140],
                38: [87, 89],
                43: [86, 94, 170, 170, 170]
                }
overwrite_Hs = {5: [10] * 2,      6: [10] * 2,
                7: [10, 30],      8: [10],
                9: [10],          12: [10],
                13: [30, 30, 20], 14: [30],
                15: [30, 32, 20], 16: [30, 30, 30],
                17: [14],         18: [10, 14],
                19: [16, 14],     24: [14],
                25: [14],         26: [18, 16],
                27: [8, 8],       31: [8, 10],
                32: [10, 10, 10, 10],
                35: [10, 10, 14, 14],   36: [10, 10, 14, 14],
                37: [12, 12, 12],       39: [16, 14, 14, 10, 10, 10],
                38: [12, 10],
                42: [14, 14, 16, 18, 20]
                }

# Output parameters
id_taus = {
    5: [5, 5.5, 6],
    6: [5, 5.5, 6],
    7: [15, 152.7, 195],
    9: [1, 2, 3, 4],
    12: [80, 96],
    13: [20, 25],
    14: [28.2, 28.642],
    15: [4, 12],
    16: [13, 15],
    17: [10, 20, 26, 38],
    18: [3.5, 13, 16, 18.5, 23, 34],
    19: [25, 79, 86, 108],
    24: [6, 13, 28, 42.5],
    25: [17.5, 18, 33.5, 34],
    # 38: [13.1, 18.8, 35.7, 47.14, 59.85, 81.6, 106.3],
    # 39: [13.1, 18.8, 35.7, 47.14, 59.85, 81.6, 106.3],
    # 40: [8.946, 17.688, 19, 21.173],
    # 41: [8.946, 17.688, 19, 21.173],
    # 42: [60, 110, 134, 197, 212.7, 202.8],
    # 43: [9.416, 20.04, 39.296, 43.840, 49.6],
    # 44: [12.938, 31.7, 35.808, 107.784, 136.32, 136.452, 136.632]
}
default_taus = []   # Times (second) to visualise
whitelist = [       # Entries to do the analysis. Note: 8, 40, 41 is less than 40s long.
             # 5, 6, 7, 9,                           # 2
             # 12, 13, 14, 15, 16, 17, 18, 19,       # 3
             # 25, 26, 27, 28, 29, 30, 31, 32,       # 4
             # 33, 35,
             38, 40, 42, 43, 44                      # 5
             ]
write_pan_mix = False

#
# Computation
#
pattern = re.compile("AuSep(.*?).wav")

for dir_name, subdir_list, files in os.walk(root):
    # Inside directory: dir_name

    # Find all the parts
    part_names = list()
    for filename in files:
        if pattern.match(filename) is not None:
            part_names.append(filename)

    # If there are at least 2 parts, pan-mix them
    n = len(part_names)
    if n <= 1:
        continue

    # id is the first 2 characters of the dir name
    idx = int(dir_name.split("/")[-1][:2])

    if idx not in whitelist:
        continue

    # Load the parts
    signals = np.array([librosa.load(f"{dir_name}/{part}", sr=samp_rate, mono=True)[0] for part in part_names])

    # Define L/R channel amplitude ratios
    amps = np.arange(0, n) / n * 0.7 + 0.2

    left_signal = np.sum((amps * signals.T).T, axis=0)
    right_signal = np.sum(((1 - amps) * signals.T).T, axis=0)
    src = np.array((left_signal, right_signal))

    # Perform stereo ADRess
    ds = overwrite_ds.get(idx, positions.get(n, [beta // 2] * n))
    Hs = overwrite_Hs.get(idx, widths.get(n, [beta // 40] * n))

    _, recons, extra = adress.adress_stereo(left_signal=left_signal,
                                            right_signal=right_signal,
                                            samp_rate=samp_rate,
                                            ds=ds,
                                            Hs=Hs,
                                            beta=beta,
                                            window=window,
                                            nperseg=nperseg,
                                            noverlap=noverlap,
                                            # print_options=["progress"],
                                            more_outputs=["stfts", "stft_f", "stft_t"])

    # Write audio:
    # Pan-mixed audio
    if write_pan_mix:
        wav.write(f"audio/{idx}-stereo", src.T, samp_rate=samp_rate)

    # Reconstructed audio
    for i, recon in enumerate(recons):
        wav.write(f"audio/{idx}-d={ds[i]}-H={Hs[i]}", recon, samp_rate=samp_rate,
                  # auto_timestamp=True
                  )

    # Grab the other specified quantities
    t = extra["stft_t"]
    f = extra["stft_f"]
    left_stft = extra["stfts"]["left"]
    right_stft = extra["stfts"]["right"]

    # Compute the null/peak spectrogram for each specified tau
    nulls = []
    peaks = []
    taus = id_taus.get(idx, default_taus)

    for i, tau in enumerate(taus):
        nu, p, nargs, _, _ = adress.adress_stereo_null_peak_at_sec(tau,
                                                                   t=t,
                                                                   left_stft=left_stft,
                                                                   right_stft=right_stft,
                                                                   beta=beta)
        nulls.append(nu)

        # The full 2d peaks freq-azi spectrogram
        pmat = np.zeros(nu.shape)
        pmat[np.arange(nargs.shape[0]), nargs] = p
        peaks.append(pmat)

    # Encapsulate & Plot data
    fig_data = {}

    vmax = max(np.max(np.abs(left_stft)), np.max(np.abs(right_stft)))

    azi_fig_params = {
        "options": ["grid"],
        "plot_type": "pcolormesh",
        "xlabel": "Azimuth",
        "ylabel": "Frequency"
    }
    azi_line_options = [{"vmin": 0,
                         "shading": 'gouraud',
                         "cmap": "plasma"}]

    for i, tau in enumerate(taus):
        azi_line_options[0]["vmax"] = np.max(nulls[i])
        null_fig = FigData(xs=np.arange(beta + 1),
                           ys=f,
                           zs=nulls[i],
                           title=f"Frequency-azimuth spectrogram\n(tau={'{:.2f}'.format(tau)}s)",
                           line_options=azi_line_options,
                           ylim=(0, 4000),
                           **azi_fig_params)

        azi_line_options[0]["vmax"] = np.max(peaks[i]) * 0.25
        peak_fig = FigData(xs=np.arange(beta + 1),
                           ys=f,
                           zs=peaks[i],
                           title=f"Null magnitude estimation (tau={'{:.2f}'.format(tau)}s)",
                           line_options=azi_line_options,
                           ylim=(0, 4000),
                           **azi_fig_params)

        fig_data[(0, i)] = null_fig
        fig_data[(1, i)] = peak_fig

    # Plot null/peaks
    aplot.single_subplots(grid_size=(2, len(taus)),
                          fig_data=fig_data,
                          individual_figsize=(6, 4),
                          folder=f"URMP_Analysis/{idx}",
                          savefig_path=f"{taus}.png".replace(" ", ""),
                          auto_timestamp=False,
                          show=False
                          )

