"""
urmp.py

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
    4: [25, 63, 118, 162],
    5: [25, 50, 90, 137, 170]
}
# number of sources: width
widths = {
    2: [20] * 2,
    3: [20] * 3,
    4: [22] * 4,
    5: [22] * 5,
}
overwrite_ds = {24: [162, 125, 63, 35]}
overwrite_Hs = {12: [5] * 3,
                24: [20, 20, 20, 20]}

# Output parameters
freq_absence_tol = 5e-2  # Frequency magnitude to tell that "this frequency has nothing"
taus = [5, 15, 25, 35]   # Times (second) to visualise
whitelist = [8, 40, 41]   # Entries to do the analysis. Note: 8, 40, 41 is less than 40s long.

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

