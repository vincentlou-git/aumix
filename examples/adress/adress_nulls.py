"""
adress_nulls.py

ADRess

@author: DoraMemo
"""

import numpy as np
import copy
from scipy import signal
import librosa
import music21 as m21

import aumix.signal.stationary_signal as sts
from aumix.plot.fig_data import *
import aumix.plot.plot as aplot
import aumix.plot.preset as apreset
import aumix.music.notes as anotes
import aumix.io.wav as wav


#
# Parameters
#

# Audio input parameters
# audio_path = "data/audio/DS-131m.mp3"
chord_names = ["C4", "C#5"]   # Create a CM chord and a C#M chord so they sound clashed together
notes_per_chord = 3
left_intensities = np.array([0.75, 0.75, 0.75, 0.25, 0.25, 0.25])
duration = 1
samp_rate = 44100

# STFT parameters
window = "hann"
nperseg = 4096  # FFT window length
noverlap = 3072  # nperseg - Step size

# ADRess parameters
left_d = 33  # Position of C5
left_H = 4  # Azimuth subspace width
right_d = 33
right_H = 4
beta = 100  # Azimuth resolution
est_method = "invert_min_only"

# Output parameters
freq_absence_tol = 5e-2   # Frequency magnitude to tell that "this frequency has nothing"
tau_idx = 45 // 2   # Time position in the STFT to plot the frequency-azimuth figures
name = f"{str(chord_names).replace(' ', '')}={str(left_intensities).replace(' ', ',')}"

#
# Computation
#


def _null2peak(fas, n_freq_bins, beta, method="invert_all"):
    """Estimates the magnitude of the null in a frequency-azimuth spectrogram."""

    null_args = np.argmin(fas, axis=-1)  # Index of the smallest freq-azi value ("null") in each frequency bin
    max_fa = np.max(fas, axis=-1)  # Max. freq-azi value in each frequency bin
    min_fa = np.min(fas, axis=-1)  # Min. freq-azi value in each frequency bin
    # Estimated peak magnitude
    azi_peak = np.zeros((n_freq_bins, beta + 1))

    if method == "invert_all":
        for k in range(fas.shape[0]):
            for i in range(beta+1):
                azi_peak[k, i] = max_fa[k] - fas[k, i]
    elif method == "invert_min_only":
        for k in range(fas.shape[0]):
            azi_peak[k, null_args[k]] = max_fa[k] - min_fa[k]   # Estimated peak magnitude
    elif method == "fitzgerald":
        pass   # TODO: implement

    return azi_peak


# Generate signal
freqs = [anotes.major_chord(scale_name=sn, n_notes=notes_per_chord, output="freq") for sn in chord_names]
true_sigs = [sts.StationarySignal(sin_freqs=freqs[i],
                                  sin_coeffs=left_intensities[i * notes_per_chord : (i+1) * notes_per_chord],
                                  duration=duration,
                                  samp_rate=samp_rate) for i in range(len(freqs))]

freqs = np.squeeze( np.reshape( freqs, (1, notes_per_chord * len(chord_names)) ) )
lsig = sts.StationarySignal(sin_freqs=freqs,
                            sin_coeffs=left_intensities,
                            duration=duration,
                            samp_rate=samp_rate)
rsig = sts.StationarySignal(sin_freqs=freqs,
                            sin_coeffs=1 - left_intensities,
                            duration=duration,
                            samp_rate=samp_rate)
left_signal = lsig.data
right_signal = rsig.data
src = np.array((left_signal, right_signal))

# Alternatively, load audio from a file
# src = librosa.load(audio_path, sr=44100, mono=False)
# left_signal = src[0][0]
# right_signal = src[0][1]


# Perform STFT on the left and right signals
f, t, left_stft = signal.stft(x=left_signal,
                              fs=samp_rate,
                              window=window,
                              nperseg=nperseg,
                              noverlap=noverlap)

_, _, right_stft = signal.stft(x=right_signal,
                               fs=samp_rate,
                               window=window,
                               nperseg=nperseg,
                               noverlap=noverlap)


# Define variables to store the result in
l_azi_null_tau = None
l_azi_peak_tau = None
r_azi_null_tau = None
r_azi_peak_tau = None

left_sep_stft = np.zeros(left_stft.shape, dtype=np.complex128)
right_sep_stft = np.zeros(right_stft.shape, dtype=np.complex128)


# Source cancellation variables
g = np.array([i / beta for i in range(beta + 1)])
# pos = np.zeros(beta + 1)

left_lower = left_d - int(left_H / 2)
left_upper = left_d + int(left_H / 2)
right_lower = right_d - int(right_H / 2)
right_upper = right_d + int(right_H / 2)

left_azi_null_curr = np.zeros((f.shape[0], beta + 1))
right_azi_null_curr = np.zeros((f.shape[0], beta + 1))

# For each time frame, compute the frequency-azimuth plane
for tau in range(left_stft.shape[1]):
    print(f"{tau}", sep=",") if tau % 10 == 0 else None
    for i in range(beta + 1):
        for freq_idx in range(len(f)):
            left_azi_null_curr[freq_idx, i] = np.abs(left_stft[freq_idx, tau] - g[i] * right_stft[freq_idx, tau])
            right_azi_null_curr[freq_idx, i] = np.abs(right_stft[freq_idx, tau] - g[i] * left_stft[freq_idx, tau])

    # Estimate the magnitude of the nulls
    left_azi_peak_curr = _null2peak(left_azi_null_curr, f.shape[0], beta, method=est_method)
    right_azi_peak_curr = _null2peak(right_azi_null_curr, f.shape[0], beta, method=est_method)

    # Store the null/peaks at the specified tau index
    if tau == tau_idx:
        l_azi_null_tau = copy.deepcopy(left_azi_null_curr)
        l_azi_peak_tau = copy.deepcopy(left_azi_peak_curr)
        r_azi_null_tau = copy.deepcopy(right_azi_null_curr)
        r_azi_peak_tau = copy.deepcopy(right_azi_peak_curr)

    # Magnitude of the sum of each frequency bin of the azimuth subspace (chosen source)
    # (short time power spectrum of the separated source)
    YL = np.sum(left_azi_peak_curr[:, left_lower:left_upper], axis=-1)
    YR = np.sum(right_azi_peak_curr[:, right_lower:right_upper], axis=-1)

    # Calculate the phase (unit angle? in complex number form) of the STFT at the current time tau
    left_curr_phase = left_stft[:, tau] / np.abs(left_stft[:, tau])
    right_curr_phase = right_stft[:, tau] / np.abs(right_stft[:, tau])

    # Combine estimated magnitude and original bin phases
    # Separated range of synthesized STFT values, at time tau
    left_sep_stft[:, tau] = YL * left_curr_phase
    right_sep_stft[:, tau] = YR * right_curr_phase

# Finish separation on all time frames. Inverse it
t_recon, left_recon = signal.istft(left_sep_stft,
                                   samp_rate,
                                   window=window,
                                   nperseg=nperseg,
                                   noverlap=noverlap)

_, right_recon = signal.istft(right_sep_stft,
                              samp_rate,
                              window=window,
                              nperseg=nperseg,
                              noverlap=noverlap)


# Analyze the reconstructed signals with STFT
_, _, left_stft_recon = signal.stft(x=left_recon,
                                    fs=samp_rate,
                                    window=window,
                                    nperseg=nperseg,
                                    noverlap=noverlap)

_, _, right_stft_recon = signal.stft(x=right_recon,
                                     fs=samp_rate,
                                     window=window,
                                     nperseg=nperseg,
                                     noverlap=noverlap)

#
# Encapsulate data
#

# Zoom: Find max / min frequency present in the signal (less than some tolerance)
l_ylim = apreset.ylim_zoom(f, left_stft, absence_tol=freq_absence_tol, max_offset=2.0)
r_ylim = apreset.ylim_zoom(f, right_stft, absence_tol=freq_absence_tol, max_offset=2.0)

azi_fig_params = {
    "options": ["grid"],
    "plot_type": "pcolormesh",
    "xlabel": "Azimuth",
    "ylabel": "Frequency"
}

l_azi_null_fig = FigData(xs=np.arange(beta + 1),
                         ys=f,
                         zs=l_azi_null_tau,
                         title=f"Frequency-azimuth spectrogram (Left Channel, tau={t[tau_idx]}s)",
                         line_options=[{"vmin": 0,
                                        "vmax": np.max(l_azi_null_tau),
                                        "shading": 'gouraud'}],
                         ylim=l_ylim,
                         **azi_fig_params)

l_azi_peak_fig = FigData(xs=np.arange(beta + 1),
                         ys=f,
                         zs=l_azi_peak_tau,
                         title=f"Null magnitude estimation ({est_method})",
                         line_options=[{"vmin": 0,
                                        "vmax": np.max(l_azi_peak_tau),
                                        "shading": 'gouraud'}],
                         ylim=l_ylim,
                         **azi_fig_params)

r_azi_null_fig = FigData(xs=np.arange(beta + 1),
                         ys=f,
                         zs=r_azi_null_tau,
                         title=f"Frequency-azimuth spectrogram (Right Channel, tau={t[tau_idx]}s)",
                         line_options=[{"vmin": 0,
                                        "vmax": np.max(r_azi_null_tau),
                                        "shading": 'gouraud'}],
                         ylim=r_ylim,
                         **azi_fig_params)

r_azi_peak_fig = FigData(xs=np.arange(beta + 1),
                         ys=f,
                         zs=r_azi_peak_tau,
                         title=f"Null magnitude estimation ({est_method})",
                         line_options=[{"vmin": 0,
                                        "vmax": np.max(r_azi_peak_tau),
                                        "shading": 'gouraud'}],
                         ylim=r_ylim,
                         **azi_fig_params)

l_stft_src_fig = apreset.stft_pcolormesh(t=t,
                                         f=f,
                                         Zxx=left_stft,
                                         title=f"Left Channel STFT Magnitude w/ {window} window\n"
                                               f"winlen = {nperseg}, noverlap = {noverlap}",
                                         yscale="linear",
                                         ylim=l_ylim)

r_stft_src_fig = apreset.stft_pcolormesh(t=t,
                                         f=f,
                                         Zxx=right_stft,
                                         title=f"Right Channel STFT Magnitude w/ {window} window\n"
                                               f"winlen = {nperseg}, noverlap = {noverlap}",
                                         yscale="linear",
                                         ylim=r_ylim)

l_stft_rec_fig = apreset.stft_pcolormesh(t=t,
                                         f=f,
                                         Zxx=left_stft_recon,
                                         title=f"Separated source at d={left_d} with width H={left_H}",
                                         yscale="linear",
                                         ylim=l_ylim)

r_stft_rec_fig = apreset.stft_pcolormesh(t=t,
                                         f=f,
                                         Zxx=right_stft_recon,
                                         title=f"Separated source at d={right_d} with width H={right_H}",
                                         yscale="linear",
                                         ylim=r_ylim)

#
# Output
#

# Plot the frequency-azimuth spectrogram at the specified time frame
aplot.single_subplots(grid_size=(2, 2),
                      fig_data={(0, 0): l_azi_null_fig,
                                (1, 0): l_azi_peak_fig,
                                (0, 1): l_stft_src_fig,
                                (1, 1): l_stft_rec_fig,
                                },
                      individual_figsize=(6, 4))

aplot.single_subplots(grid_size=(2, 2),
                      fig_data={(0, 0): r_azi_null_fig,
                                (1, 0): r_azi_peak_fig,
                                (0, 1): r_stft_src_fig,
                                (1, 1): r_stft_rec_fig,
                                },
                      individual_figsize=(6, 4))

# Output audio file
for i, chord in enumerate(chord_names):
    wav.write(f"audio/{name}_true_{chord}", true_sigs[i].data, samp_rate=samp_rate)

wav.write(f"audio/{name}_src", src.T, samp_rate=samp_rate)
wav.write(f"audio/{name}_left", lsig.data, samp_rate=samp_rate)
wav.write(f"audio/{name}_right", rsig.data, samp_rate=samp_rate)

wav.write(f"audio/{est_method}_{name}_sep1", left_recon, samp_rate=samp_rate)
wav.write(f"audio/{est_method}_{name}_sep2", right_recon, samp_rate=samp_rate)


# unused
# Left in stereo space
# for i in range(int(beta/2)+1):
#     # Scan across the stereo space, giving g_i
#     g[i] = i/beta
#
#     # Compute frequency-azimuth plane at current time frame tau
#     for freq_idx in range(len(f)):
#         azi_curr[freq_idx, i] = np.abs(left_stft[freq_idx, tau] - g[i] * right_stft[freq_idx, tau])
#
# # Right in stereo space
# for i in range(int(beta/2)+1, beta+1):
#     # Scan across the stereo space, giving g_i
#     g[i] = (beta-i)/beta
#
#     # Compute frequency-azimuth plane at current time frame tau
#     for freq_idx in range(len(f)):
#         azi_curr[freq_idx, i] = np.abs(right_stft[freq_idx, tau] - g[i] * left_stft[freq_idx, tau])

