"""
adress_cl_center.py

Experiment of ADRess against non pan-mixed audio.

@author: DoraMemo
"""

import sys
import music21 as m21

import aumix.signal.stationary_signal as sts
import aumix.signal.non_stationary_signal as nst
import aumix.analysis.adress as adress
from aumix.plot.fig_data import *
import aumix.plot.plot as aplot
import aumix.plot.preset as apreset
import aumix.io.wav as wav

#
# Parameters
#

# Audio creation parameters
soprano = ["E4", "C5", "B4", "G4"]  # melody
alto = ["C4", "A4", "E4", "E4"]  # chord w/ bass
tenor = ["G3", "E4", "B3", "C4"]  # melody
bass = ["E3", "C4", "G3", "G3"]  # chord w/ alto
left_intensities = np.array([0.00, 0.00, 1.00, 1.00])
note_duration = 1
samp_rate = 44100

# STFT parameters
window = "hann"
nperseg = 4096  # FFT window length
noverlap = 3072  # nperseg - Step size

# ADRess parameters
left_ds = [0, 50, 101]
left_Hs = [4, 110, 4]
right_ds = [0, 50, 101]
right_Hs = [4, 110, 4]

beta = 100  # Azimuth resolution
est_method = "invert_min_only"

# Output parameters
freq_absence_tol = 5e-2  # Frequency magnitude to tell that "this frequency has nothing"
taus = [note_duration/2 + note_duration*i for i in range(len(soprano))]  # Time positions (in sec) in the STFT to plot the frequency-azimuth figures
left_sep_names = ["Leftmost", "AllLeft", "L-Centermost"]
right_sep_names = ["R-Centermost", "AllRight", "Rightmost"]
name = f"melody-chord-cl-LR={str(left_intensities).replace(' ', ',')}_sr={samp_rate}"
tech_name = f"{name}_{window}_{nperseg}_{noverlap}"

#
# Computation
#

# Format names
short_left_sep_names = [n.split(" ")[0] for n in left_sep_names]
short_right_sep_names = [n.split(" ")[0] for n in right_sep_names]
short_sep_names = short_left_sep_names + short_right_sep_names

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

_, left_recons, right_recons, extra = adress.adress(left_signal=left_signal,
                                                    right_signal=right_signal,
                                                    samp_rate=samp_rate,
                                                    left_ds=left_ds,
                                                    left_Hs=left_Hs,
                                                    right_ds=right_ds,
                                                    right_Hs=right_Hs,
                                                    beta=beta,
                                                    window=window,
                                                    nperseg=nperseg,
                                                    noverlap=noverlap,
                                                    method=est_method,
                                                    print_options=["progress"],
                                                    more_outputs=["stfts", "stft_f", "stft_t"])

t = extra["stft_t"]
f = extra["stft_f"]
left_stft = extra["stfts"]["left"]
right_stft = extra["stfts"]["right"]
left_recon_stfts = extra["stfts"]["left_recons"]
right_recon_stfts = extra["stfts"]["right_recons"]

# Also compute the null/peak spectrogram for each specified tau
left_nulls = []
left_peaks = []
right_nulls = []
right_peaks = []

for i, tau in enumerate(taus):
    ln, lp, rn, rp = adress.adress_null_peak_at_sec(tau,
                                                    t=t,
                                                    left_stft=left_stft,
                                                    right_stft=right_stft,
                                                    beta=beta,
                                                    method=est_method)
    left_nulls.append(ln)
    left_peaks.append(lp)
    right_nulls.append(rn)
    right_peaks.append(rp)


#
# Encapsulate & Plot data
#
fig_data = {}

# Zoom: Find max / min frequency present in the signal (less than some tolerance)
l_ylim = apreset.ylim_zoom(f, left_stft, absence_tol=freq_absence_tol, max_offset=1.1)
r_ylim = apreset.ylim_zoom(f, right_stft, absence_tol=freq_absence_tol, max_offset=1.1)

azi_fig_params = {
    "options": ["grid"],
    "plot_type": "pcolormesh",
    "xlabel": "Azimuth",
    "ylabel": "Frequency"
}
azi_line_options = [{"vmin": 0,
                     "shading": 'gouraud',
                     "cmap": "plasma"}]

# Subplots for both channels
for channel, ylim, nulls, peaks, stft, recon_stfts, sep_names, ds, Hs in \
        (("Left", l_ylim, left_nulls, left_peaks, left_stft,
          left_recon_stfts, left_sep_names, left_ds, left_Hs),
         ("Right", r_ylim, right_nulls, right_peaks, right_stft,
          right_recon_stfts, right_sep_names, right_ds, right_Hs)
         ):
    for i, tau in enumerate(taus):
        azi_line_options[0]["vmax"] = np.max(nulls[i])
        null_fig = FigData(xs=np.arange(beta + 1),
                           ys=f,
                           zs=nulls[i],
                           title=f"Frequency-azimuth spectrogram\n({channel} Channel, tau={'{:.2f}'.format(tau)}s)",
                           line_options=azi_line_options,
                           ylim=ylim,
                           **azi_fig_params)

        azi_line_options[0]["vmax"] = np.max(peaks[i])
        peak_fig = FigData(xs=np.arange(beta + 1),
                           ys=f,
                           zs=peaks[i],
                           title=f"Null magnitude estimation ({est_method})",
                           line_options=azi_line_options,
                           ylim=ylim,
                           **azi_fig_params)

        fig_data[(0, i)] = null_fig
        fig_data[(1, i)] = peak_fig

    stft_src_fig = apreset.stft_pcolormesh(t=t,
                                           f=f,
                                           Zxx=stft,
                                           title=f"{channel} Channel STFT Magnitude w/ {window} window\n"
                                                 f"winlen = {nperseg}, noverlap = {noverlap}",
                                           yscale="linear",
                                           ylim=ylim)
    fig_data[(2, 0)] = stft_src_fig

    recon_stft_figs = [None for i in range(len(recon_stfts))]
    for i, recon_stft in enumerate(recon_stfts):
        recon_stft_figs[i] = apreset.stft_pcolormesh(t=t,
                                                     f=f,
                                                     Zxx=recon_stft,
                                                     title=f"Separated source '{sep_names[i]}'\nat d={ds[i]} with width H={Hs[i]}",
                                                     yscale="linear",
                                                     ylim=ylim)
        fig_data[(2, i+1)] = recon_stft_figs[i]

    # Plot null/peaks, src & source STFTs
    aplot.single_subplots(grid_size=(3, 4),
                          fig_data=fig_data,
                          individual_figsize=(6, 4),
                          # savefig_path=f"ADRess_{channel}_{tech_name}.png"
                          )

#
# Output audio
#

wav.write(f"audio/{name}_stereo", src.T, samp_rate=samp_rate)
wav.write(f"audio/{name}_left", left_signal, samp_rate=samp_rate)
wav.write(f"audio/{name}_right", right_signal, samp_rate=samp_rate)

for i, left_recon in enumerate(left_recons):
    wav.write(f"audio/{tech_name}_{short_sep_names[i]}", left_recon, samp_rate=samp_rate)

for i, right_recon in enumerate(right_recons):
    wav.write(f"audio/{tech_name}_{short_sep_names[i + len(left_recons)]}", right_recon, samp_rate=samp_rate)

