"""
adress_cl_stereo.py

Experiment of the stereo version of ADRess.

@author: Chan Wai Lou
"""

import music21 as m21
import scipy.signal as signal

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
taus = [note_duration/2 + note_duration*i for i in range(len(soprano))]  # Time positions (in sec) in the STFT to plot the frequency-azimuth figures
sep_names = ["Melody (Soprano & Tenor)", "Alto (Chord w/ Bass)", "Bass (Chord w/ Alto)"]
name = f"melody-chord-cl-centre={str(left_intensities).replace(' ', ',')}_sr={samp_rate}"
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
             for i in range(chord_freqs.shape[1])]

melody_data = np.array([nst.NonStationarySignal(ns).data for ns in note_sigs])

left_signal = np.sum((left_intensities * melody_data.T).T, axis=0)
right_signal = np.sum(((1 - left_intensities) * melody_data.T).T, axis=0)
src = np.array((left_signal, right_signal))

# Compute the audio at each position
true_signals = {}
for li in set(left_intensities):
    part_args = np.where(left_intensities == li)   # Which parts have the same intensity?
    true_signals[li] = np.sum(melody_data[part_args], axis=0)

# Perform FitzGerald's ADRess
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

# Also compute the null/peak spectrogram for each specified tau
nulls = []
peaks = []

for i, tau in enumerate(taus):
    n, p, nargs, _, _ = adress.adress_stereo_null_peak_at_sec(tau,
                                                              t=t,
                                                              left_stft=left_stft,
                                                              right_stft=right_stft,
                                                              beta=beta)
    nulls.append(n)

    # The full 2d peaks freq-azi spectrogram
    pmat = np.zeros(n.shape)
    pmat[np.arange(nargs.shape[0]), nargs] = p
    peaks.append(pmat)


#
# Encapsulate & Plot data
#
fig_data = {}

# Zoom: Find max / min frequency present in the signal (less than some tolerance)
l_ylim = apreset.ylim_zoom(f, left_stft, absence_tol=freq_absence_tol, max_offset=1.1)
r_ylim = apreset.ylim_zoom(f, right_stft, absence_tol=freq_absence_tol, max_offset=1.1)
global_ylim = (min(l_ylim[0], r_ylim[0]), max(l_ylim[1], r_ylim[1]))

azi_fig_params = {
    "options": ["invert_xaxis"],
    "plot_type": "pcolormesh",
    "xlabel": "Azimuth",
    "ylabel": "Frequency"
}
azi_line_options = [{"vmin": 0,
                     "shading": 'gouraud',
                     "cmap": "plasma"}]

for i, (channel, stft, ylim) in enumerate((("Left", left_stft, l_ylim), ("Right", right_stft, r_ylim))):
    channel_stft_fig = apreset.stft_pcolormesh(t=t,
                                               f=f,
                                               Zxx=stft,
                                               title=f"{channel} Channel STFT Magnitude w/ {window} window\n"
                                                     f"winlen = {nperseg}, noverlap = {noverlap}",
                                               yscale="linear",
                                               ylim=ylim,
                                               colorbar_params={})
    fig_data[(0, i)] = channel_stft_fig

for i, tau in enumerate(taus):
    azi_line_options[0]["vmax"] = np.max(nulls[i])
    null_fig = FigData(xs=np.arange(beta + 1),
                       ys=f,
                       zs=nulls[i],
                       title=f"Freq-azi spectrogram (tau={'{:.2f}'.format(tau)}s)",
                       line_options=azi_line_options,
                       ylim=global_ylim,
                       **azi_fig_params)

    azi_line_options[0]["vmax"] = np.max(peaks[i])*0.25
    peak_fig = FigData(xs=np.arange(beta + 1),
                       ys=f,
                       zs=peaks[i],
                       title=f"Null magnitude estimation (tau={'{:.2f}'.format(tau)}s)",
                       line_options=azi_line_options,
                       ylim=global_ylim,
                       **azi_fig_params)

    fig_data[(1, i)] = null_fig
    fig_data[(2, i)] = peak_fig

recon_stft_figs = [None for i in range(len(recon_stfts))]
for i, recon_stft in enumerate(recon_stfts):
    recon_stft_figs[i] = apreset.stft_pcolormesh(t=t,
                                                 f=f,
                                                 Zxx=recon_stft,
                                                 title=f"Separated source '{sep_names[i]}'\nat d={ds[i]} with width H={Hs[i]}",
                                                 yscale="linear",
                                                 ylim=global_ylim,
                                                 colorbar_params={}
                                                 )
    fig_data[(3, i)] = recon_stft_figs[i]

# True signals at each stereo position
for i, (pos, audio) in enumerate(true_signals.items()):
    _, _, stft = signal.stft(x=audio,
                             fs=samp_rate,
                             window=window,
                             nperseg=nperseg,
                             noverlap=noverlap)
    true_pos_stft_fig = apreset.stft_pcolormesh(t=t,
                                                f=f,
                                                Zxx=stft,
                                                title=f"True signal at stereo position {pos}",
                                                yscale="linear",
                                                ylim=global_ylim,
                                                colorbar_params={})
    fig_data[(4, i)] = true_pos_stft_fig

# Plot L/R STFTs, null/peaks, recon STFTs & true STFTs at each stereo pos
aplot.single_subplots(grid_size=(5, 4),
                      fig_data=fig_data,
                      individual_figsize=(4.8, 3),
                      savefig_path=f"StereoADRess_{tech_name}.png",
                      show=False
                      )

aplot.single_subplots(grid_size=(4, 2),
                      fig_data={(0, 0): fig_data[(0, 0)],
                                (0, 1): fig_data[(0, 1)]},
                      individual_figsize=(4.8, 3),
                      savefig_path=f"ADRess_{tech_name}_LR.png",
                      show=False
                      )

aplot.single_subplots(grid_size=(4, 2),
                      fig_data={(0, 0): fig_data[(1, 0)],
                                (0, 1): fig_data[(2, 0)],
                                (1, 0): fig_data[(1, 1)],
                                (1, 1): fig_data[(2, 1)],
                                (2, 0): fig_data[(1, 2)],
                                (2, 1): fig_data[(2, 2)],
                                (3, 0): fig_data[(1, 3)],
                                (3, 1): fig_data[(2, 3)]},
                      individual_figsize=(4.8, 3),
                      savefig_path=f"ADRess_{tech_name}_NP.png",
                      show=False
                      )

aplot.single_subplots(grid_size=(4, 2),
                      fig_data={(0, 0): fig_data[(3, 0)],
                                (0, 1): fig_data[(4, 0)],
                                (1, 0): fig_data[(3, 1)],
                                (1, 1): fig_data[(4, 1)],
                                (2, 0): fig_data[(3, 2)],
                                (2, 1): fig_data[(4, 2)]},
                      individual_figsize=(4.8, 3),
                      savefig_path=f"ADRess_{tech_name}_RECON.png",
                      show=False
                      )

#
# Output audio
#

wav.write(f"audio/{name}_stereo", src.T, samp_rate=samp_rate)
wav.write(f"audio/{name}_left", left_signal, samp_rate=samp_rate)
wav.write(f"audio/{name}_right", right_signal, samp_rate=samp_rate)

# True audio at each stereo position
for pos, audio in true_signals.items():
    wav.write(f"audio/{name}_pos={pos}", audio, samp_rate=samp_rate)

# Reconstructed audio
for i, recon in enumerate(recons):
    wav.write(f"audio/{tech_name}_{short_sep_names[i]}", recon, samp_rate=samp_rate,
              auto_timestamp=True
              )

