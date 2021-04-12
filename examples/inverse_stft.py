"""
inverse_stft.py

Inverse short-time Fourier transform attempt.

@author: Chan Wai Lou
"""

import copy
from scipy import signal
import matplotlib.ticker as ticker
import music21 as m21

import aumix.music.notes as notes
import aumix.signal.stationary_signal as sts
import aumix.signal.non_stationary_signal as nsts
import aumix.io.wav as wav
import aumix.plot.plot as aplot
import aumix.plot.preset as apreset
from aumix.plot.fig_data import *

#
# Parameters
#
# Musical parameters
scale_name = "A4"
extra_notes = ["C6", "E6", "A4", "C5"]
n_chords = 5

# Filtering parameters
freq_absence_tol = 5e-2
lopass_threshold = m21.note.Note("B5").pitch.frequency
hipass_threshold = m21.note.Note("D5").pitch.frequency

# Signal parameters
samp_rate = 44100
norm_amp = [[1.5] * 4] * (n_chords - 1) + [[2] * 3]
durations = [1] * n_chords
chop_ranges = [None] * n_chords

# STFT parameters
window_length = 8192
overlap_percent = 0.5

window = signal.windows.hann(window_length, sym=True)             # 50%: COLA satisfied
# window = "boxcar"           # 75%: COLA satisfied

# "Shaky" reconstructions. For check_COLA, see ola.py
# window = "boxcar"           # 25%: COLA not satisfied
# window = "blackmanharris"   # 50%: COLA not satisfied

window_name = window        # Rename explicitly if necessary

#
# Generate data
#

cutoff_duration = sum(durations) * 0.4
noverlap = int(window_length * overlap_percent)

cola = signal.check_COLA(window, window_length, noverlap)
nola = signal.check_NOLA(window, window_length, noverlap)

# A4 C5 E5 for "A4"
chord = notes.minor_chord(scale_name=scale_name, n_notes=3, output="chord")

melody = m21.stream.Stream()
for i in range(1, n_chords+1):
    inv = i % 3
    melody.append(chord)
    chord_copy = copy.deepcopy(chord)
    chord_copy.inversion(inv)   # Move root note to the top
    chord = chord_copy

# Add the extra notes to each chord - we want to remove these after STFT
for i, name in enumerate(extra_notes):
    melody[i].add(name)

# Generate signal
signals = [sts.StationarySignal(sin_freqs=[p.frequency for p in chord.pitches],
                                sin_coeffs=norm_amp[i],
                                duration=durations[i],
                                samp_rate=samp_rate,
                                chop_range=chop_ranges[i])
           for i, chord in enumerate(melody)]
nst = nsts.NonStationarySignal(signals)

# Compute STFT
f, t, Zxx = signal.stft(x=nst.data,
                        fs=nst.samp_rate,
                        window=window,
                        nperseg=window_length,
                        noverlap=noverlap)

Zxx_max = np.max(np.abs(Zxx))


# Filtering. Lowpass the first half and highpass the second half.
# This cannot be done with a single FFT & IFFT since such a threshold function
# is non-linear, hence STFT (or other methods, such as Butterworth's filter) must be used.

looff = np.where(f <= lopass_threshold)[0][-1]
hioff = np.where(f >= hipass_threshold)[0][0]
toff = np.where(t >= cutoff_duration)[0][0]
Zxxf = np.zeros(Zxx.shape, dtype=Zxx.dtype)

low = Zxx[:looff, :toff]
high = Zxx[hioff:, toff:]
Zxxf[:looff, :toff] = low
Zxxf[hioff:, toff:] = high


# Compute Inverse STFT
trec, xrec = signal.istft(Zxxf, samp_rate, window=window, nperseg=window_length, noverlap=noverlap)


#
# Encapsulate data
#

# Zoom: Find max / min frequency present in the signal (less than some tolerance)
ylim = apreset.ylim_zoom(f, Zxx, absence_tol=freq_absence_tol)

stft_fig_params = {
    "line_options": [{"vmin": 0,
                      "vmax": Zxx_max,
                      "shading": 'gouraud'}],
    "options": ["grid"],
    "plot_type": "pcolormesh",
    "xlabel": "Time (s)",
    "ylabel": "Frequency (Hz)",
    "ylim": ylim,
    "yscale": "log",
    "tickers": {
       "yaxis": {
           "major": {
               "locator": ticker.AutoLocator(),
               "formatter": ticker.LogFormatter(labelOnlyBase=False,
                                                minor_thresholds=(np.inf, np.inf))
           }
       }
    },
    "fit_data": False
}

signal_fig = FigData(xs=nst.samp_nums,
                     ys=nst.data,
                     title=f"'Contaminated' {scale_name} minor chord (Pure sine waves)\n"
                           f"Sampling Rate = {samp_rate}Hz",
                     plot_type="plot",
                     xlabel="Time (s)",
                     ylabel="Amplitude")

stft_fig = FigData(xs=t,
                   ys=f,
                   zs=np.abs(Zxx),
                   title=f"STFT Magnitude w/ {window_name} window\n"
                         f"Window length = {window_length}, Overlap = {overlap_percent*100}%\n"
                         f"COLA = {cola}, NOLA = {nola}",
                   **stft_fig_params)

stftf_fig = FigData(xs=t,
                    ys=f,
                    zs=np.abs(Zxxf),
                    title=f"Filtered STFT\n"
                          f"Cutoff duration: {cutoff_duration}s\n"
                          f"Before cutoff: Lowpass below {'{:.2f}'.format(lopass_threshold)}Hz\n"
                          f"After cutoff: Highpass above {'{:.2f}'.format(hipass_threshold)}Hz",
                    **stft_fig_params)

recon_fig = FigData(xs=trec,
                    ys=xrec,
                    title="Reconstructed after filtering:\n"
                          f"{scale_name} minor triads w/ inversions",
                    plot_type="plot",
                    xlabel="Time (s)",
                    ylabel="Amplitude")

#
# Display results
#

aplot.single_subplots(grid_size=(2, 2),
                      fig_data={(0, 1): stft_fig,
                                (1, 1): stftf_fig,
                                (0, 0): signal_fig,
                                (1, 0): recon_fig
                                },
                      individual_figsize=(6, 4),
                      savefig_path=f"ISTFT_{window_name}_{window_length}_{int(overlap_percent*100)}%_Pure_{scale_name}_minor_chord_sampr={samp_rate}"
                      )

wav.write(f"audio/ISTFT_{window_name}_{window_length}_{int(overlap_percent * 100)}%_Pure_{scale_name}m", np.concatenate((nst.data, xrec)), samp_rate=samp_rate)
