"""
inverse_stft.py

Inverse short-time Fourier transform attempt.

@author: Chan Wai Lou
"""

from scipy import signal

import aumix.music.major_scale as maj
import aumix.signal.stationary_signal as sts
import aumix.signal.non_stationary_signal as nsts
from aumix.io.wav import *
import aumix.plot.plot as aplot
from aumix.plot.fig_data import *

#
# Parameters
#
scale_name = "A4"
n_notes = 4
samp_rate = 44100
freq_absence_tol = 5e-3
filter_threshold = 3100   # Hz

freqs = maj.maj_freqs(scale_name=scale_name, n_notes=n_notes*2)
freqs = freqs[::2]   # Create arpeggio
durations = [0.25] * n_notes
chop_ranges = [None] * n_notes

window = "blackmanharris"

#
# Generate data
#

half_duration = sum(durations) / 2

# Generate signal
signals = [sts.ClarinetApproxSignal(freq=freqs[i],
                                    duration=durations[i],
                                    samp_rate=samp_rate,
                                    chop_range=chop_ranges[i])
           for i in range(len(freqs))]
nst = nsts.NonStationarySignal(signals)

# Compute STFT
f, t, Zxx = signal.stft(x=nst.data,
                        fs=nst.samp_rate,
                        window=window,
                        nperseg=1024)

Zxx_max = np.max(np.abs(Zxx))


# Filtering. Lowpass the first half and highpass the second half.
# This cannot be done with a single FFT & IFFT since such a threshold function
# is non-linear, hence STFT (or other methods, such as Butterworth's filter) must be used.

freqoff = np.where(f >= filter_threshold)[0][0]
toff = np.where(t >= half_duration)[0][0]
Zxxf = np.zeros(Zxx.shape, dtype=Zxx.dtype)

low = Zxx[:freqoff, :toff]
high = Zxx[freqoff:, toff:]
Zxxf[:freqoff, :toff] = Zxx[:freqoff, :toff]
Zxxf[freqoff:, toff:] = Zxx[freqoff:, toff:]


# Compute Inverse STFT
trec, xrec = signal.istft(Zxxf, samp_rate)


#
# Encapsulate data
#
signal_fig = FigData(xs=nst.samp_nums,
                     ys=nst.data,
                     title=f"{scale_name} Major scale arpeggio (Artificial clarinet sound)",
                     plot_type="plot",
                     xlabel="Time (s)",
                     ylabel="Amplitude")

stft_fig = FigData(xs=t,
                   ys=f,
                   zs=np.abs(Zxx),
                   title=f"STFT Magnitude",
                   line_options=[{"vmin": 0,
                                  "vmax": Zxx_max,
                                  "shading": 'gouraud'}],
                   options=["grid"],
                   plot_type="pcolormesh",
                   xlabel="Time (s)",
                   ylabel="Frequency (Hz)",
                   fit_data=False)

stftf_fig = FigData(xs=t,
                    ys=f,
                    zs=np.abs(Zxxf),
                    title=f"Filtered STFT: Lowpass on first half; highpass on second half\nThreshold at {filter_threshold}Hz",
                    line_options=[{"vmin": 0,
                                   "vmax": Zxx_max,
                                   "shading": 'gouraud'}],
                    options=["grid"],
                    plot_type="pcolormesh",
                    xlabel="Time (s)",
                    ylabel="Frequency (Hz)",
                    fit_data=False)

recon_fig = FigData(xs=trec,
                    ys=xrec,
                    title="Reconstructed after filtering",
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
                                (1, 0): recon_fig},
                      individual_figsize=(6, 4),
                      # savefig_path=f"ISTFT_Clarinet_{scale_name}_Major_scale_arpeggio_sampr={samp_rate}"
                      )

signal2wav(f"audio/istft_cl_{scale_name}_Major_arpeggio", np.concatenate((nst.data, xrec)), samp_rate=samp_rate)
