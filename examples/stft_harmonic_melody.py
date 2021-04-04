"""
stft_harmonic_melody.py

Investigation on using scipy's Short-Time Fourier Transform,
on a synthesized clarinet melody.

@author: DoraMemo
"""

from scipy import signal
from scipy.fft import fft, fftfreq

import aumix.music.major as maj
import aumix.signal.stationary_signal as sts
import aumix.signal.non_stationary_signal as nsts
from aumix.io.wav import *
import aumix.plot.plot as aplot
from aumix.plot.fig_data import *

#
# Parameters
#
scale_name = "A#3"
n_notes = 8
samp_rate = 44100
freq_absence_tol = 5e-3

freqs = maj.maj_freqs(scale_name=scale_name, n_notes=n_notes)
durations = [0.25] * n_notes
chop_ranges = [None] * n_notes

window = "blackmanharris"

#
# Generate data
#

# Generate signal
signals = [sts.ClarinetApproxSignal(freq=freqs[i],
                                    duration=durations[i],
                                    samp_rate=samp_rate,
                                    chop_range=chop_ranges[i])
           for i in range(len(freqs))]
nst = nsts.NonStationarySignal(signals)

# Compute FFT to demonstrate that FFT does not retain time information.
nst_fft = fft(nst.data)
nst_fft_x = fftfreq(nst.samp_nums.shape[0], 1 / nst.samp_rate)

# Only consider positive frequencies (since the spectrum is symmetric)
nst_slice_num = nst.samp_nums.shape[0] // 2

# Index of the max frequency present in the signal (less than some tolerance)
# cond = np.abs(nst_fft)[:nst_slice_num] / nst.samp_rate < freq_absence_tol
# fft_max_present_freq_idx = len(cond) - np.where(cond == False)[0][-1] - 1

# Compute STFT
f, t, Zxx = signal.stft(x=nst.data,
                        fs=nst.samp_rate,
                        window=window,
                        nperseg=1024)

Zxx_max = np.max(np.abs(Zxx))

print(t.shape, f.shape, Zxx.shape)

# Index of the max frequency present in the signal (less than some tolerance)
cond = [all(x < freq_absence_tol) for x in Zxx]
cond.reverse()
stft_max_present_freq_idx = len(cond) - cond.index(False) - 1

#
# Encapsulate data
#
signal_fig = FigData(xs=nst.samp_nums,
                     ys=nst.data,
                     title="Non stationary signal",
                     plot_type="plot",
                     xlabel="Time (s)",
                     ylabel="Amplitude")

fft_fig = FigData(xs=nst_fft_x[:nst_slice_num],
                  ys=np.abs(nst_fft[:nst_slice_num]),
                  title=f"FFT of {scale_name} Major scale\n(Artificial clarinet sound)",
                  options=["grid"],
                  xlabel="Frequency (Hz)",
                  ylabel="Amplitude")

fft_zoomed_fig = FigData(xs=nst_fft_x[:nst_slice_num],
                         ys=np.abs(nst_fft[:nst_slice_num]),
                         xlim=(-0.05 * f[stft_max_present_freq_idx],
                               f[stft_max_present_freq_idx]),
                         title=f"FFT (Zoomed) of {scale_name} Major scale\n(Artificial clarinet sound)",
                         options=["grid"],
                         xlabel="Frequency (Hz)",
                         ylabel="Amplitude")

stft_fig = FigData(xs=t,
                   ys=f,
                   zs=np.abs(Zxx),
                   title=f"STFT Magnitude of {scale_name} Major scale\n(Artificial clarinet sound)",
                   line_options=[{"vmin": 0,
                                  "vmax": Zxx_max,
                                  "shading": 'gouraud'}],
                   options=["grid"],
                   plot_type="pcolormesh",
                   xlabel="Time (s)",
                   ylabel="Frequency (Hz)",
                   fit_data=False)

stft_zoomed_fig = FigData(xs=t,
                          ys=f[:stft_max_present_freq_idx],
                          zs=np.abs(Zxx)[:stft_max_present_freq_idx],
                          title=f"STFT Magnitude (Zoomed) of {scale_name} Major scale\n(Artificial clarinet sound)",
                          line_options=[{"vmin": 0,
                                         "vmax": Zxx_max,
                                         "shading": 'gouraud'}],
                          options=["grid"],
                          plot_type="pcolormesh",
                          xlabel="Time (s)",
                          ylabel="Frequency (Hz)",
                          fit_data=False)

#
# Display results
#

aplot.single_plot(fig_data=signal_fig)

aplot.single_subplots(grid_size=(2, 2),
                      fig_data={(0, 0): fft_fig,
                                (1, 0): fft_zoomed_fig,
                                (0, 1): stft_fig,
                                (1, 1): stft_zoomed_fig},
                      individual_figsize=(6, 4),
                      savefig_path=f"STFT_Clarinet_{scale_name}_Major_scale_sampr={samp_rate}"
                      )

# signal2wav(f"audio/cl_{scale_name}_scale", nst)
