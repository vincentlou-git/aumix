"""
stft_harmonic_melody.py

Investigation on using scipy's Short-Time Fourier Transform,
on a synthesized clarinet melody.

@author: DoraMemo
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
scale_name = "A#3"
n_notes = 8
samp_rate = 44100

freqs = maj.maj_freqs(scale_name=scale_name, n_notes=n_notes)
durations = [0.25] * n_notes
chop_ranges = [None] * n_notes

window = "blackmanharris"

#
# Generate data
#
signals = [sts.ClarinetApproxSignal(freq=freqs[i],
                                    duration=durations[i],
                                    samp_rate=samp_rate,
                                    chop_range=chop_ranges[i])
           for i in range(len(freqs))]
nst = nsts.NonStationarySignal(signals)

# Compute FFT


# Compute STFT
f, t, Zxx = signal.stft(x=nst.data,
                        fs=nst.samp_rate,
                        window=window,
                        nperseg=1024)

Zxx_max = np.max(np.abs(Zxx))

print(t.shape, f.shape, Zxx.shape)

# Index of the max frequency present in the signal (less than some tolerance)
tol = 5e-3
cond = [all(x < tol) for x in Zxx]
cond.reverse()
max_present_freq_idx = len(cond) - cond.index(False) - 1

#
# Encapsulate data
#
signal_fig = FigData(xs=nst.samp_nums,
                     ys=nst.data,
                     title="Non stationary signal",
                     plot_type="plot",
                     xlabel="Time (s)",
                     ylabel="Amplitude")

stft_fig = FigData(xs=t,
                   ys=f,
                   zs=np.abs(Zxx),
                   title=f"STFT Magnitude of {scale_name} Major scale\n(Approximated clarinet sound)",
                   line_options=[{"vmin": 0,
                                  "vmax": Zxx_max,
                                  "shading": 'gouraud'}],
                   options=["grid"],
                   plot_type="pcolormesh",
                   xlabel="Time (s)",
                   ylabel="Frequency (Hz)",
                   fit_data=False)

stft_zoomed_fig = FigData(xs=t,
                          ys=f[:max_present_freq_idx],
                          zs=np.abs(Zxx)[:max_present_freq_idx],
                          title=f"STFT Magnitude (Zoomed) of {scale_name} Major scale\n(Approximated clarinet sound)",
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

aplot.single_subplots(grid_size=(2, 1),
                      fig_data={(0, 0): stft_fig,
                                (1, 0): stft_zoomed_fig},
                      individual_figsize=(6, 4),
                      savefig_path=f"STFT_Clarinet_{scale_name}_Major_scale_sampr={samp_rate}"
                      )

# signal2wav(f"audio/cl_{scale_name}_scale", nst)
