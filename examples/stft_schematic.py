"""
stft_schematic.py

Schematic of STFT.

@author: Chan Wai Lou
"""

from scipy import signal
from scipy.fft import fft, fftfreq

import aumix.signal.simple_signal as ss
import aumix.signal.non_stationary_signal as nsts
import aumix.plot.plot as aplot
import aumix.plot.preset as apreset
from aumix.plot.fig_data import *


#
# Parameters
#
n_signals = 4

samp_rate = 120
freqs = range(10, 10 + 10 * n_signals, 10)
durations = [1] * n_signals

window = signal.windows.blackmanharris
window_params = {}
nperseg = samp_rate//2

#
# Generate data
#
half_nperseg = nperseg//2

# Signal
signals = [ss.SineSignal(freq=freqs[i],
                         duration=durations[i],
                         samp_rate=samp_rate)
           for i in range(len(freqs))]
nst = nsts.NonStationarySignal(signals)
padded_data = np.concatenate((np.zeros(nperseg//2), nst.data, np.zeros(nperseg//2)))

# Window
short_w = window(nperseg, **window_params)
xs = np.arange(nst.duration * samp_rate + nperseg) / samp_rate - nperseg / (samp_rate*2)

# FFT (only the positive part)
ws_ffts = list()
slice_num = nst.samp_nums.shape[0] // 2
fft_x = fftfreq(nst.samp_nums.shape[0], 1 / samp_rate)[:slice_num]

# STFT
f, t, Zxx = signal.stft(x=nst.data,
                        fs=nst.samp_rate,
                        window=window(nperseg, **window_params),
                        nperseg=nperseg)

# Window each section of the signal and plot the windowed signal and its FFT
n_segments = 2*nst.samp_nums.shape[0] // nperseg + 1   # zero padded 50% overlap
for seg in range(n_segments):

    # Zero padded window
    curr_w = np.zeros(xs.shape[0])
    curr_w[seg*half_nperseg:(seg+2)*half_nperseg] = short_w

    # Windowed signal
    ws = padded_data * curr_w

    # Compute FFT of the windowed signal.
    # Only store positive frequencies (since the spectrum is symmetric)
    ws_ffts.append(fft(ws)[:slice_num] / np.sqrt(nst.duration * samp_rate))   # normalized

    tau = '{:.2f}'.format(seg*half_nperseg/samp_rate)

    # Encapsulate signals data
    ws_fig = FigData(xs=xs,
                     ys=[ws,
                         curr_w],
                     title=f"tau={tau}s",
                     line_options=[{"label": "Windowed Signal"},
                                   {"label": "Window"}],
                     plot_type="plot",
                     xlabel="Time (s)",
                     ylabel="Amplitude")

    fft_fig = FigData(xs=fft_x[:slice_num],
                      ys=np.abs(ws_ffts[seg])**2,
                      title="Fourier Transform Magnitude",
                      options=["grid"],
                      ylim=(-0.025, 0.25),
                      xlabel="Frequency (Hz)",
                      ylabel="Normalised Amplitude")

    aplot.single_subplots(grid_size=(2, 1),
                          fig_data={(0, 0): ws_fig,
                                    (1, 0): fft_fig},
                          individual_figsize=(5, 2.5),
                          auto_timestamp=False,
                          folder="stft_schematic",
                          savefig_path=f"tau={tau}s.png"
                          )

# Plot the 3D vis of the FFTs which forms the STFT
x, y = np.meshgrid(fft_x, np.arange(0, n_segments) / sum(durations))
fft_stack_fig = FigData(xs=x,
                        ys=y,
                        zs=[np.abs(np.array(ws_ffts))**2],
                        title="STFT surface from stacking FFT",
                        plot_type="plot_surface",
                        xlabel="Frequency (Hz)",
                        ylabel="Time (s)",
                        zlabel="Normalised Amplitude",
                        line_options=[{"cmap": "viridis"}],
                        options=["invert_xaxis"],
                        figsize=(6, 5),
                        colorbar_params={"pad": 0.1,
                                         "vmax": 0.25})
aplot.single_plot(fft_stack_fig,
                  auto_timestamp=False,
                  folder="stft_schematic",
                  savefig_path="fft_stack.png"
                  )

# Plot the original signal and the STFT
signal_fig = FigData(xs=nst.samp_nums,
                     ys=[nst.data],
                     title="Non stationary signal",
                     plot_type="plot",
                     xlabel="Time (s)",
                     ylabel="Amplitude",
                     figsize=(5, 2.5))
aplot.single_plot(signal_fig,
                  auto_timestamp=False,
                  folder="stft_schematic",
                  # savefig_path="signal.png"
                  )


stft_fig = apreset.stft_pcolormesh(t=t,
                                   f=f,
                                   Zxx=Zxx,
                                   title="STFT Magnitude",
                                   yscale="linear",
                                   figsize=(5, 2.5),
                                   colorbar_params={})
aplot.single_plot(stft_fig,
                  auto_timestamp=False,
                  folder="stft_schematic",
                  savefig_path="stft.png"
                  )


