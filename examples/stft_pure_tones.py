"""
short_time_fourier_transform.py

Investigation on using scipy's Short-Time Fourier Transform.

@author: DoraMemo
"""

from scipy import signal
import matplotlib.pyplot as pl
import numpy as np

# Generate a test signal,
# a 2 Vrms sine wave whose frequency is slowly modulated around 3kHz,
# corrupted by white noise of exponentially decreasing magnitude sampled at 10 kHz.

# fs = 10e3
# N = 1e5
# amp = 2 * np.sqrt(2)
# noise_power = 0.01 * fs / 2
# time = np.arange(N) / float(fs)
# mod = 500*np.cos(2*np.pi*0.25*time)
# carrier = amp * np.sin(2*np.pi*3e3*time + mod)
# noise = np.random.normal(scale=np.sqrt(noise_power),
#                          size=time.shape)
# noise *= np.exp(-time/5)
# x = carrier + noise


import aumix.signal.simple_signal as ss
import aumix.signal.non_stationary_signal as nsts
import aumix.plot.plot as aplot
from aumix.plot.fig_data import *


#
# Parameters
#
n_signals = 4

samp_rate = 160

# freqs = range(1, n_signals+1)
freqs = range(10, 10 + 10 * n_signals, 10)

durations = [1] * n_signals
# durations = [max(1 - i * 0.25, 0.25) for i in range(n_signals)]

# chop_ranges = [(0, 0.5)] * n_signals
chop_ranges = [None] * n_signals

window = "blackmanharris"

#
# Generate data
#
signals = [ss.SineSignal(freq=freqs[i],
                         duration=durations[i],
                         samp_rate=samp_rate,
                         chop_range=chop_ranges[i])
           for i in range(len(freqs))]
nst = nsts.NonStationarySignal(signals)

# Compute FFT


# Compute STFT
# f: Array of sample frequencies, len 129 real numbers. [0. , 172.265625 , 344.53125...]
# t: Array of segment times, len 1380 real numbers. [0. , 0.00290249 , 0.00580499...]
# Zxx: STFT of x
f, t, Zxx = signal.stft(x=nst.data,
                        fs=nst.samp_rate,
                        window=window,
                        nperseg=100)
print(f.shape, f[:10])
print(t.shape, t[:10])
print(Zxx.shape, Zxx[:10])
# print(len([stft**2 for stft in Zxx]), [stft**2 for stft in Zxx][:10])

signal_fig = FigData(xs=nst.samp_nums,
                     ys=[nst.data],
                     title="Non stationary signal",
                     plot_type="plot",
                     xlabel="Time (s)",
                     ylabel="Amplitude")

stft_fig = FigData(xs=t,
                   ys=f,
                   zs=np.abs(Zxx),
                   title="STFT Magnitude",
                   line_options=[{"vmin": 0,
                                  "vmax": 1,
                                  "shading": 'gouraud'}],
                   options=["grid"],
                   plot_type="pcolormesh",
                   xlabel="Time (s)",
                   ylabel="Frequency (Hz)")

# aplot.single_plot(signal_fig)

# pl.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=1, shading='gouraud')
# pl.title('STFT Magnitude')
# pl.ylabel('Frequency [Hz]')
# pl.xlabel('Time [sec]')
# pl.show()

aplot.single_subplots(grid_size=(2, 1),
                      fig_data={(0, 0): signal_fig,
                                (1, 0): stft_fig},
                      individual_figsize=(6, 4))
