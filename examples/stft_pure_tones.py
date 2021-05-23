"""
stft_pure_tones.py

Investigation on using scipy's Short-Time Fourier Transform.

@author: DoraMemo
"""

from scipy import signal

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
import aumix.plot.preset as apreset


#
# Parameters
#
n_signals = 4

samp_rate = 100

base_freq = 10
freq_inc = 10

durations = [1] * n_signals
# durations = [max(1 - i * 0.25, 0.25) for i in range(n_signals)]

window = "blackmanharris"

#
# Generate data
#
freqs = np.arange(n_signals) * freq_inc + base_freq

signals = [ss.SineSignal(freq=freqs[i],
                         duration=durations[i],
                         samp_rate=samp_rate)
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
print(t.shape, f.shape, Zxx.shape)
# print(f.shape, f[:10])
# print(t.shape, t[:10])
# print(Zxx.shape, Zxx[:10])
# print(len([stft**2 for stft in Zxx]), [stft**2 for stft in Zxx][:10])

signal_fig = FigData(xs=nst.samp_nums,
                     ys=[nst.data],
                     title=f"Non stationary signal (samp rate = {samp_rate})",
                     plot_type="plot",
                     xlabel="Time (s)",
                     ylabel="Amplitude")

stft_fig = apreset.stft_pcolormesh(t=t,
                                   f=f,
                                   Zxx=Zxx,
                                   title=f"STFT Magnitude",
                                   yscale="linear",
                                   ylim=(0, 50),
                                   colorbar_params={"pad": 0.05})

aplot.single_subplots(grid_size=(1, 2),
                      fig_data={(0, 0): signal_fig,
                                (0, 1): stft_fig},
                      individual_figsize=(5, 3.5),
                      savefig_path="stft_sines.png"
                      )
