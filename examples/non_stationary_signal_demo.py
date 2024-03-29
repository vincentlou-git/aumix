"""
non_stationary_signal_demo.py

Example of using the NonStationarySignal class.

@author: DoraMemo
"""

import aumix.signal.simple_signal as ss
import aumix.signal.non_stationary_signal as nsts
import aumix.io.wav as wav
import aumix.plot.plot as aplot
from aumix.plot.fig_data import *


#
# Parameters
#
n_signals = 4
# freqs = range(1, n_signals+1)
durations = [1] * n_signals
# chop_ranges = [(0, 0.5)] * n_signals
freqs = range(220, 220 + 220 * n_signals, 220)
# durations = [max(1 - i * 0.25, 0.25) for i in range(n_signals)]
chop_ranges = [None] * n_signals

#
# Generate data
#
signals = [ss.SineSignal(freq=freqs[i], duration=durations[i], chop_range=chop_ranges[i])
           for i in range(len(freqs))]
nst = nsts.NonStationarySignal(signals)

#
# Encapsulate data in FigData
#

fig = FigData(xs=nst.samp_nums,
              ys=nst.data,
              title="Non stationary signal",
              plot_type="plot",
              xlabel="Time (s)",
              ylabel="Amplitude")

#
# Plot
#
aplot.single_plot(fig)


#
# Output audio
#
wav.write("audio/nonst8", nst, dtype=np.uint8)
