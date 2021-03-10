"""
scipy_windows.py

Investigation on the different window functions used in Short-time Fourier Transform.

@author: DoraMemo
"""

import numpy as np
import scipy.signal.windows as windows
from scipy.fft import fft, fftshift

import aumix.plot.plot as aplot
from aumix.plot.fig_data import *


#
# Parameters
#
Nx = 51

window_names = ['barthann',
                'bartlett',
                'blackman',
                'blackmanharris',
                'bohman',
                'boxcar',
                ('chebwin', 100),  # attenuation
                'cosine',
                ('dpss', 2.5),  # normalized half-bandwidth
                ('exponential', 25, 3),  # center, decay scale
                'flattop',
                ('gaussian', 7),  # std
                'general_cosine',
                ('general_gaussian', 1.5, 7),  # power, width
                'general_hamming',
                'hamming',
                'hann',
                ('kaiser', 14),  # beta
                'nuttall',
                'parzen',
                ('taylor', 5, 50),  # number of constant sidelobes, sidelobe level
                'triang',
                ('tukey', 0.5)  # taper fraction
                ]

total_fignum = len(window_names)
ncols = 4
nrows = int( np.ceil(total_fignum / ncols) )
print(nrows, ncols)

#
# Variables
#
window_data = {}
figs = {}
figs_response = {}

#
# Generate window data
#

for window in window_names:
    try:
        # Generate signal data
        window_data[window] = windows.get_window(window, Nx)

        # Generate frequency response
        A = fft(window_data[window], 2048) / (len(window_data[window]) / 2.0)
        freq = np.linspace(-0.5, 0.5, len(A))
        response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))

        # Figure position
        curr_fignum = len(window_data)
        print(curr_fignum)
        row = (curr_fignum-1) % nrows
        col = int( np.floor((curr_fignum-1) / nrows) )

        # Create figures
        figs[(row, col)] = FigData(xs=np.array(range(Nx)),
                                   ys=[window_data[window]],
                                   title=window)
        figs_response[(row, col)] = FigData(xs=np.array(freq),
                                            ys=[response],
                                            title=window,
                                            xlim=(-0.5, 0.5),
                                            ylim=(-120, 0))
    except ValueError as e:
        pass


#
# Plot window data
#

aplot.single_subplots(grid_size=(nrows, ncols),
                      fig_data=figs,
                      individual_figsize=(4, 2.5),
                      title="All Scipy Supported Windows",
                      xlabel="Sample",
                      ylabel="Amplitude",
                      savefig_path="scipy_windows")

aplot.single_subplots(grid_size=(nrows, ncols),
                      fig_data=figs_response,
                      individual_figsize=(4, 2.5),
                      title="Frequency Response of each window",
                      xlabel="Normalized frequency (cycles per sample)",
                      ylabel="Normalized magnitude (dB)",
                      savefig_path="scipy_windows_freq_responses")
