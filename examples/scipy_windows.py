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
                ('dpss', 3),  # normalized half-bandwidth   *(exists as a function)
                ('exponential', 25, 3),  # center, decay scale
                'flattop',
                ('gaussian', 7),  # std
                ('general_cosine', [1, 1.942604, 1.340318, 0.440811, 0.043097]),   # Sequence of weighting coefficients   *(exists as a function)
                ('general_gaussian', 1.5, 7),  # power, width
                ('general_hamming', 0.5),   # alpha   *(exists as a function)
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
ncols = 3
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

    window_name = str(window)

    try:
        # Generate signal data
        window_data[window_name] = windows.get_window(window, Nx)

    except ValueError as e:
        # Call the function itself if it exists
        if window in windows.__all__:
            window_data[window_name] = getattr(windows, window)(Nx)

        elif type(window) is tuple and window[0] in windows.__all__:
            window_data[window_name] = getattr(windows, window[0])(Nx, *window[1:])

        else:
            print(f"{window} failed: {e}")
            continue

    # Generate frequency response
    A = fft(window_data[window_name], 2048) / (len(window_data[window_name]) / 2.0)
    freq = np.linspace(-0.5, 0.5, len(A))
    response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))

    # Figure position
    curr_fignum = len(window_data)
    print(curr_fignum)
    row = (curr_fignum-1) % nrows
    col = int( np.floor((curr_fignum-1) / nrows) )

    # Create figures
    figs[(row, col)] = FigData(xs=np.array(range(Nx)),
                               ys=[window_data[window_name]],
                               title=window)
    figs_response[(row, col)] = FigData(xs=np.array(freq),
                                        ys=[response],
                                        title=window,
                                        xlim=(-0.5, 0.5),
                                        ylim=(-120, 0))


#
# Plot window data
#

aplot.single_subplots(grid_size=(nrows, ncols),
                      fig_data=figs,
                      individual_figsize=(3.5, 1.6),
                      title="All Scipy Supported Windows",
                      xlabel="Sample",
                      ylabel="Amplitude",
                      savefig_path="scipy_windows")

aplot.single_subplots(grid_size=(nrows, ncols),
                      fig_data=figs_response,
                      individual_figsize=(3.5, 1.6),
                      title="Frequency Response of each window",
                      xlabel="Normalized frequency (cycles per sample)",
                      ylabel="Normalized magnitude (dB)",
                      savefig_path="scipy_windows_freq_responses")
