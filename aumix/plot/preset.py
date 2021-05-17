"""
preset.py

Module for creating preset FigData and calculating common quantities (e.g. zoom).

@author: Chan Wai Lou
"""

import matplotlib.ticker as ticker

from aumix.plot.fig_data import *


def ylim_zoom(f,
              Zxx,
              absence_tol=5e-2,
              min_offset=0.95,
              max_offset=1.1):
    """
    Zoom: Find max / min frequencies value present in a STFT (less than some tolerance)
    """

    cond = [all(x < absence_tol) for x in Zxx]
    min_freq_idx = cond.index(False) - 1
    cond.reverse()
    max_freq_idx = len(cond) - cond.index(False) - 1
    return f[min_freq_idx] * min_offset, f[max_freq_idx] * max_offset


def stft_pcolormesh(t, f, Zxx,
                    title="STFT Magnitude",
                    xlabel="Time (s)",
                    ylabel="Frequency (Hz)",
                    ylim=None,
                    yscale="log",
                    figsize=(6, 4),
                    options=["grid"],
                    plot_type="pcolormesh",
                    shading="gouraud",
                    max_mag=None,
                    **kwargs
                    ):

    tickers = {
        "yaxis": {
            "major": {
                "locator": ticker.AutoLocator(),
                "formatter": ticker.LogFormatter(labelOnlyBase=False,
                                                 minor_thresholds=(np.inf, np.inf))
            }
        }
    }

    mag = np.abs(Zxx)**2

    return FigData(xs=t,
                   ys=f,
                   zs=mag,
                   title=title,
                   xlabel=xlabel,
                   ylabel=ylabel,
                   line_options=[{"vmin": 0,
                                  "vmax": np.max(mag) if max_mag is None else max_mag,
                                  "shading": shading}],
                   ylim=ylim,
                   yscale=yscale,
                   figsize=figsize,
                   options=options,
                   plot_type=plot_type,
                   tickers=tickers if yscale == "log" else None,
                   fit_data=False,  # This must be False for a 2D zs array
                   **kwargs
                   )
