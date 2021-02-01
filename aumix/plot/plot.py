# -*- coding: utf-8 -*-
"""
plot.py

Plotting functions.

@author: DoraMemo
"""

import matplotlib.pyplot as pl
# from mpl_toolkits.mplot3d import Axes3D
import numpy as np



def savefig(function):
    """
    Save the figure in drawn function if savefig_path is specified.
    
    Intended to be used as a decorator.

    Parameters
    ----------
    function : function
        Plotting function.
    """
    def wrapper(*args, **kwargs):
        plot_func = function(*args, **kwargs)
        
        savefig_path = kwargs.get("savefig_path", None)
        if savefig_path is not None:
            pl.savefig(savefig_path, bbox_inches='tight')
            
        pl.show()
        
        return plot_func
    return wrapper




@savefig
def fourier_plot(ts: np.ndarray, fsignals: list, signal: np.ndarray, 
                 fsignal_labels: list,
                 title="", figsize=(9, 7), **kwargs):
    """
    Plot multiple signals and a true signal in the same plot.

    Parameters
    ----------
    ts : np.ndarray
        Time samples. For example, np.array([0, 0.1, 0.2, 0.3, 0.4]).
        
    fsignals : list
        A list of fourier series approximated signals.
        Each list contains an np.ndarray of same length as ts, where
        each element i represents the amplitude at ts[i].
        
    signal : np.ndarray
        The true signal's amplitudes, stored in an np.ndarray.
        signal should have the same length as ts.
        
    fsignal_labels : list
        A list of string labels for each signal in fsignals.
        
    title : str, optional
        The title of the plot. The default is "".
        
    figsize : tuple, optional
        Size of the figure used in pyplot.figure. The default is (9, 7).
        
    **kwargs : dict
        Other keyward arguments.

    Returns
    -------
    None.

    """
    # Get keyword arguments
    true_linewidth = kwargs.get("true_linewidth", 2.5)
    true_color = kwargs.get("true_color", "black")
    
    # Create the figure
    fig = pl.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_ylabel("Amplitude")
    ax.set_xlabel("time (second)")
    
    # Plot all Fourier signals
    for i, fsignal in enumerate(fsignals):
        signal_len = fsignal.shape[0]
        ax.plot(ts[:signal_len], fsignal, label=fsignal_labels[i])
    
    # Plot the true signal
    ax.plot(ts, signal, label="True signal", color=true_color, linewidth=true_linewidth)

    ax.legend()