# -*- coding: utf-8 -*-
"""
plot.py

Plotting functions.

@author: Chan Wai Lou / Vincent Lou
"""

import matplotlib.pyplot as pl
# from mpl_toolkits.mplot3d import Axes3D
import numpy as np


# ------------------------- Decorators --------------------------- #

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


# ---------------------------------------------------------------------- #


@savefig
def single_plot(xs: np.ndarray,
                ys: list,
                line_options=[],
                title="",
                xlabel="",
                ylabel="",
                figsize=(9, 7),
                options=[],
                **kwargs
                ):
    """
    Plot ys against xs in a line chart using 1 figure.

    Parameters
    ----------
    xs : np.ndarray
        Data points in the x-axis. 
        For example, np.array([0, 0.1, 0.2, 0.3, 0.4]).
        
    ys : list
        A list of data points stored in `np.ndarray`s.
        Each `np.ndarray` contains elements y_i 
        that is plotted against x_i in xs.
        
    line_options : list, default: []
        A list each containing a dictionary for options to draw each line.
        Not all lines need to be supplied an option, i.e. the length of
        line_options does not need to be the same as ys.
        
        For example, if we have two lines, we can create line_options as so:
            line_options[0] = {
                    label="Line 1"
                }
            line_options[1] = {
                    label="Line 2",
                    linewidth=2.5,
                    color="black"
                }
        
    title : str, default: ""
        The title of the plot.
        
    xlabel : str, default: ""
        The x-axis label.
        
    ylabel : str, default: ""
        The y-axis label.
        
    figsize : tuple, default: (9, 7)
        Figure size of the plot.

    options : list, default: []
        "grid": display a grid behind the plot.
        
    **kwargs : dict
        Other keyward arguments.

    Returns
    -------
    None.

    """
    # Fill up line_options if it's not fully specified
    num_lines = len(ys)
    num_options = len(line_options)
    __line_options = [{} for line in range(num_lines)]

    # Specified in line_options, use it
    for line_idx in range(num_options):
        __line_options[line_idx] = line_options[line_idx]

    # Create the figure
    fig = pl.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    if "grid" in options:
        ax.grid()

    # Plot all ys
    for i, line in enumerate(ys):
        line_len = len(line)
        ax.plot(xs[:line_len], line, **__line_options[i])

    ax.legend()


@savefig
def single_subplots(xs={},
                    ys={},
                    n_rows=1,
                    n_cols=1,
                    individual_figsize=(4, 4),
                    line_options={},
                    titles={},
                    xlabels={},
                    ylabels={},
                    options=[],
                    **kwargs
                    ):

    # Create figure template
    fig = pl.figure(figsize=(individual_figsize[0] * n_rows, individual_figsize[1] * n_cols))
    fig.subplots_adjust(left=-0.2, right=1, top=1, bottom=-0.2)

    # Find figure positions where both the x-axis and y-axis have data
    fig_pos = set(xs.keys()) & set(ys.keys())   # Python 2 friendly

    for key in fig_pos:
        (row, col) = key

        # Fill up line_options if it's not fully specified
        num_lines = len(ys[key])    # ys must contain the key since it's intersected
        num_options = len(line_options.get(key, {}))
        __line_options = [{} for line in range(num_lines)]

        # Specified in line_options, use it
        for line_idx in range(num_options):
            __line_options[line_idx] = line_options[line_idx]

        # Create the specified subplot
        ax = fig.add_subplot(n_rows, n_cols, row * n_cols + col + 1)
        ax.set_title(titles.get(key, ""))
        ax.set_ylabel(ylabels.get(key, ""))
        ax.set_xlabel(xlabels.get(key, ""))

        if "grid" in options:
            ax.grid()

        # Plot all ys
        for i, line in enumerate(ys[key]):
            line_len = len(line)
            ax.plot(xs[key][:line_len], line, **__line_options[i])

        ax.legend()


@savefig
def small_plot(xs: np.ndarray,
               ys: list,
               figsize=(2.75, .5),
               **kwargs
               ):
    """
    Generate a small, thumbnail-styled plot.

    Parameters
    ----------
    xs : np.ndarray
        Data points in the x-axis. 
        For example, np.array([0, 0.1, 0.2, 0.3, 0.4]).
        
    ys : list
        A list of data points stored in `np.ndarray`s.
        Each `np.ndarray` contains elements y_i 
        that is plotted against x_i in xs.
        
    figsize : tuple, default: (9, 7)
        Figure size of the plot.
        
    **kwargs : dict
        Other keyward arguments.

    Returns
    -------
    None.

    """
    fig = pl.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.plot(xs, ys)
    pl.axis('off')
