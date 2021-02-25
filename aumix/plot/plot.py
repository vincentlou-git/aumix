# -*- coding: utf-8 -*-
"""
plot.py

Plotting functions.

@author: Chan Wai Lou / Vincent Lou
"""

from aumix.plot.fig_data import *

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

    def wrapper(*args, savefig_path=None, **kwargs):
        plot_func = function(*args, **kwargs)

        if savefig_path is not None:
            pl.savefig(savefig_path, bbox_inches='tight')

        pl.show()

        return plot_func

    return wrapper


# ---------------------------------------------------------------------- #


@savefig
def single_plot(fig_data: FigData = None,
                **kwargs
                ):
    """
    Plot ys against xs in a line chart using 1 figure.

    Parameters
    ----------
    fig_data : FigData
        Encapsulated figure data.

    **kwargs : dict
        Other keyward arguments.

    Returns
    -------
    None.

    """

    # Don't plot anything if fig_data isn't specified
    if fig_data is None:
        return

    # Create the figure dict
    fig_dict = {(1, 1, 1): fig_data}
    single_subplots(fig_dict,
                    individual_figsize=fig_data.figsize,
                    **kwargs)

    # Create the figure
    # fig = pl.figure(figsize=fig_data.figsize)
    # ax = fig.add_subplot(111)
    # ax.set_title(fig_data.title)
    # ax.set_ylabel(fig_data.ylabel)
    # ax.set_xlabel(fig_data.xlabel)
    #
    # if "grid" in fig_data.options:
    #     ax.grid()
    #
    # # Plot all ys
    # for i, line in enumerate(fig_data.ys):
    #     line_len = len(line)
    #     ax.plot(fig_data.xs[:line_len], line, **fig_data.line_options[i])
    #
    # ax.legend()


@savefig
def single_subplots(fig_data: dict = None,
                    individual_figsize=(6, 6),
                    title="",
                    **kwargs
                    ):
    """
    Plot one figure with n_rows-by-n_cols subplots.

    Parameters
    ----------
    fig_data: dict, optional
        A dictionary of (subplot_pos, figure_data).
        subplot_pos expects a 3-tuple to be used in fig.add_subuplot.
            For example, (1, 2, 1) means the first subplot with a whole figure having 1 row and 2 columns.
        figure_data expects a FigData object, encapsulating the figure's data.

        Justification: Using a dict rather than a list with each FigData having a "subplot_pos" keyword variable
        is more explicit, and so is preferred.

    individual_figsize: tuple, default: (6, 6)
        Size of each individual subplot.
        Note: the figsize tuple value in each FigData object in fig_data is ignored.

    title: string, default: ""
        Title of the master figure.

    kwargs: dict
        Other keyword arguments.

    Returns
    -------

    """

    # Don't plot anything if fig_data isn't specified, or contains no figure data
    if fig_data is None or len(fig_data) == 0:
        return

    # Find out the maximum required rows and columns
    n_rows = max(fig_data.keys())[0]
    n_cols = max(fig_data.keys(), key=lambda tup: tup[1])[1]

    # Create figure template
    fig = pl.figure(figsize=(individual_figsize[0] * n_cols, individual_figsize[1] * n_rows))
    fig.suptitle(title)
    fig.subplots_adjust(left=-0.2, right=1, top=1, bottom=-0.2)

    for ((row, col, num), f) in fig_data.items():

        # Create the specified subplot
        ax = fig.add_subplot(row, col, num)
        ax.set_title(f.title)
        ax.set_ylabel(f.ylabel)
        ax.set_xlabel(f.xlabel)
        ax.set(**f.kwargs)

        if "grid" in f.options:
            ax.grid()

        # Determine the plot type
        plot_map = {
            "plot": ax.plot,
            "errorbar": ax.errorbar,
            "scatter": ax.scatter,
            "step": ax.step,
            "loglog": ax.loglog,
            "semilogx": ax.semilogx,
            "semilogy": ax.semilogy,
            "bar": ax.bar,
            "barh": ax.barh,
            "stem": ax.stem,
            "eventplot": ax.eventplot,
            "pie": ax.pie,
            "stackplot": ax.stackplot,
            "broken_barh": ax.broken_barh,
            "vlines": ax.vlines,
            "hlines": ax.hlines,
            "fill": ax.fill
        }

        # If the figure does not have a plot type (or the plot type is undefined), use the normal plot
        plot_func = plot_map.get(f.plot_type, ax.plot)

        # Plot data contained in the figure
        for i, line in enumerate(f.ys):
            line_len = len(line)
            plot_func(f.xs[:line_len], line, **f.line_options[i])

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
