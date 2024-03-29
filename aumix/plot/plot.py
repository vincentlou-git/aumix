# -*- coding: utf-8 -*-
"""
plot.py

Plotting functions.

@author: Chan Wai Lou / Vincent Lou
"""

from aumix.plot.fig_data import *

import os
from datetime import datetime
import matplotlib.pyplot as pl
import numpy as np


# Possible 3D plot types
plots_3dable = ["plot", "plot_surface"]


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

    def wrapper(*args, folder="figures", show=True, savefig_path=None, auto_timestamp=True, auto_version_tag=False, **kwargs):
        # TODO: Implement auto version tag by checking the last modified date of the calling script
        plot_func = function(*args, **kwargs)

        if not os.path.exists(folder):
            os.makedirs(folder)

        if savefig_path is not None:
            timestamp = datetime.now().strftime("%y%m%d-%H%M%S") + "-" if auto_timestamp else ""
            pl.savefig(f"{folder}/{timestamp}{savefig_path}", bbox_inches='tight')

        if show:
            pl.show()
        else:
            pl.close()

        return plot_func

    return wrapper


# ---------------------------------------------------------------------- #


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
    fig_dict = {(0, 0): fig_data}
    single_subplots(grid_size=(1, 1),
                    fig_data=fig_dict,
                    individual_figsize=fig_data.figsize,
                    **kwargs)


@savefig
def single_subplots(grid_size,
                    fig_data: dict = None,
                    individual_figsize=(6, 6),
                    title="",
                    dpi=300,
                    **kwargs
                    ):
    """
    Plot one figure with n_rows-by-n_cols subplots with the same width and height.

    Parameters
    ----------
    grid_size: tuple
        A 2-tuple specifying the number of rows and columns of figures.

    fig_data: dict, optional
        A dictionary of (subplot_pos, figure_data).
        subplot_pos expects a 2-tuple or a 4-tuple to be used in fig.add_subuplot.
        Index begins from 0.
            For 2-tuples, e.g. (1, 2) means the subplot will be placed at the
            second row and the third column.
            For 4-tuples, e.g. (0, 0, 2, 1) means a subplot at (0, 0) spanning
            2 rows and 1 column.
        figure_data expects a FigData object, encapsulating the figure's data.

        Justification: Using a dict rather than a list with each FigData having a
        "subplot_pos" keyword variable is more explicit, and so is preferred.

    individual_figsize: tuple, default: (6, 6)
        Size of each individual subplot.
        Note: the figsize tuple value in each FigData object in fig_data is ignored.

    title: string, default: ""
        Title of the master figure.

    dpi: int, default: 300
        dpi.

    kwargs: dict
        Other keyword arguments.

    Returns
    -------

    """

    # Don't plot anything if fig_data isn't specified, or contains no figure data
    if fig_data is None or len(fig_data) == 0:
        return

    # Retrieve number of rows and columns
    n_rows = grid_size[0]
    n_cols = grid_size[1]

    # Create figure template
    fig = pl.figure(figsize=(individual_figsize[0] * n_cols, individual_figsize[1] * n_rows), dpi=dpi)
    fig.suptitle(title)

    # Create a big subplot to set any desired style / parameter
    ax = fig.add_subplot(111, frameon=False)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

    gs = pl.GridSpec(n_rows, n_cols, figure=fig)

    for (fig_pos, f) in fig_data.items():

        # Create the specified subplot
        row = fig_pos[0]
        col = fig_pos[1]
        row_span = fig_pos[2] if len(fig_pos) > 2 else 1
        col_span = fig_pos[3] if len(fig_pos) > 3 else 1
        projection = "3d" if (f.dim == 3 and f.plot_type in plots_3dable) else None

        ax = pl.subplot(gs.new_subplotspec((row, col), rowspan=row_span, colspan=col_span), projection=projection)
        ax.set_title(f.title)
        ax.set_ylabel(f.ylabel)
        ax.set_xlabel(f.xlabel)
        ax.set(**f.kwargs)

        # Perform optional settings
        option_map = {
            "grid": ax.grid,
            "invert_xaxis": ax.invert_xaxis,
            "invert_yaxis": ax.invert_yaxis,
            "invert_zaxis": ax.invert_zaxis if projection is not None else lambda: None,
        }

        for option in f.options:
            func = option_map.get(option, None)
            if func is not None:
                func()

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
            "fill": ax.fill,
            "imshow": ax.imshow,
            "pcolormesh": ax.pcolormesh,
            "plot_surface": ax.plot_surface if projection is not None else ax.plot
        }

        # If the figure does not have a plot type (or the plot type is undefined),
        # use the normal plot
        plot_func = plot_map.get(f.plot_type, ax.plot)

        # Plot data contained in the figure
        for i in range(f.nlines):
            if f.dim == 2 and "xres" in f.options:
                line_len = len(f.ys[i]) if f.fit_data else None
                plot_func(f.xs[i][:line_len], f.ys[i][:line_len], **f.line_options[i])
            elif f.dim == 2:
                line_len = len(f.ys[i]) if f.fit_data else None
                plot_func(f.xs[:line_len], f.ys[i][:line_len], **f.line_options[i])
            elif f.dim == 3:
                line_len = len(f.zs[i]) if f.fit_data else None
                plot_func(f.xs[:line_len], f.ys[:line_len], f.zs[i][:line_len], **f.line_options[i])

        # Tick locator & formatters
        for axis_label, axis_obj in [("xaxis", ax.xaxis), ("yaxis", ax.yaxis)]:

            tick = f.tickers.get(axis_label, {})

            for m, locform in tick.items():

                loca_func = axis_obj.set_major_locator if m == "major" else axis_obj.set_minor_locator
                form_func = axis_obj.set_major_formatter if m == "major" else axis_obj.set_minor_formatter

                loca_obj = locform.get("locator", None)
                form_obj = locform.get("formatter", None)

                loca_func(loca_obj)
                form_func(form_obj)

        # Colorbar
        if type(f.colorbar_params) is dict:

            # Take bounds from colorbar_params
            # If not specified then use the figure's bounds.
            # If it is still not specified, take the value from the range of data
            dat = np.array(f.ys) if f.dim == 2 else np.array(f.zs)
            vmin = f.colorbar_params.get("vmin",
                                         f.line_options[0].get("vmin",
                                                               np.min(dat)))
            vmax = f.colorbar_params.get("vmax",
                                         f.line_options[0].get("vmax",
                                                               np.max(dat)))

            # Use cmap if specified
            cmap = f.colorbar_params.get("cmap",
                                         f.line_options[0].get("cmap",
                                                               pl.get_cmap()))
            sm = pl.cm.ScalarMappable(norm=pl.Normalize(vmin=vmin, vmax=vmax),
                                      cmap=cmap)

            # Remove vmax and vmin from colorbar params
            non_cbp = ["vmin", "vmax"]
            cbp = {k: v for k, v in f.colorbar_params.items() if k not in non_cbp}
            pl.colorbar(sm, ax=ax, **cbp)

        ax.legend(**f.legend_options)
        pl.tight_layout()

    ax.set(**kwargs)


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
