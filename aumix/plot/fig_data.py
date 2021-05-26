"""
fig_data.py

Figure data encapsulation class.

@author: Chan Wai Lou / Vincent Lou
"""

import numpy as np


class FigData:
    """Encapsulation class for a figure."""

    def __init__(self,
                 xs: np.ndarray = None,
                 ys=None,
                 zs=None,
                 line_options=None,
                 title="",
                 xlabel="",
                 ylabel="",
                 figsize=(9, 7),
                 plot_type="plot",
                 fit_data=True,
                 tickers=None,
                 options=None,
                 legend_options=None,
                 colorbar_params=None,
                 **kwargs
                 ):
        """
        xs : np.ndarray, optional
            An array of data points in the x-axis.
            For example, np.array([0, 0.1, 0.2, 0.3, 0.4]).

        ys : list, np.ndarray, optional
            A list of data point arrays in the y-axis, or an array of data points if
            zs is specified.
            Each item is a line in the figure.
            Each `np.ndarray` contains elements y_i that is plotted against x_i in xs.

        zs : list, optional
            A list of data point arrays in the z-axis.

        line_options : list, optional
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

        plot_type : str, default: "plot"
            A string indicator for the plot type.
            This can be detected in some plotting function for different chart types.

        fit_data : bool, default: True
            An indicator for whether to only plot data up to the array length
            of the last axis or not.
            This is merely an indicator and does not affect any values stored inside the
            object.

        tickers : dict, optional
            A dictionary for configuring the tick position and format of each axis.
            Structure:
            "xaxis"
                -- "major"
                    -- "locator"
                    -- "formatter"
                -- "minor"
                    -- "locator"
                    -- "formatter"
            "yaxis"
                -- "major"
                    -- "locator"
                    -- "formatter"
                -- "minor"
                    -- "locator"
                    -- "formatter"

            EXAMPLE
            -------
            tickers = {
                "yaxis": {
                    "major": {
                        "locator": mticker.AutoLocator(),
                        "formatter": mticker.LogFormatter(labelOnlyBase=False, minor_thresholds=(np.inf, np.inf))
                    }
                }
            }

        options : list, optional
            "grid": display a grid behind the plot.

        legend_options : dict, optional
            Parameters passed to the legend.
        """
        self.xs = np.array([]) if xs is None else xs
        self.ys = [] if ys is None else ys
        self.zs = [] if zs is None else zs
        self.dim = 2 if zs is None else 3  # Dimensionality of the plot

        # When only 1 line is specified and the figure is 2D,
        # wrap ys in a list if it's not a list of ndarrays.
        if self.dim == 2 and (type(self.ys) != list or type(self.ys[0]) != np.ndarray):
            self.ys = [self.ys]

        # When only 1 line is specified and the figure is 3D,
        # wrap zs in a list if it's not a list of ndarrays.
        if self.dim == 3 and (type(self.zs) != list or type(self.zs[0]) != np.ndarray):
            self.zs = [self.zs]

        self.nlines = len(self.ys) if self.dim == 2 else len(self.zs)  # number of lines

        self.line_options = [] if line_options is None else line_options
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.figsize = figsize
        self.plot_type = plot_type
        self.fit_data = fit_data
        self.tickers = {} if tickers is None else tickers
        self.options = [] if options is None else options
        self.legend_options = {} if legend_options is None else legend_options

        self.colorbar_params = colorbar_params

        # Remove any kwargs that have None as their value
        self.kwargs = dict()
        for k, v in kwargs.items():
            if v is not None:
                self.kwargs[k] = v

        self._fill_line_options()

    def _fill_line_options(self):
        # Fill up line_options if it's not fully specified
        num_options = len(self.line_options)
        __line_options = [{} for line in range(self.nlines)]

        # Specified in line_options, use it
        for line_idx in range(num_options):
            __line_options[line_idx] = self.line_options[line_idx]

        self.line_options = __line_options

    # TODO: move plot functions inside the class
