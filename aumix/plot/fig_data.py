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
                 ys: list = None,
                 line_options=None,
                 title="",
                 xlabel="",
                 ylabel="",
                 figsize=(9, 7),
                 options=None,
                 **kwargs
                 ):
        """
        xs : np.ndarray, optional
            Data points in the x-axis.
            For example, np.array([0, 0.1, 0.2, 0.3, 0.4]).

        ys : list, optional
            A list of data points stored in `np.ndarray`s.
            Each item is a line in the figure.
            Each `np.ndarray` contains elements y_i that is plotted against x_i in xs.

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

        options : list, optional
            "grid": display a grid behind the plot.
        """
        self.xs = np.array([]) if xs is None else xs
        self.ys = [] if ys is None else ys
        self.line_options = [] if line_options is None else line_options
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.figsize = figsize
        self.options = [] if options is None else options

        self.__fill_line_options()

        # Wrap ys in a list if it's not a list. For example this could happen when only 1 line is specified.
        self.ys = [self.ys] if type(self.ys) != list else self.ys

    def __fill_line_options(self):
        # Fill up line_options if it's not fully specified
        num_lines = len(self.ys)
        num_options = len(self.line_options)
        __line_options = [{} for line in range(num_lines)]

        # Specified in line_options, use it
        for line_idx in range(num_options):
            __line_options[line_idx] = self.line_options[line_idx]

        self.line_options = __line_options
