import os
import sys
import numpy as np
from Time_Series_DataAnalysis_Tool.time_series_analysis.auto_coorelation import AutoCorrelation

if __name__ == "__main__":
    """
    Example on a small synthetic data to demonstrate the working of this package
    """

    # initializing here force data, need to manipulate a bit before computing
    # computes the for fO-fD, i.e., the force on the oscillator
    force = np.loadtxt("./sample_data.txt")
    force = force[:1000]  # only using a small portion of the data here for demo

    # substracting here for this data, which is not necessary in general
    force_diff = np.array([force[i] - force[i + 1] for i in range(0, len(force) - 1, 2)])

    number_of_procs = 2  # number of processors
    start_lag = 0  # start time step to compute autocorrelation
    end_lag = 50  # end time step to compute autocorrelation

    autocorr = AutoCorrelation()
    autocorr.data = force_diff  # only using a small portion of the data here for demo

    autocorr.run(number_of_procs, start_lag, end_lag, to_normalize=True, axis=None,
                 to_dump=True, to_plot_correlation=True)
