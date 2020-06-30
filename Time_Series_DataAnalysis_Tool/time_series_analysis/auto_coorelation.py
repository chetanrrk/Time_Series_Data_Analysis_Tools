import os
import sys
import pickle
import time
import numpy as np
from matplotlib import pyplot as plt
import multiprocessing as mp

from .data_analysis import DataAnalysis
from .data_reader import DataReader

"""
Created on Fri Feb 28 16:53:29 2020

@author: chetan rupakheti
"""


class AutoCorrelation(DataReader):
    """
    specialized class that performs autocorrelation analysis of a time series data
    """

    def __init__(self):
        """
        Constructor of the auto-correlation class. Inherits DataReader class and uses data object
        @param None
        @return None
        """

        super().__init__()
        self.lag_corrs = {}  # stores computed correlation values for a time lag from 0 ..N as keys

    @staticmethod
    def compute_normalized_autocorrelation(dat_t0, dat_tn, lag):
        """
        Computes the pearson's correlation
        @param
        dat_t0: numpy array, mean-centered data at time 0
        dat_tn: numpy array, mean-centered data at time n
        @return
        coor: correlation between range 0 to 1
        lag: time lags (int type)
        """

        # dot product used to include all column/feature for correlation
        norm_data_0 = (dat_t0 - np.mean(dat_t0, axis=0))
        norm_data_n = (dat_tn - np.mean(dat_tn, axis=0))

        corr = np.average([np.dot(norm_data_0[i], (np.transpose(norm_data_n[i]))) for i in range(len(dat_t0))])
        corr = corr / (np.std(dat_t0, axis=0) * np.std(dat_tn, axis=0))  # normalizing by the stddev
        return corr, lag

    @staticmethod
    def compute_autocorrelation(dat_t0, dat_tn, lag):
        """
        Computes the correlation without dividing by the standard deviation
        @param
        dat_t0: numpy array, mean centered data at time 0
        dat_tn: numpy array, mean centered data at time n
        axis: computes correlation along a provided axis/column/feature
        @return
        coor: unnormalized correlation coefficient
        lag: time lags (int type)
        """

        norm_data_0 = (dat_t0 - np.mean(dat_t0, axis=0))
        norm_data_n = (dat_tn - np.mean(dat_tn, axis=0))

        corr = np.average([np.dot(norm_data_0[i], (np.transpose(norm_data_n[i]))) for i in range(len(dat_t0))])

        return corr, lag

    def log_result(self, result):
        """
        appends the computed correlation result from a sub-process
        @param result: dictionary with lag as key and correlation as value
        @return None
        """

        self.lag_corrs[result[1]] = result[0]

    def compute_autocorrelation_with_mp(self, procs, lag_range, norm=False, axis=None):
        """
        Initializes the data used to compute the correlation
        Uses multiprocessing to parallelize the computation for each lag
        @param
        procs : number of processors to be used
        lag_range: computes correlation from start to end time lag range
        norm: to normalize the computed correlation
        axis: index of the axis can be provided to compute auto-correlation
        @return None
        """
        dt = 1  # spacing of data for example 1 time unit

        with mp.Pool(processes=procs) as pool:  # need context manager in python 3.0 >
            for i in range(lag_range[0], lag_range[1]):  # lags loop goes from 0...len(timeseries)
                lag = i
                tmax = int(len(self.data) - (lag * 1.0 / dt * 1.0))
                t0 = []
                tn = []
                for j in range(tmax):
                    if axis is not None:  # takes only data along the desired dimension
                        if self.dim > 0:  # slicing only if data is > 1D
                            t0.append(self.data[:, axis][j])
                            tn.append(self.data[:, axis][j + lag])
                        else:  # slicing not necessary if 1D data
                            t0.append(self.data[j])
                            tn.append(self.data[j + lag])
                    else:  # takes data along all dimension since None given
                        t0.append(self.data[j])
                        tn.append(self.data[j + lag])

                if len(t0) <= 1 or len(tn) <= 1:
                    continue

                if norm:
                    result = pool.apply_async(self.compute_normalized_autocorrelation,
                                              args=(np.array(t0), np.array(tn), lag), callback=self.log_result)
                    result.get()  # need to get the result else returns empty
                else:
                    result = pool.apply_async(self.compute_autocorrelation,
                                              args=(np.array(t0), np.array(tn), lag), callback=self.log_result)
                    result.get()  # need to get the result else returns empty

        pool.close()
        pool.join()

    def mean_center_data(self):
        """
        Mean centers the given data
        @param None
        @return None
        """

        data_mean = np.average(self.data, axis=0)
        self.data = self.data - data_mean

    @staticmethod
    def plot_correlations(correlations, to_save=False):
        """
        Plots the computed correlation
        @param
        correlations: computed autocorrelations
        to_save: bool to save or not the file as a pdf
        @return None
        """

        plt.plot(range(len(correlations)), correlations, "o")
        plt.xlabel("Lag")
        plt.ylabel("Correlation")
        if to_save:
            plt.savefig("autocorrelations.pdf")
            plt.show()
        else:
            plt.show()

    def run(self, procs_to_use, start_lag, end_lag, axis=None, to_normalize=False, to_dump=False,
            to_plot_correlation=False):
        """
        Main funtion to compute the autocorrelation analysis
        @param
        procs_to_use: Number of processors to be used
        start_lag: start lag time (int) from where correlation is to be computed
        end_lag: end lag time (int) till where correlation is to be computed
        axis: axis along which correlation is to be computed; if None given whole dimension are used
        to_normalize: computes the pearson correlation if True
        to_dump: pickle the computed correlation or not. created a computed_correlations.p file if True
        to_plot_correlation: to plot the correlation per time lag or not
        @return
        python dictionary containing auto-correlation for each lag
        pickle dumps the computed correlations in "computed_correlations.p"
        """

        self.mean_center_data()  # mean centers data right away
        lag_range = [start_lag, end_lag]

        start_time = time.time()  # start of auto-correlation calculation
        self.compute_autocorrelation_with_mp(procs_to_use, lag_range, to_normalize, axis)  # results at lags_corrs
        end_time = time.time()
        print("Autocorr calculation done in " + str((end_time - start_time) / 60.0) + " mins")

        correlations = []
        for lag in range(start_lag, end_lag):
            if lag in self.lag_corrs:  # python > 3 has_key() is deprecated!
                correlations.append(self.lag_corrs[lag])

        #print("correlations {}".format(correlations))

        if to_dump:
            pickle.dump(correlations, open("computed_correlations.p", "wb"))  # serializing results

        if to_plot_correlation:
            self.plot_correlations(correlations)
