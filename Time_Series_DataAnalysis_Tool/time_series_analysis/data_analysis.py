import numpy as np
from matplotlib import pyplot as plt
from .data_reader import DataReader


class DataAnalysis(DataReader):
    """
    Provides methods to apply time trace analysis of a data
    Inherits DataReader that provides Data object
    """

    def __init__(self):
        """
        Constructor method for DataAnalysis class.
        @param None
        @return None
        """

        super().__init__()  # Constructs Data object from DataReader class

    def plot_time_trace(self, col=None, out_file=None):
        """
        Plots the trace of a specific feature of the data
        @param
            column number starting from 0
            out_file: file name to save the fig
        @return
            None
        """
        if col is None:
            plt.plot(self.data, "o-")
        else:
            plt.plot(self.data[:, col],)
        plt.xlabel("Time")
        plt.ylabel("column val")
        plt.title("time trace of column {}".format(col))
        if out_file is None:
            plt.show()
        else:
            plt.savefig(out_file)
            plt.show()

    def plot_time_trace_all(self, out_file=None):
        """
        plots the time trace of all columns in a single plot
        @param out_file: file name to save the fig
        @return None
        """
        
        for i in range(self.dim):  # plotting for each dimension
            plt.plot(self.data[:, i], "o-", label="col"+str(i))
        
        plt.xlabel("Time")
        plt.ylabel("column val")
        plt.legend()
        plt.title("time trace of data")

        if out_file is None:
            plt.show()
        else:
            plt.savefig(out_file)
            plt.show()

    def calculate_mean(self, col=None):
        """
        Computes the mean for a provided column or the whole data
        @param
            col (int) : column for which mean is to be computed; if None then mean for whole data is computed
        @return
            mean (float) : mean of the desired column or the whole data
        """
        
        if col is None:
            self.mean = np.mean(self.data, axis=0)
            return self.mean
        else:
            self.mean[col] = np.mean(self.data[:, col])
            return self.mean[col]

    def calculate_stdev(self, col=None):
        """
        Computes the standard deviation for a provided column or the whole data
        @param
            col (int) : column for which stddev is to be computed; if None then stddev for whole data is computed
        @return
            stddev (float) : stddev of the desired column or the whole data
        """
        
        if col is None:
            self.stdev = np.std(self.data, axis=0)
            return self.stdev
        else:
            self.stdev[col] = np.std(self.data[:, col])
            return self.stdev[col]

    def identify_outlier(self, dev_fact=2.0, col=None):
        """
        Identifies outliers time point in the data for a given column if given or the whole data if None provided
        Any data point above 3 standard deviation from the mean is considered an outlier
        @param
            dev_fact (float): number of stddev away from mean to be considered an outlier
            col (int) : column for which outliers are to be computed; if None then outliers for whole data is computed
            
        @return
            indices of the data points in an numpy array that are outliers for a column (if provided) or the whole data
            index start from 0 to len(self.data)
        """
        
        mean_centered_data = np.abs(np.array(self.mean - self.data))  # mean centering the data first

        if col is None:  # 1D data
            tmp_data = np.argwhere(mean_centered_data > dev_fact * self.stdev)
            outlier = [int(d) for d in tmp_data]  # converting to list format for simplicity
            return np.array(outlier)
        else:
            tmp_data = mean_centered_data[:, col]
            new_data = np.argwhere(tmp_data > dev_fact * self.stdev[col])
            outlier = [int(d) for d in new_data]  # converting to list format for simplicity
            return np.array(outlier)  # returns in a numpy array format
