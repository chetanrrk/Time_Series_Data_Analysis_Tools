import numpy as np
from matplotlib import pyplot as plt
from .data_analysis import DataAnalysis
from .data_reader import DataReader


class ConvergenceAnalysis(DataReader):
    """
    Specialized class to test the convergence of time series data
    Inherits DataAnalysis class to use its data object
    """

    def __init__(self):
        """
        Constructor method for DataAnalysis class. Inherits constructor of DataReader class
        @param None
        @return None
        """
        super().__init__()  # constructor of DataReader
        self.sample_means = []  # stores mean of the samples obtained from bootstrapping

    def calculate_sample_mean(self, sample_size=1):
        """
        Constructs samples by splitting data across time series
        @param
           sample_size: sample size to be used to compare the convergence
        @return None
        """

        if len(self.data) == 1:
            raise ValueError("The size of data has to be > 1 to test convergence!")

        data = self.data[len(self.data) % sample_size:]  # provides starting index to create samples
        # burns out first few data points to create equal sample size

        samples = np.split(data, len(data) / sample_size)  # gives equal len samples that are consecutive in time

        for sample in samples:
            """
            creates objects of DataAnalysis class for each sample
            """
            data_obj = DataAnalysis()
            data_obj.data = sample
            self.sample_means.append(data_obj.calculate_mean())

    def calculate_convergence_for_a_dim(self, col=None, threshold=0.05, to_compare=5):
        """
        Applies bootstrapping to check if convergence is attained along time progression for a given column
        or the full data
        Checks if the sample mean is converging across different time frames using pair wise comparison of sample means
        @param
           threshold (float): the mean of the samples have to be within the percent threshold of one another
           to_compare (int): number of last samples to compare for convergence
           col (int): column for which convergence is to be tested
        @return
           bool: converged or not 
        """

        max_var = 0.0
        startidx = len(self.sample_means) - to_compare
        for i in range(startidx, len(self.sample_means) - 1):
            for j in range(i, len(self.sample_means)):
                # comparing deviations in sample mean for a column
                if col is None:
                    var = np.abs(self.sample_means[i] - self.sample_means[j]) / self.sample_means[i]
                else:
                    var = np.abs(self.sample_means[i][col] - self.sample_means[j][col]) / self.sample_means[i][col]
                if var > max_var:
                    max_var = var

        if max_var > threshold:
            return False
        else:
            return True

    def calculate_convergence(self, threshold=0.05, to_compare=5):
        """
        checks convergence across all dimensions
        @param
           threshold (float): the mean of the samples have to be within the percent threshold of one another
           to_compare (int): number of last samples to compare for convergence
           col (int): column for which convergence is to be tested
        @return
           bool array: converged or not for each dimension         
        """

        is_converged = []
        for i in range(self.dim):
            is_converged.append(self.calculate_convergence_for_a_dim(0.05, 5, i))

        return is_converged
