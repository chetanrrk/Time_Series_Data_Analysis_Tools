import unittest
import numpy as np
from Time_Series_DataAnalysis_Tool.time_series_analysis.convergence_analysis import ConvergenceAnalysis


class ConvergenceAnalysisTestCase(unittest.TestCase):
    def test_compute_sample_mean(self):
        """
        Tests if sample means are being computed correctly
        """

        data_obj = ConvergenceAnalysis()
        data_obj.data = np.arange(20)  # using this dummy data to test
        data_obj.calculate_sample_mean(5)
        samples = np.split(np.arange(20), 4)  # creates dummy samples to test the result from the method
        samples_means = np.array([np.mean(samples[i], axis=0) for i in range(len(samples))])
        is_equals = samples_means == data_obj.sample_means  # stores comparison (bool) in a variable

        # is_equals must contains all True bool to pass the test
        for i in range(len(is_equals)):  # testing each value at a time
            self.assertEqual(is_equals[i], True)  # must be True to pass


if __name__ == '__main__':
    unittest.main()
