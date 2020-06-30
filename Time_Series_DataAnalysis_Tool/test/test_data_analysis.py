import unittest
import numpy as np
from Time_Series_DataAnalysis_Tool.time_series_analysis.data_analysis import DataAnalysis


class DataAnalysisTestCase(unittest.TestCase):
    def test_file_init(self):
        """
        Testing data reader using file as before
        """

        data_obj = DataAnalysis()
        data_obj.read_data("../examples/sample_data.txt")  # reading data in the file
        self.assertEqual(data_obj.data.shape, (2000000, ))

    def test_mean(self):
        """
        Testing if mean is computed correctly
        """
        data = np.loadtxt("../examples/sample_data.txt")
        data_mean_np = np.mean(data, axis=0)
        data_obj = DataAnalysis()  # data object created
        data_obj.read_data("../time_series_analysis/x.out")  # read the data in file
        data_mean_code = data_obj.calculate_mean()  # computed mean
        self.assertEqual(data_mean_np, data_mean_code)

    def test_stddev(self):
        """
        Testing if standard deviation is computed correctly
        """
        data = np.loadtxt("../examples/sample_data.txt")
        data_std_np = np.std(data, axis=0)
        data_obj = DataAnalysis()  # data object created
        data_obj.read_data("../examples/sample_data.txt")  # read the data in file
        data_std_code = data_obj.calculate_stdev()  # computed mean
        self.assertEqual(data_std_np, data_std_code)

    def test_identify_outlier(self):
        """
        Testing if the outliers are identifies correctly
        """

        data_obj = DataAnalysis()
        test_d = np.random.randint(0, 5, (3, 4))  # dummy numpy array for testing
        data_obj.data = test_d
        data_obj.calculate_stdev()
        data_obj.calculate_mean()

        data_obj.data[0][0] = 100  # manipulating to test if outlier is identified

        self.assertEqual(data_obj.identify_outlier(col=0), np.array([0]))


if __name__ == '__main__':
    unittest.main()
