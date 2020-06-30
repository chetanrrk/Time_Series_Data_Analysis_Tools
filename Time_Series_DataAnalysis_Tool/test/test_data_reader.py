import unittest
from Time_Series_DataAnalysis_Tool.time_series_analysis.data_reader import DataReader


class DataReaderTestCase(unittest.TestCase):
    def test_read_data(self):
        """
        Testing data reader method
        """
        data_obj = DataReader()
        data_obj.read_data("../examples/sample_data.txt")  # reading data in the file
        self.assertEqual(data_obj.data.shape, (2000000, ))


if __name__ == '__main__':
    unittest.main()
