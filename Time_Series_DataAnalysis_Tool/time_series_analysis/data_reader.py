import numpy as np


class DataReader:
    """
    Reads in the data (real numbers) that is organized in a flat file
    """
    
    def __init__(self):
        """
        Constructor method for DataReader class
        @param None
        @return None
        """

        self.data = []  # over all data
        self.dim = 0
        self.mean = np.zeros(self.dim)  # over all mean of the data
        self.stdev = np.zeros(self.dim)  # over all standard deviation of the data

    def read_data(self, file):
        """
        Reads data from a file and creates a numpy array
        @param
        file: Flat file that contains the data organized in columns and rows
        Each row corresponds to change in each dimension along time steps
        @return None
        """
        
        self.data = np.loadtxt(file)
        try:
            self.dim = self.data.shape[1]
        except IndexError:  # handling exception in case the data is 1D data
            self.dim = 1


