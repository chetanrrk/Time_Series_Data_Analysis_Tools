This package provides some subpackages containing classes and functions to analyze time series data.

"time_series_analysis" package contains four main classes

1) data_reader.py:
Contains DataReader class that reads in a time series data from a flat file
Data should be in multiple rows per time step data could contain 1 or more columns that evolves in time

2) data_analysis.py:
Contains DataAnalysis class that inherits reader from DataReader
Contains functions to compute mean, standard deviation, 
Contains function to plot time tracy along any dimension

3) convergence_analysis.py:
Contains function to test convergence using samples along time to test if mean vary
Can test convergence along one or all dimensions

4) auto_correlation.py:
Contains AutoCorrelation class that inherits DataReader
Computes autocorrelation along one or several dimension and reports correlation along time

=======================================================================================================

"test" package 
Contains unit test of the above classes

For example these tests could be run as follows:
python -m unittest test.test_data_reader.py

=======================================================================================================

"example" package:
contains worked out examples 

"example.py" demonstrates how to perform autocorrelation using this tool on a n*m time series data
where, "n" are samples observed in time and "m" is the dimension of the data
the code handles 1 to "n" higher dimensional data

=======================================================================================================

To install the package:

pip install Time-Series-DataAnalysis-Tool==0.1.9