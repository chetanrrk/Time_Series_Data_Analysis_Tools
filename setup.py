from setuptools import setup, find_packages

with open("README.md", "r") as f:
      long_description = f.read()

setup(name='Time_Series_DataAnalysis_Tool',
      version='0.1.9.1',
      description='provides modules to read and perform analysis of a time series data',
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages=['Time_Series_DataAnalysis_Tool'],
      author='Chetan Raj Rupakheti',
      author_email='chetanrrk@gmail.com',
      python_requires='>=3',
      zip_safe=False)



