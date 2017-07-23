#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Preprocessing Module

.. moduleauthor:: Timothy Helton <timothy.j.helton@gmail.com>
"""
import logging
import os.path as osp

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, LeaveOneOut, KFold

from k2datascience.utils import ax_formatter, size, save_fig


log_format = ('%(asctime)s  %(levelname)8s  -> %(name)s <- '
              '(line: %(lineno)d) %(message)s\n')
date_format = '%m/%d/%Y %I:%M:%S'
logging.basicConfig(format=log_format, datefmt=date_format,
                    level=logging.INFO)

current_dir = osp.dirname(osp.realpath(__file__))
data_dir = osp.realpath(osp.join(current_dir, '..', 'data', 'preprocessing'))
boston_housing = osp.join(data_dir, 'boston_housing.csv')
loan_default = osp.join(data_dir, 'loan_default.csv')
stock_market = osp.join(data_dir, 'stock_market.csv')


def prob_bootstrap(n):
    """
    Probability sample is included in a bootstrap.

    :param int n: number of samples
    :return: probability a single sample is included in a bootstrap
    :rtype: float
    """
    return 1 - (1 - 1 / n)**n


class Auto:
    """
    Attributes and methods related to the auto dataset.

    :Attributes:

    - **data**: *DataFrame* data
    - **data_file**: *str* path to data file
    - **data_types**: *dict* data type definitions
    - **x_train**: *DataFrame* training features
    - **y_train**: *Series* training response
    - **x_test**: *DataFrame* testing features
    - **y_test**: *Series* testing response
    """
    def __init__(self):
        super().__init__()
        self.data = None
        self.data_file = auto_data
        self.data_types = {
            'mpg': np.float64,
            'cylinders': np.int32,
            'displacement': np.float64,
            'horsepower': np.int32,
            'weight': np.int32,
            'acceleration': np.float64,
            'year': np.int32,
            'origin': np.int32,
            'name': str,
        }
        self.x_test = None
        self.x_train = None
        self.y_test = None
        self.y_train = None

        self.load_data()

    def __repr__(self):
        return 'Auto()'

    def load_data(self):
        """
        Load the data into a DataFrame
        """

        def date_parse(year):
            """
            Convert year from YY to 19YY
            :param str year: year to be converted
            :return: year in 19YY format
            :rtype: datetime
            """
            return pd.datetime.strptime(f'19{year}', '%Y')

        self.data = (pd.read_csv(self.data_file,
                                 dtype=self.data_types,
                                 header=None,
                                 index_col=6,
                                 names=self.data_types.keys(),
                                 parse_dates=[6],
                                 date_parser=date_parse,
                                 skiprows=1,
                                 ))
        features = (self.data
                    .select_dtypes(exclude=['object'])
                    .drop('mpg', axis=1))
        response = self.data.mpg

        self.x_train, self.x_test, self.y_train, self.y_test = (
            train_test_split(features, response, test_size=0.3)
        )
