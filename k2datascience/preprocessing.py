#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Preprocessing Module

.. moduleauthor:: Timothy Helton <timothy.j.helton@gmail.com>
"""
import logging
import os.path as osp

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, LeaveOneOut, KFold

from k2datascience import plotting


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


class LoanDefault:
    """
    Attributes and methods related to the load default dataset.

    :Attributes:

    - **coefficients**: *list* model coefficients
    - **data**: *DataFrame* data
    - **data_file**: *str* path to data file
    - **data_types**: *dict* data type definitions
    - **error**: *float* model error rate
    - **features**: *DataFrame* features to be included in the model
    - **intercept**: *list* model bias
    - **model**: model type
    - **predict**: *ndarray* model predicted values
    - **response**: *Series* response of the model
    - **score**: *float* mean accuracy of the model
    - **test_size**: *float* percentage of data to be used for the test set
    - **x_train**: *DataFrame* training features
    - **y_train**: *Series* training response
    - **x_test**: *DataFrame* testing features
    - **y_test**: *Series* testing response
    """
    def __init__(self):
        super().__init__()
        self.data = None
        self.data_file = loan_default
        self.data_types = {
            'default': 'category',
            'student': 'category',
            'balance': np.float64,
            'income': np.float64,
        }

        self.load_data()

        # Model Parameters
        self.coefficients = None
        self.error = None
        self.features = self.data.loc[:, ['balance', 'income']]
        self.intercept = None
        self.model = None
        self.predict = None
        self.response = self.data.default.cat.codes
        self.score = None
        self.test_size = 0.3

        self.x_test = None
        self.x_train = None
        self.y_test = None
        self.y_train = None

    def __repr__(self):
        return 'LoadDefault()'

    def load_data(self):
        """
        Load the data into a DataFrame
        """
        self.data = (pd.read_csv(self.data_file,
                                 dtype=self.data_types,
                                 header=None,
                                 names=self.data_types.keys(),
                                 skiprows=1,
                                 ))

    def logistic_bootstrap(self, n):
        """
        Bootstrap for Logistic Regression.

        :param int n: number of bootstrap dataset to generate.
        """
        error_rates = []
        for n in range(n):
            self.validation_split()
            self.logistic_regression(seed=n)
            error_rates.append(self.error)

        error = pd.Series(error_rates)
        print(f'Error Rate Mean: {error.mean():.3f}')
        print(f'Error Rate Median: {error.median():.3f}')
        print(f'Error Rate Standard Deviation: {error.std():.3f}')

        plotting.distribution_plot(error, 'Error Rate', 'Error Rate',
                                   n_bins=20)

    def logistic_regression(self, seed=0):
        """
        Perform Logistic Regression for the features and response.

        :param int seed: seed number for random state
        """
        self.model = (LogisticRegression(C=1e5, tol=1e-7, random_state=seed)
                      .fit(self.x_train, self.y_train))
        self.intercept = self.model.intercept_
        self.coefficients = self.model.coef_
        self.predict = self.model.predict(self.x_test)
        self.score = self.model.score(self.x_test, self.y_test)
        self.error = (1 - self.score) * 100

    def logistic_summary(self, seed=0):
        """
        Print summary statistics about the Logistic Regression model.

        :param int seed: seed number for random state
        """
        self.logistic_regression(seed=seed)
        print(f'Bias: {self.intercept[0]:.3f}')
        print(f'Coefficients: {self.coefficients}')
        print(f'Model Score: {self.score:.3f}')
        print(f'Error Rate: {self.error:.2f}%')

    def validation_split(self):
        """
        Split the data based on a Validation set.
        :return:
        """
        self.x_train, self.x_test, self.y_train, self.y_test = (
            train_test_split(self.features, self.response,
                             test_size=self.test_size)
        )
