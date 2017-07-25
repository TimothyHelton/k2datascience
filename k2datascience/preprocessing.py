#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Preprocessing Module

.. moduleauthor:: Timothy Helton <timothy.j.helton@gmail.com>
"""
import logging
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model as sklm
import sklearn.model_selection as skms
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from k2datascience.utils import size, save_fig


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


class Resample:
    """
    Base class for resampling data.

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
    - **x**: *DataFrame* all feature samples
    - **x_train**: *DataFrame* training features
    - **x_test**: *DataFrame* testing features
    - **y**: *DataFrame* all response samples
    - **y_train**: *Series* training response
    - **y_test**: *Series* testing response
    """
    def __init__(self):
        super().__init__()
        self.data = None
        self.data_file = None
        self.data_types = {}

        # Model Parameters
        self.coefficients = None
        self.error = None
        self.features = None
        self.intercept = None
        self.model = None
        self.predict = None
        self.response = None
        self.score = None
        self.test_size = 0.3
        self.x = None
        self.x_test = None
        self.x_train = None
        self.y = None
        self.y_test = None
        self.y_train = None

    def linear_leave_one_out(self, degree=1):
        """
        Bootstrap for Leave One Out Cross Validation of Linear Regression.

        :param int degree: the degree of the polynomial features to generate
        """
        loo = skms.LeaveOneOut()
        model = make_pipeline(PolynomialFeatures(degree),
                              sklm.LinearRegression())
        scores = skms.cross_val_score(model, self.x, self.y,
                                      scoring='neg_mean_squared_error', cv=loo)
        print(f'Leave One Out Mean Square Error: {scores.mean() * -1:.3f} '
              f'(+/- {scores.std():.3f})')

    def logistic_bootstrap(self, n):
        """
        Bootstrap for Logistic Regression.

        :param int n: number of bootstrap datasets to generate.
        """
        model = sklm.LogisticRegression()
        scores = skms.cross_val_score(model, self.x, self.y, cv=n)
        print(f'Validation Accuracy: {scores.mean() * 100:.3f}% '
              f'(+/- {scores.std() * 100:.3f}%)')
        print(f'Validation Error: {(1 - scores.mean()) * 100:.3f}%')

    def logistic_leave_one_out(self, degree=1, seed=0):
        """
        Bootstrap for Leave One Out Cross Validation of Logistic Regression.

        :param int degree: the degree of the polynomial features to generate
        :param int seed: seed value to initialize the logistic regression model
        """
        loo = skms.LeaveOneOut()
        model = make_pipeline(PolynomialFeatures(degree),
                              sklm.LogisticRegression(random_state=seed))
        scores = skms.cross_val_score(model, self.x, self.y, cv=loo)
        print(f'Leave One Out Accuracy: {scores.mean() * 100:.2f}% '
              f'(+/- {scores.std() * 100:.2f}%)')

    def logistic_regression(self, seed=0):
        """
        Perform Logistic Regression for the features and response.

        :param int seed: seed number for random state
        """
        self.model = (sklm.LogisticRegression(C=1e5, tol=1e-7,
                                              random_state=seed)
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
        """
        self.x_train, self.x_test, self.y_train, self.y_test = (
            skms.train_test_split(self.features, self.response,
                                  test_size=self.test_size)
        )
        self.x = pd.concat([self.x_train, self.x_test])
        self.y = pd.concat([self.y_train, self.y_test])


class LoanDefault(Resample):
    """
    Attributes and methods related to the loan default dataset.
    """
    def __init__(self):
        super().__init__()
        self.data_file = loan_default
        self.data_types = {
            'default': 'category',
            'student': 'category',
            'balance': np.float64,
            'income': np.float64,
        }

        self.load_data()

        self.features = self.data.loc[:, ['balance', 'income']]
        self.response = self.data.default.cat.codes

        self.validation_split()

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


class Simulated(Resample):
    """
    Attributes and methods related to a simulated dataset.
    """
    def __init__(self):
        super().__init__()
        self.random_seed = 0
        self.load_data()

        self.features = self.data.x
        self.response = self.data.y

        self.validation_split()
        self.single_feature()

    def __repr__(self):
        return 'Simulated()'

    def load_data(self):
        """
        Load the data into a DataFrame
        """
        np.random.seed(self.random_seed)
        x = np.random.randn(100)
        e = np.random.randn(100)
        y = x - 2 * x ** 2 + e
        self.data = pd.DataFrame(np.c_[x, y],
                                 columns=['x', 'y'])

    def scatter_plot(self, save=False):
        """
        Create a scatter plot of the simulated data.

        :param bool save: if True the figure will be saved
        """
        plt.figure('Simulated Scatter Plot', figsize=(10, 8),
                   facecolor='white', edgecolor='black')
        rows, cols = (1, 1)
        ax0 = plt.subplot2grid((rows, cols), (0, 0))

        self.data.plot(kind='scatter', x='x', y='y', alpha=0.5,
                       edgecolor='black', ax=ax0)

        ax0.set_title('Simulated Data Scatter Plot', fontsize=size['title'])
        ax0.set_xlabel('X', fontsize=size['label'])
        ax0.set_ylabel('Y', fontsize=size['label'])

        save_fig('simulated_scatter', save)

    def single_feature(self):
        """
        Add dimension to x attribute, since on one feature exists.
        """
        self.x = self.x.values.reshape(-1, 1)


class StockMarket(Resample):
    """
    Attributes and methods related to the stock market dataset.
    """
    def __init__(self):
        super().__init__()
        self.data_file = stock_market
        self.data_types = {
            'idx': np.int32,
            'year': np.int64,
            'lag1': np.float64,
            'lag2': np.float64,
            'lag3': np.float64,
            'lag4': np.float64,
            'lag5': np.float64,
            'volume': np.float64,
            'today': np.float64,
            'direction': 'category',
        }

        self.load_data()

        self.features = self.data.loc[:, ['lag1', 'lag2']]
        self.response = self.data.direction.cat.codes

        self.validation_split()

    def __repr__(self):
        return 'StockMarket()'

    def load_data(self):
        """
        Load the data into a DataFrame
        """
        self.data = (pd.read_csv(self.data_file,
                                 dtype=self.data_types,
                                 header=None,
                                 index_col=1,
                                 names=self.data_types.keys(),
                                 parse_dates=[1],
                                 skiprows=1,
                                 )
                     .drop('idx', axis=1))
