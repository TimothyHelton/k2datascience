#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Classification Module

.. moduleauthor:: Timothy Helton <timothy.j.helton@gmail.com>
"""
import logging
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import statsmodels.api as sm
import statsmodels.formula.api as smf

from k2datascience.utils import size, save_fig


log_format = ('%(asctime)s  %(levelname)8s  -> %(name)s <- '
              '(line: %(lineno)d) %(message)s\n')
date_format = '%m/%d/%Y %I:%M:%S'
logging.basicConfig(format=log_format, datefmt=date_format,
                    level=logging.INFO)

current_dir = osp.dirname(osp.realpath(__file__))
data_dir = osp.realpath(osp.join(current_dir, '..', 'data', 'classification'))
auto_data = osp.join(data_dir, 'auto.csv')
boston_data = osp.join(data_dir, 'boston.csv')
weekly_data = osp.join(data_dir, 'weekly.csv')


class Auto:
    """
    Attributes and methods related to the auto dataset.

    :Attributes:

    - **x_train**: *DataFrame* training features
    - **y_train**: *Series* training response
    - **x_test**: *DataFrame* testing features
    - **y_test**: *Series* testing response
    """
    def __init__(self):
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

        self.train_pct = 0.8
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

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
        binary_mpg = self.data.mpg.values.copy()
        binary_mpg_mean = binary_mpg.mean()
        binary_mpg[self.data.mpg < binary_mpg_mean] = 0
        binary_mpg[self.data.mpg >= binary_mpg_mean] = 1

        train_idx = int(self.data.shape[0] * self.train_pct)
        self.x_train = self.data[:train_idx]
        self.y_train = pd.Series(binary_mpg[:train_idx])
        self.x_test = self.data[train_idx:]
        self.y_test = pd.Series(binary_mpg[train_idx:])

        self.data['binary_mpg'] = binary_mpg

    def box_plots(self, save=False):
        """
        Box plot of MPG vs Cylinders and MPG vs Origin

        :param bool save: if True the figure will be saved
        """
        fig = plt.figure('Correlation Heatmap', figsize=(12, 5),
                         facecolor='white', edgecolor='black')
        rows, cols = (1, 2)
        ax0 = plt.subplot2grid((rows, cols), (0, 0))
        ax1 = plt.subplot2grid((rows, cols), (0, 1), sharey=ax0)

        sns.boxplot(x='cylinders', y='mpg', data=self.data, width=0.4, ax=ax0)

        ax0.set_title('MPG vs Cylinders', fontsize=size['title'])
        ax0.set_xlabel('Cylinders', fontsize=size['label'])

        sns.boxplot(x='origin', y='mpg', data=self.data, width=0.4, ax=ax1)

        ax1.set_title('MPG vs Origin', fontsize=size['title'])
        ax1.set_xlabel('Origin', fontsize=size['label'])

        for ax in (ax0, ax1):
            ax.set_ylabel('MPG', fontsize=size['label'])

        plt.suptitle('Auto Dataset', fontsize=size['super_title'], y=1.03)

        save_fig('mpg_vs_cylinders', save)


class Weekly:
    """
    Attributes and methods related to the weekly dataset.

    :Attributes:

    - **classification** *str* classification report
    - **confusion** *DataFrame* confusion matrix
    - **data**: *DataFrame* data
    - **data_file**: *str* path to data file
    - **data_types**: *dict* data type definitions
    - **knn_model**: *KNeighborsClassifier* K-Nearest Neighbors model
    - **lda_model**: *LinearDiscriminantAnalysis* LDA model
    - **logistic_formula**: *str* logistic regression formula
    - **logistic_model**: *GLMResultsWrapper* statsmodels logistic regression \
        model
    - **predict**: *ndarray* model predicted values
    - **predicted_prob**: *ndarray* cutoff probability to make classification
    - **prediction_nom**: *ndarray* binary normalized predicted values
    - **qda_model**: *QuadraticDiscriminantAnalysis* QDA model
    - **train_pct**: *float* percentage of data to be used for training
    - **x_train**: *DataFrame* training features
    - **y_train**: *Series* training response
    - **x_test**: *DataFrame* testing features
    - **y_test**: *Series* testing response
    """
    def __init__(self):
        self.classification = None
        self.confusion = None
        self.data = None
        self.data_file = weekly_data
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
            'direction': str,
        }

        self.knn_model = None

        self.lda_model = None

        lags = ' + '.join([f'lag{x}' for x in range(1, 6, 1)])
        self.logistic_formula = f'direction ~ {lags} + volume'
        self.logistic_model = None
        self.predict = None
        self.predicted_prob = 0.5
        self.prediction_nom = None

        self.qda_model = None

        self.train_pct = 0.8
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

        self.load_data()

    def __repr__(self):
        return 'Weekly()'

    def calc_prediction(self, actual, predicted):
        """
        Predict response of logistic_model based on given values.

        :param Series actual: actual response values
        :param Series predicted: predicted response values
        """
        self.confusion = pd.DataFrame(confusion_matrix(actual, predicted))
        self.classification = classification_report(actual, predicted)

    def categorize(self, data):
        """
        Convert predicted values into categories

        :param DataFrame data: data to be used to categorize predictions
        """
        self.predict = self.logistic_model.predict(data)
        self.prediction_nom = self.predict.copy()
        self.prediction_nom[self.prediction_nom >= self.predicted_prob] = 1
        self.prediction_nom[self.prediction_nom < self.predicted_prob] = 0

    def knn(self, n=1):
        """
        K-Nearest Neighbors Analysis of the data.

        :param int n: number of nearest neighbors to consider
        """
        self.knn_model = (sklearn.neighbors
                          .KNeighborsClassifier(n_neighbors=n)
                          .fit(self.x_train.drop('direction', axis=1),
                               self.y_train))
        self.predict = (self.knn_model
                        .predict(self.x_test.drop('direction', axis=1)))
        self.calc_prediction(actual=self.y_test, predicted=self.predict)

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
        self.data.direction = (self.data.direction
                               .astype('category'))

        train_idx = int(self.data.shape[0] * self.train_pct)
        self.x_train = self.data[:train_idx]
        self.y_train = self.data.direction[:train_idx].cat.codes
        self.x_test = self.data[train_idx:]
        self.y_test = self.data.direction[train_idx:].cat.codes

    def lda(self):
        """
        Linear Discriminate Analysis of the data.
        """
        self.lda_model = (LinearDiscriminantAnalysis()
                          .fit(self.x_train.drop('direction', axis=1),
                               self.y_train))
        self.predict = (self.lda_model
                        .predict(self.x_test.drop('direction', axis=1)))
        self.calc_prediction(actual=self.y_test, predicted=self.predict)

    def logistic_regression(self, data):
        """
        Create logistic regression model with direction as the result.

        :param DataFrame data: data features including response
        """
        self.logistic_model = (smf.glm(formula=self.logistic_formula,
                                       data=data,
                                       family=sm.families.Binomial())
                               .fit())

        self.categorize(data)
        y_test = data.direction.cat.codes
        self.calc_prediction(actual=y_test, predicted=self.prediction_nom)

    def qda(self):
        """
        Quadratic Discriminate Analysis of the data.
        """
        self.qda_model = (QuadraticDiscriminantAnalysis()
                          .fit(self.x_train.drop('direction', axis=1),
                               self.y_train))
        self.predict = (self.qda_model
                        .predict(self.x_test.drop('direction', axis=1)))
        self.calc_prediction(actual=self.y_test, predicted=self.predict)
