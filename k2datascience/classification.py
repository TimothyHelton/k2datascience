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
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
import statsmodels.api as sm
import statsmodels.formula.api as smf

from k2datascience.utils import ax_formatter, save_fig, size


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


class Weekly:
    """
    Attributes and methods related to the weekly dataset.

    :Attributes:

    - **classification** *str* classification report
    - **confusion** *DataFrame* confusion matrix
    - **data**: *DataFrame* data
    - **data_file**: *str* path to data file
    - **data_types**: *dict* data type definitions
    - **logistic_formula**: *str* logistic regression formula
    - **logistic_model**: *GLMResultsWrapper* statsmodels logistic regression \
        model
    - **predict**: *ndarray* model predicted values
    - **predicted_prob**: *ndarray* cutoff probability to make classification
    - **prediction_nom**: *ndarray* binary normalized predicted values
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

        lags = ' + '.join([f'lag{x}' for x in range(1, 6, 1)])
        self.logistic_formula = f'direction ~ {lags} + volume'
        self._logistic_model = None
        self.predict = None
        self.predicted_prob = 0.5
        self.prediction_nom = None

        self.load_data()

    @property
    def logistic_model(self):
        self.logistic_regression()
        return self._logistic_model

    def __repr__(self):
        return 'Weekly()'

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

    def logistic_regression(self):
        """
        Create logistic regression model with direction as the result.
        """
        self._logistic_model = (smf.glm(formula=self.logistic_formula,
                                        data=self.data,
                                        family=sm.families.Binomial())
                                .fit())

        self.predict = self._logistic_model.predict()
        self.prediction_nom = self.predict.copy()
        self.prediction_nom[self.prediction_nom >= self.predicted_prob] = 1
        self.prediction_nom[self.prediction_nom < self.predicted_prob] = 0

        actual = self.data.direction.cat.codes
        self.confusion = pd.DataFrame(confusion_matrix(actual,
                                                       self.prediction_nom))
        self.classification = classification_report(actual,
                                                    self.prediction_nom)
