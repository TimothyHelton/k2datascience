#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Linear Regression Module

.. moduleauthor:: Timothy Helton <timothy.j.helton@gmail.com>
"""
import logging
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from k2datascience.utils import save_fig, size


log_format = ('%(asctime)s  %(levelname)8s  -> %(name)s <- '
              '(line: %(lineno)d) %(message)s\n')
date_format = '%m/%d/%Y %I:%M:%S'
logging.basicConfig(format=log_format, datefmt=date_format,
                    level=logging.INFO)

current_dir = osp.dirname(osp.realpath(__file__))
data_dir = osp.realpath(osp.join(current_dir, '..', 'data',
                                 'linear_regression'))
advertising_data = osp.join(data_dir, 'advertising.csv')


class LinearRegression:
    """
    Attributes and methods related to linear regression.

    :Attributes:

    - **data**: *pd.DataFrame* original data
    - **data_name**: *str* descriptive name of dataset
    - **target**: *str* name of target field
    - **X**: *pd.DataFrame* original data with target column removed
    """
    def __init__(self, data_file=None):
        self.data = None
        self.data_name = None
        self.target = None
        self._X =None

    @property
    def X(self):
        self.calc_X()
        return self._X

    def __repr__(self):
        return f'LinearRegression(data_file={self.data})'

    def calc_X(self):
        """
        Remove the target column from the original data.
        """
        self._X = self.data.drop(self.target, axis=1)

    def plot_correlation_heatmap(self, save=False, title=None):
        """
        Plot the correlation values as a heatmap.

        :param bool save: if True the figure will be saved
        :param str title: data set title
        """
        plot_title = 'Dataset Correlation'
        if title:
            title = f'{title} {plot_title}'
        else:
            title = f'{self.data_name} {plot_title}'

        fig = plt.figure('Correlation Heatmap', figsize=(10, 8),
                         facecolor='white', edgecolor='black')
        rows, cols = (1, 1)
        ax0 = plt.subplot2grid((rows, cols), (0, 0))

        sns.heatmap(self.data.corr(),
                    annot=True, cbar_kws={'orientation': 'vertical'},
                    fmt='.2f', linewidths=5, vmin=-1, vmax=1, ax=ax0)

        ax0.set_title(title, fontsize=size['title'])
        ax0.set_xticklabels(ax0.xaxis.get_majorticklabels(),
                            fontsize=size['label'], rotation=80)
        ax0.set_yticklabels(ax0.yaxis.get_majorticklabels(),
                            fontsize=size['label'], rotation=0)

        save_fig(title, save)

    def plot_correlation_joint_plots(self):
        """
        Create joint distribution plots for top 4 correlated variables.
        """
        plots = (self.data.corr()[self.target]
                 .abs()
                 .sort_values(ascending=False)
                 .head(5)
                 .index)[1:]

        for plot in plots:
            with sns.axes_style("white"):
                sns.jointplot(plot, self.target, data=self.data,
                              kind='kde', size=4, space=0)


class AdvertisingSimple(LinearRegression):
    """
    Simple linear regression methods related to the advertising dataset.

    :Attributes:
    """
    def __init__(self, data_file=advertising_data, **kwargs):
        super().__init__(**kwargs)
        self.data = None
        self.data_file = data_file
        self.data_name = 'Advertising'
        self.data_types = {
            'tv': np.float64,
            'radio': np.float64,
            'newspaper': np.float64,
            'sales': np.float64,
        }
        self.target = 'sales'

        self.load_data()

    def __repr__(self):
        return f'AdvertisingSimple()'

    def load_data(self):
        """
        Load original dataset.
        """
        self.data = pd.read_csv(self.data_file,
                                dtype=self.data_types,
                                header=None,
                                index_col=0,
                                names=self.data_types.keys(),
                                skiprows=1)


class Multiple(LinearRegression):
    """
    Attributes and methods related to multiple linear regression.

    :Attributes:
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __repr__(self):
        return f'Multiple()'