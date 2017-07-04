#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Principle Components Analysis Module

.. moduleauthor:: Timothy Helton <timothy.j.helton@gmail.com>
"""
import logging
import os.path as osp

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.decomposition import PCA

from k2datascience.utils import ax_formatter, save_fig, size


log_format = ('%(asctime)s  %(levelname)8s  -> %(name)s <- '
              '(line: %(lineno)d) %(message)s\n')
date_format = '%m/%d/%Y %I:%M:%S'
logging.basicConfig(format=log_format, datefmt=date_format,
                    level=logging.INFO)

current_dir = osp.dirname(osp.realpath(__file__))
data_dir = osp.realpath(osp.join(current_dir, '..', 'data', 'pca'))
gym_data = osp.join(data_dir, 'gym.csv')
movie_data = osp.join(data_dir, 'movie.csv')


def fahrenheit_to_celsius(data):
    """
    Convert Fahrenheit temperatures to Celsius.

    :param pd.Series data: temperature data in Fahrenheit
    :returns: temperature data in Celsius
    :rtype: pd.Series
    """
    values = data.values
    return pd.Series((values - 32) * 5 / 9)


class PrincipleComponents:
    """
    Attributes and methods related to Principle Components Analysis.

    :Attributes:

    - **data**: *pd.DataFrame* original data
    - **data_name**: *str* descriptive name of dataset
    - **label_column**: *str* column name in original data for labels
    - **label_max**: *float* maximum value in label column
    - **label_min**: *float* minimum value in label column
    - **n_components**: *int* number of principle components to find
    - **pca**: *sklearn.PCA* scikit-learn instance of PCA class
    - **var_pct**: *pd.Series* principle components variance percentage
    - **var_pct_cum**: *pd.Series* principle components cumulative variance /
        percentage
    - **X**: *pd.DataFrame* original data with the label column removed
    """
    def __init__(self, label_column=None):
        self.data = None
        self.data_name = None
        self.feature_columns = None
        self.label_column = label_column
        self.label_max = None
        self.label_min = None
        self.n_components = None
        self.pca = None
        self.var_pct = None
        self.var_pct_cum = None
        self.X = None
        self.y = None

    def __repr__(self):
        return f'PrincipleComponents()'

    def calc_components(self):
        """
        Calculate the data's Principle Components.
        """
        drop_cols = [x for x in self.data.columns
                     if x not in self.feature_columns]
        self.X = (self.data
                  .drop(drop_cols, axis=1)
                  .fillna(value=0, axis=1))

        X_std = (sklearn.preprocessing
                 .StandardScaler()
                 .fit_transform(self.X))
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(X_std)

        self.var_pct = pd.Series(self.pca.explained_variance_ratio_)
        self.var_pct_cum = self.var_pct.cumsum()

        pca_top_3 = PCA(n_components=3)
        self.y = (pd.DataFrame(pca_top_3.fit_transform(X_std),
                               columns=['comp_1', 'comp_2', 'comp_3'])
                  .assign(label=self.data[self.label_column])
                  .rename(columns={'label': self.label_column}))

    def calc_label_range(self):
        """
        Calculate the label column range.
        """
        self.label_min = self.data[self.label_column].min()
        self.label_max = self.data[self.label_column].max()

    def plot_componets_1_2_3(self, save=False):
        """
        Plot the 1st, 2nd and 3rd principle components.

        :param bool save: if True the figure will be saved
        """
        self.calc_label_range()

        if self.y is None:
            self.calc_components()

        with sns.axes_style("white"):
            fig = plt.figure('PCA 1, 2, 3 Scatter', figsize=(10, 8),
                             facecolor='white', edgecolor='black')
            ax = Axes3D(fig)

            sc = ax.scatter(self.y.comp_1, self.y.comp_2, self.y.comp_3,
                            c=self.y[self.label_column], cmap='gnuplot',
                            vmin=self.label_min, vmax=self.label_max)
            plt.colorbar(sc)

        ax.set_title(f'Data Colored by {self.label_column}',
                     fontsize=size['title'], y=1.02)
        ax.set_xlabel('PCA $1^{st}$ Component', fontsize=size['label'])
        ax.set_ylabel('PCA $2^{nd}$ Component', fontsize=size['label'])
        ax.set_zlabel('PCA $3^{rd}$ Component', fontsize=size['label'])

        plt.suptitle('$1^{st}$, $2^{nd}$ and $3^{rd}$ Principle Components',
                     fontsize=size['super_title'], x=0.4, y=1.04)

        save_fig('pca_1_2_3', save)

    def plot_component_2_vs_1(self, save=False):
        """
        Plot the 2nd principle component vs 1st principle component.

        :param bool save: if True the figure will be saved
        """
        self.calc_label_range()

        if self.y is None:
            self.calc_components()

        fig = plt.figure('PCA 2 vs 1 Scatter', figsize=(10, 8),
                         facecolor='white', edgecolor='black')
        ax0 = fig.add_subplot(111)

        sc = plt.scatter(x=self.y.comp_1, y=self.y.comp_2,
                         c=self.y[self.label_column], cmap='gnuplot',
                         vmin=self.label_min, vmax=self.label_max)
        plt.colorbar(sc)

        ax0.set_title(f'Data Colored by {self.label_column}',
                      fontsize=size['title'])
        ax0.set_xlabel('PCA $1^{st}$ Component', fontsize=size['label'])
        ax0.set_ylabel('PCA $2^{nd}$ Component', fontsize=size['label'])

        plt.suptitle('$2^{nd}$ Principle Component vs $1^{st}$ Principle '
                     'Component',
                     fontsize=size['super_title'])

        save_fig('pca_2_vs_1', save)

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
                            fontsize=size['label'])

        save_fig(title, save)

    def plot_variance(self, fig_size=(10, 5), save=False, title=None):
        """
        Bar plot of Principle Components variance percentage.

        :param tuple fig_size: figure size
        :param bool save: if True the figure will be saved
        :param str title: data set title
        """
        if self.pca is None:
            self.calc_components()

        plot_title = 'Principal Components Variance'
        if title:
            title = f'{title} {plot_title}'
        else:
            title = f'{self.data_name} {plot_title}'

        variance = pd.concat([self.var_pct, self.var_pct_cum], axis=1)
        ax = (variance
              .rename(index={x: x + 1 for x in range(self.var_pct.size)})
              .plot(kind='bar', alpha=0.5, edgecolor='black',
                    figsize=fig_size))

        ax.set_title(title, fontsize=size['title'])
        ax.legend(['Individual Variance', 'Cumulative Variance'],
                  fontsize=size['legend'])
        ax.set_xlabel('Components', fontsize=size['label'])
        ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=0)
        ax.set_ylabel('Percent (%)', fontsize=size['label'])
        ax.yaxis.set_major_formatter(ax_formatter['percent'])

        for patch in ax.patches:
            height = patch.get_height()
            ax.text(x=patch.get_x() + patch.get_width() / 2,
                    y=height + 0.01,
                    s=f'{height * 100:1.1f}%',
                    ha='center')

        save_fig(title, save)

    def scree_plot(self, fig_size=(10, 5), save=False, title=None):
        """
        Scree plot of Principle Components variance percentage.

        :param tuple fig_size: figure size
        :param bool save: if True the figure will be saved
        :param str title: data set title
        """
        if self.pca is None:
            self.calc_components()

        plot_title = 'Scree Plot'
        if title:
            title = f'{title} {plot_title}'
        else:
            title = f'{self.data_name} {plot_title}'

        ax = (self.var_pct
              .rename(index={x: x + 1 for x in range(self.var_pct.size)})
              .plot(kind='line', alpha=0.5, figsize=fig_size, marker='d',
                    xticks=[x + 1 for x in self.var_pct.index], ylim=(0, 1)))

        ax.set_title(title, fontsize=size['title'])
        ax.set_xlabel('Components', fontsize=size['label'])
        ax.set_ylabel('Percent (%)', fontsize=size['label'])
        ax.yaxis.set_major_formatter(ax_formatter['percent'])

        save_fig(title, save)


class Gym(PrincipleComponents):
    """
    Attributes and methods related to campus gym usage.

    :Attributes:

    - **data**: *pd.DataFrame* original data
    - **data_file**: *str* path to data file
    - **data_types**: *dict* data types for original data
    """
    def __init__(self, data_file=gym_data, **kwargs):
        super().__init__(**kwargs)
        self.data = None
        self.data_file = data_file
        self.data_types = {
            'people': np.int32,
            'time': str,
            'day_number': np.int32,
            'weekend': np.int32,
            'holiday': np.int32,
            'apparent_temp': np.float64,
            'temp': np.float64,
            'start_of_semester': np.int32,
        }

        self.load_data()

    def __repr__(self):
        return f'Gym(data_file={self.data_file})'

    def load_data(self):
        """
        Load the original dataset.
        """
        def date_parser(x):
            """
            Define date string format for parser.

            :param x: individual date entry
            :return: formatted date entry
            :rtype: str
            """
            return pd.to_timedelta(f'00:00:{x}', unit='s')

        self.data = pd.read_csv(self.data_file,
                                date_parser=date_parser,
                                dtype=self.data_types,
                                header=None,
                                names=self.data_types.keys(),
                                parse_dates=[1],
                                skiprows=1,
                                )
        day_names = {
            0: 'Monday',
            1: 'Tuesday',
            2: 'Wednesday',
            3: 'Thursday',
            4: 'Friday',
            5: 'Saturday',
            6: 'Sunday',
        }

        self.data['day_name'] = (self.data.day_number
                                 .replace(day_names)
                                 .astype('category'))

        for t in ('apparent_temp', 'temp'):
            self.data.loc[:, t] = fahrenheit_to_celsius(self.data[t])

        self.data['seconds'] = self.data.time.dt.seconds

    def plot_correlation(self, save=False):
        """
        Plot the correlation grid.

        :param bool save: if True the figure will be saved
        """
        cols = (
            'people',
            'day_number',
            'apparent_temp',
            'temp',
        )
        grid = sns.pairplot(self.data.loc[:, cols],
                            diag_kws={'alpha': 0.5, 'bins': 30,
                                      'edgecolor': 'black'},
                            plot_kws={'alpha': 0.7})

        grid.fig.suptitle('Gym Dataset Correlation',
                          fontsize=size['super_title'], y=1.03)

        for n in range(len(cols)):
            grid.axes[len(cols) - 1, n].set_xlabel(cols[n],
                                                   fontsize=size['label'])
            grid.axes[n, 0].set_ylabel(cols[n], fontsize=size['label'])

        save_fig('gym_correlations', save)


class Movies(PrincipleComponents):
    """
    Attributes and methods related to movie success prediction.

    :Attributes:

    - **data**: *pd.DataFrame* original data
    - **data_file**: *str* path to data file
    - **data_numeric** *pd.DataFrame* only numeric values from the original \
        dataset with NaNs replaced by 0
    - **data_types**: *dict* data types for original data
    """
    def __init__(self, data_file=movie_data, **kwargs):
        super().__init__(**kwargs)
        self.data = None
        self.data_file = data_file
        self.data_numeric = None
        self.data_types = {
            'color': str,
            'director': str,
            'num_critic_reviews': np.float64,
            'duration': np.float64,
            'director_facebook_likes': np.float64,
            'actor_3_facebook_likes': np.float64,
            'actor_2_name': str,
            'actor_1_facebook_likes': np.float64,
            'gross': np.float64,
            'genres': str,
            'actor_1_name': str,
            'movie_title': str,
            'num_voted_users': np.int64,
            'cast_total_facebook_likes': np.int64,
            'actor_3_name': str,
            'face_number_in_poster': np.float64,
            'plot_keywords': str,
            'movie_url': str,
            'num_user_for_reviews': np.float64,
            'language': str,
            'country': str,
            'content_rating': str,
            'budget': np.float64,
            'title_year': np.float64,
            'actor_2_facebook_likes': np.float64,
            'imdb_score': np.float64,
            'aspect_ratio': np.float64,
            'movie_facebook_likes': np.int64,
        }

        self.load_data()

    def __repr__(self):
        return f'Movies(data_file={self.data_file})'

    def load_data(self):
        """
        Load the origin dataset.
        """
        self.data = pd.read_csv(self.data_file,
                                dtype=self.data_types,
                                header=None,
                                names=self.data_types.keys(),
                                skiprows=1,
                                )
        self.data = self.data.reindex_axis(sorted(self.data_types), axis=1)

        self.data_numeric = (self.data.select_dtypes(exclude=['O'])
                             .fillna(value=0, axis=1))

    def top_correlation_joint_plots(self):
        """
        Create joint distribution plots for top 4 correlated variables.
        """
        plots = (self.data.corr()
                 .imdb_score
                 .abs()
                 .sort_values(ascending=False)
                 .head(5)
                 .index)[1:]

        for plot in plots:
            with sns.axes_style("white"):
                sns.jointplot('imdb_score', plot, data=self.data_numeric,
                              kind='kde', size=4, space=0)
