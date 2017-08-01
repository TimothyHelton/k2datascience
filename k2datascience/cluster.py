#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Clustering Module

.. moduleauthor:: Timothy Helton <timothy.j.helton@gmail.com>
"""
import logging
import os.path as osp

import bokeh.io as bkio
import bokeh.models as bkm
import bokeh.plotting as bkplt
from bokeh.sampledata.us_states import data as states
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as spch
import seaborn as sns
import sklearn
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances

from k2datascience.utils import ax_formatter, size, save_fig


log_format = ('%(asctime)s  %(levelname)8s  -> %(name)s <- '
              '(line: %(lineno)d) %(message)s\n')
date_format = '%m/%d/%Y %I:%M:%S'
logging.basicConfig(format=log_format, datefmt=date_format,
                    level=logging.INFO)

current_dir = osp.dirname(osp.realpath(__file__))
data_dir = osp.realpath(osp.join(current_dir, '..', 'data', 'clustering'))
arrests_data = osp.join(data_dir, 'us_arrests.csv')
genes_data = osp.join(data_dir, 'genes.csv')


class Cluster:
    """
    Attributes and methods related to general clustering of a dataset.

    :Attributes:

    - **data**: *DataFrame* data
    - **linking**: *ndarray*
    - **model**: classification model type
    - **n_components**: *int* number of principle components to find
    - **pca**: *sklearn.PCA* scikit-learn instance of PCA class
    - **std_x**: *DataFrame* standardized data
    - **var_pct**: *pd.Series* principle components variance percentage
    - **var_pct_cum**: *pd.Series* principle components cumulative variance /
        percentage
    """
    def __init__(self):
        self.clusters = None
        self.data = None
        self.data_file = None
        self.data_types = {}
        self.linkage = None
        self.n_components = 3
        self.pca = None
        self.std_x = None
        self.var_pct = None
        self.var_pct_cum = None

    def __repr__(self):
        return 'Cluster()'

    def calc_pca(self):
        """
        Calculate the principle components for the dataset.
        """
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(self.std_x)

        self.var_pct = pd.Series(self.pca.explained_variance_ratio_)
        self.var_pct_cum = self.var_pct.cumsum()

    def calc_pca_eq(self):
        """
        Calculate PCA Proportion of Variance Explained (PVE).
        :return: PVE
        :rtype: ndarray
        """
        if self.pca is None:
            self.calc_pca()

        numerator = (
            np.sum(np.square(np.matmul(self.std_x,
                                       np.transpose(self.pca.components_))),
                   axis=0)
        )
        denominator = np.sum(np.square(self.std_x))
        return numerator / denominator

    def hierarchical_cluster(self, n_clusters, criterion='maxclust',
                             method='ward', metric='euclidean'):
        """
        Plot agglomerative hierarchical clustering dendrogram.

        :param int n_clusters: number of clusters
        :param str criterion: criterion to use in forming flat clusters
        :param str method: agglomerative clustering method to be used
        :param str metric: distance metric
        """
        self.linkage = spch.linkage(self.data, method=method, metric=metric)
        self.clusters = pd.Series(spch.fcluster(self.linkage,
                                                n_clusters,
                                                criterion),
                                  index=self.data.index)


class Arrests(Cluster):
    """
    Attributes and methods related to the US Arrests dataset.

    :Attributes:
    """
    def __init__(self):
        super().__init__()
        self.data_file = arrests_data
        self.data_types = {
            'state': 'category',
            'murder': np.float,
            'assault': np.int,
            'urban_pop': np.int,
            'rape': np.float,
        }

        self.load_data()

    def __repr__(self):
        return 'Arrests()'

    def load_data(self):
        """
        Load dataset
        """
        self.data = pd.read_csv(self.data_file,
                                dtype=self.data_types,
                                header=None,
                                index_col=0,
                                names=self.data_types.keys(),
                                skiprows=1,
                                )
        self.std_x = (sklearn.preprocessing
                      .StandardScaler()
                      .fit_transform(self.data))

    def us_map_clusters(self):
        """
        Plot clusters on US map
        """
        try:
            del states['DC']
        except KeyError:
            pass

        arrests_color = (self.clusters
                         .copy()
                         .astype('category'))
        arrests_color.cat.categories = ['#787f51', '#cd5b1b', '#c19408']

        source = bkm.ColumnDataSource({
            'xs': [states[code]['lons'] for code in states],
            'ys': [states[code]['lats'] for code in states],
            'color': list(arrests_color.values),
            'label': [f'Cluster: {x}' for x in self.clusters.values],
        })

        p = bkplt.figure(title='US Arrests', toolbar_location='right',
                         plot_width=800, plot_height=600,
                         x_range=bkm.Range1d(-180, -65))

        p.patches(xs='xs', ys='ys', color='color', legend='label',
                  source=source, fill_alpha=0.8, line_color='#000000',
                  line_width=2, line_alpha=0.3)

        bkio.show(p)


class Genes(Cluster):
    """
    Attributes and methods related to the Genes dataset.

    :Attributes:
    """

    def __init__(self):
        super().__init__()
        self.data_file = genes_data

        self.load_data()

    def __repr__(self):
        return 'Genes()'

    def load_data(self):
        """
        Load dataset
        """
        self.data = pd.read_csv(self.data_file, header=None)

    def box_plot(self, save=False, title=None):
        """
        Box plot of the dataset.

        :param bool save: if True the figure will be saved
        :param str title: dataset title
        """
        plt.figure('Box Plot', figsize=(16, 8),
                   facecolor='white', edgecolor='black')
        rows, cols = (1, 1)
        ax0 = plt.subplot2grid((rows, cols), (0, 0))

        sns.boxplot(data=self.data, ax=ax0)

        ax0.set_title('Genes Dataset', fontsize=size['title'])

        save_fig('genes_box_plot', save)
