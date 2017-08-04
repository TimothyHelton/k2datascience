#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Plotting Module

.. moduleauthor:: Timothy Helton <timothy.j.helton@gmail.com>
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as spch
import seaborn as sns

from k2datascience.utils import ax_formatter, size, save_fig


def confusion_heatmap_plot(matrix, names, save=False, title=None):
    """
    Plot the confusion matrix as a heatmap.

    :param pd.DataFrame matrix: confusion matrix to be plotted
    :param list names: feature names
    :param bool save: if True the figure will be saved
    :param str title: dataset title
    """
    plot_title = 'Dataset Confusion Matrix'
    if title:
        title = f'{title} {plot_title}'
    else:
        title = plot_title

    plt.figure('Confusion Heatmap', figsize=(10, 8),
               facecolor='white', edgecolor='black')
    rows, cols = (1, 1)
    ax0 = plt.subplot2grid((rows, cols), (0, 0))

    matrix.index = names
    matrix.columns = names

    thresh = int(matrix.values.max() / 2)
    sns.heatmap(matrix,
                annot=True, cbar_kws={'orientation': 'vertical'},
                cmap='Blues',
                fmt='.0f', linewidths=5, vmin=0, vmax=thresh, ax=ax0)
    cbar = ax0.collections[0].colorbar
    cbar.set_ticks([])

    ax0.set_title(title, fontsize=size['title'])
    ax0.set_xticklabels(ax0.xaxis.get_majorticklabels(),
                        fontsize=size['label'], rotation=80)
    ax0.set_yticklabels(ax0.yaxis.get_majorticklabels(),
                        fontsize=size['label'], rotation=0)

    save_fig(title, save)


def correlation_heatmap_plot(data, save=False, title=None):
    """
    Plot the correlation values as a heatmap.

    :param pd.DataFrame data: data to be plotted
    :param bool save: if True the figure will be saved
    :param str title: dataset title
    """
    plot_title = 'Dataset Correlation'
    if title:
        title = f'{title} {plot_title}'
    else:
        title = plot_title

    plt.figure('Correlation Heatmap', figsize=(10, 8),
               facecolor='white', edgecolor='black')
    rows, cols = (1, 1)
    ax0 = plt.subplot2grid((rows, cols), (0, 0))

    sns.heatmap(data.corr(),
                annot=True, cmap='Blues', cbar_kws={'orientation': 'vertical'},
                fmt='.2f', linewidths=5, vmin=-1, vmax=1, ax=ax0)

    ax0.set_title(title, fontsize=size['title'])
    ax0.set_xticklabels(ax0.xaxis.get_majorticklabels(),
                        fontsize=size['label'], rotation=80)
    ax0.set_yticklabels(ax0.yaxis.get_majorticklabels(),
                        fontsize=size['label'], rotation=0)

    save_fig(title, save)


def correlation_pair_plot(data, save=False, title=None):
    """
    Plot the correlation grid.

    :param pd.DataFrame data: data to be plotted
    :param bool save: if True the figure will be saved
    :param str title: dataset title
    """
    include = (
        np.int32,
        np.int64,
        np.float64,
    )
    data = data.select_dtypes(include=include)

    plot_title = 'Dataset Correlation'
    if title:
        title = f'{title} {plot_title}'
    else:
        title = plot_title

    grid = sns.pairplot(data,
                        diag_kws={'alpha': 0.5, 'bins': 30,
                                  'edgecolor': 'black'},
                        plot_kws={'alpha': 0.7})

    grid.fig.suptitle(title,
                      fontsize=size['super_title'], y=1.03)

    n_cols = data.shape[1]
    for n in range(n_cols):
        grid.axes[n_cols - 1, n].set_xlabel(data.columns[n],
                                            fontsize=size['label'])
        grid.axes[n, 0].set_ylabel(data.columns[n], fontsize=size['label'])

    save_fig(title, save)


def agglomerative_dendrogram_plot(data, labels, title, method='ward',
                                  metric='euclidean', save=False):
    """
    Plot agglomerative hierarchical clustering dendrogram.

    :param ndarray data: data to be plotted
    :param ndarray labels: cluster labels (x-axis)
    :param str title: plot title
    :param str method: agglomerative clustering method to be used
    :param str metric: distance metric
    :param bool save: if True the plot will be saved as .png
    """
    plot_title = 'Agglomerative Hierarchical Clustering'
    if title:
        title = f'{title} {plot_title}'
    else:
        title = plot_title

    plt.figure('Agglomerative Dendrogram', figsize=(15, 10),
               facecolor='white', edgecolor='black')
    rows, cols = (1, 1)
    ax0 = plt.subplot2grid((rows, cols), (0, 0))

    cluster = spch.linkage(data, method=method, metric=metric)
    spch.dendrogram(cluster, labels=labels, ax=ax0)

    ax0.set_title(title, fontsize=size['title'])
    ax0.set_xticklabels(ax0.xaxis.get_majorticklabels(),
                        fontsize=size['label'])

    save_fig(title, save)


def distribution_plot(data, title, x_label, n_bins=50, title_size=24,
                      label_size=14, save=False):
    """Create subplot of data with histogram and kernel density plot.
    
    :param data: data to be plotted
    :type: pd.DataFrame, pd.Series
    :param str title: plot title 
    :param str x_label: x-axis label
    :param int n_bins: histogram number of bins
    :param int title_size: title font size
    :param int label_size: label font size
    :param bool save: if True the plot will be saved as .png
    """
    plt.figure(f'{title} Distribution Plot', figsize=(10, 5),
               facecolor='white', edgecolor='black')
    ax1 = plt.subplot2grid((1, 2), (0, 0))
    ax2 = plt.subplot2grid((1, 2), (0, 1), sharex=ax1)

    data.plot(kind='hist', alpha=0.5, ax=ax1, bins=n_bins, edgecolor='black',
              label='_nolegend_')
    ax1.axvline(data.mean(), color='crimson', label='Mean', linestyle='--')
    ax1.axvline(data.median(), color='black', label='Median', linestyle='-.')
    ax1.set_ylabel('Count', fontsize=label_size)

    sns.distplot(data, ax=ax2, bins=n_bins,
                 hist_kws={'alpha': 0.5, 'edgecolor': 'black'},
                 kde_kws={'color': 'darkblue', 'label': 'KDE'})
    ax2.set_ylabel('Density', fontsize=label_size)

    for n in (ax1, ax2):
        n.set_xlabel(x_label, fontsize=label_size)
        n.legend(fontsize=label_size)

    plt.suptitle(title, fontsize=title_size, y=1.08)
    plt.tight_layout()

    save_fig(f'{"_".join(title.split())}', save)


def pca_variance(var_pct, fig_size=(10, 5), save=False, title=None):
    """
    Bar plot of Principle Components variance percentage.

    :param Series var_pct: principle components variance percentage
    :param tuple fig_size: figure size
    :param bool save: if True the figure will be saved
    :param str title: data set title
    """
    var_pct_cum = var_pct.cumsum()

    plot_title = 'Principal Components Variance'
    if title:
        title = f'{title} {plot_title}'
    else:
        title = f'{plot_title}'

    variance = pd.concat([var_pct, var_pct_cum], axis=1)
    ax = (variance
          .rename(index={x: x + 1 for x in range(var_pct.size)})
          .plot(kind='bar', alpha=0.7, cmap='bwr', edgecolor='black',
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
                y=height - 0.05,
                s=f'{height * 100:1.1f}%',
                ha='center')

    save_fig(title, save)


def pies_plot(data, title, subtitle, title_size=24, label_size=14,
              legend_loc=(1.1, 0.3), save=False):
    """Create plot with bar plot above two pie plots that share categories. 
    
    :param pd.DataFrame data: data to be plotted
    :param str title: plot title
    :param tuple subtitle: subtitles for each individual pie plot
    :param int title_size: title font size
    :param int label_size: label font size
    :param tuple legend_loc: position of the pie plots shared legend 
    :param bool save: if True the plot will be saved as .png
    :return: 
    """
    _ = plt.figure(f'{title} Pies Plot', figsize=(10, 12),
                   facecolor='white', edgecolor='black')
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((2, 2), (1, 0))
    ax3 = plt.subplot2grid((2, 2), (1, 1))

    data.plot(kind='bar', ax=ax1)
    ax1.set_xlabel(data.index.name, fontsize=label_size)

    explode_factor = [0.02] * data.shape[0]
    axes = (ax2, ax3)
    for (n, col) in enumerate(data.columns):
        data[col].plot(kind='pie', autopct='%i%%', pctdistance=0.88,
                       explode=explode_factor, labels=None, legend=None,
                       shadow=True, title=subtitle[n], ax=axes[n])
        axes[n].title.set_size(label_size + 2)
        axes[n].title.set_position([0.5, 0.95])
        axes[n].set_ylabel('')
        axes[n].set_aspect('equal')

    ax2.legend(bbox_to_anchor=legend_loc, labels=data.index,
               loc='lower center')

    plt.suptitle(title, fontsize=title_size, y=0.92)

    save_fig(f'{"_".join(title.split())}', save)
