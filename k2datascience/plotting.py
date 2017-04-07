#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Plotting Module

.. moduleauthor:: Timothy Helton <timothy.j.helton@gmail.com>
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def distribution_plot(data, title, x_label, n_bins=50, title_size=24,
                      label_size=14, save=False):
    """Create subplot of data with histogram and kernel density plot.
    
    :param pd.DataFrame data: data to be plotted 
    :param str title: plot title 
    :param str x_label: x-axis label
    :param str n_bins: histogram number of bins
    :param int title_size: title font size
    :param int label_size: label font size
    :param bool save: if True the plot will be saved as .png
    """
    fig = plt.figure(f'{title} Distribution Plot', figsize=(10, 5),
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

    if save:
        plt.savefig(f'{"_".join(title.split())}.png')
    else:
        plt.show()


def pies_plot(data, title, subtitle, title_size=24, label_size=14,
              legend_loc=(1.1, 0.3), save=False):
    """Create plot with histogram and two pie plots that share categories. 
    
    :param pd.DataFrame data: data to be plotted
    :param str title: plot title
    :param tuple subtitle: subtitles for each individual pie plot
    :param int title_size: title font size
    :param int label_size: label font size
    :param tuple legend_loc: position of the pie plots shared legend 
    :param bool save: if True the plot will be saved as .png
    :return: 
    """
    fig = plt.figure(f'{title} Pies Plot', figsize=(10, 12),
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

    if save:
        plt.savefig(f'{"_".join(title.split())}.png')
    else:
        plt.show()
