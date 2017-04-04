#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Statistics Module

.. moduleauthor:: Timothy Helton <timothy.j.helton@gmail.com>
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def distribution_plot(data, title, x_label, n_bins=50, title_font=24,
                      label_font=14, save=False):
    """Create subplot of data with histogram and kernel density plot.
    
    :param pd.DataFrame data: data to be plotted 
    :param str title: plot title 
    :param str x_label: x-axis label
    :param str n_bins: histogram number of bins
    :param int title_font: title font size
    :param int label_font: label font size
    :param save: if True the plot will be saved as .png
    """
    fig = plt.figure('E-Commerce Transaction Histogram', figsize=(10, 5),
                     facecolor='white', edgecolor='black')
    ax1 = plt.subplot2grid((1, 2), (0, 0))
    ax2 = plt.subplot2grid((1, 2), (0, 1), sharex=ax1)

    data.plot(kind='hist', alpha=0.5, ax=ax1, bins=n_bins, edgecolor='black',
              label='_nolegend_')
    ax1.axvline(data.mean(), color='crimson', label='Mean', linestyle='--')
    ax1.axvline(data.median(), color='black', label='Median', linestyle='-.')
    ax1.set_ylabel('Count', fontsize=label_font)

    sns.distplot(data, ax=ax2, bins=n_bins,
                 hist_kws={'alpha': 0.5, 'edgecolor': 'black'},
                 kde_kws={'color': 'darkblue', 'label': 'KDE'})
    ax2.set_ylabel('Density', fontsize=label_font)

    for n in (ax1, ax2):
        n.set_xlabel(x_label, fontsize=label_font)
        n.legend(fontsize=label_font)

    plt.suptitle(title, fontsize=title_font, y=1.08)
    plt.tight_layout()

    if save:
        plt.savefig(f'{"_".join(title.split())}.png')
    else:
        plt.show()
