#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Movies Module

.. moduleauthor:: Timothy Helton <timothy.j.helton@gmail.com>
"""
import logging
import os
import os.path as osp

from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
import numpy as np
import seaborn as sns


log_format = ('%(asctime)s  %(levelname)8s  -> %(name)s <- '
              '(line: %(lineno)d) %(message)s\n')
date_format = '%m/%d/%Y %I:%M:%S'
logging.basicConfig(format=log_format, datefmt=date_format,
                    level=logging.INFO)

current_dir = osp.dirname(osp.realpath(__file__))
data_dir = osp.realpath(osp.join(current_dir, '..', 'data', 'box_office_mojo'))
movie_data = osp.join(data_dir, 'movies_2013-2016.csv')


class BoxOffice:
    """Attributes and methods related to box office movie data.
    
    :Attributes:
    
    """
    def __init__(self, data_file=movie_data):
        self.columns = (
            'title',
            'budget',
            'domestic_gross',
            'director',
            'rating',
            'runtime',
            'release_date',
        )
        self.data_types = (
            str,
            float,
            int,
            str,
            str,
            int,
            str,
        )
        self.data_file = data_file
        self.data = pd.read_csv(
            self.data_file,
            dtype={x: y for x, y in zip(self.columns, self.data_types)},
            header=0,
            names=self.columns,
            parse_dates=[6],
        )

        # plot attributes
        self.millions_formatter = FuncFormatter(
            lambda x, position: f'${x * 1e-6}M')
        self.numerical_titles = (
            'Budget',
            'Domestic Gross Sales',
            'Run Time',
        )
        self.label_size = 14
        self.title_size = 18
        self.sup_title_size = 24

    def __repr__(self):
        return(f'BoxOffice('
               f'data_file={self.data_file})')

    def director_performance(self):
        """Rate the directors on domestic gross sales and quantity of hits."""

        dirs = self.data.groupby('director')['domestic_gross'].agg(['sum',
                                                                    'count'])
        dirs.columns = ['domestic_gross', 'qty']
        return dirs.sort_values(by='domestic_gross', ascending=False)

    def distribution_plot(self, save=False):
        """Box plot of numerical data types.
        
        :param bool save: if True the plot will be saved to disk
        """
        fig = plt.figure('Overview Box Plot', figsize=(10, 15),
                         facecolor='white', edgecolor='black')
        rows, cols = (3, 2)
        ax0 = plt.subplot2grid((rows, cols), (0, 0))
        ax1 = plt.subplot2grid((rows, cols), (1, 0))
        ax2 = plt.subplot2grid((rows, cols), (2, 0))
        ax3 = plt.subplot2grid((rows, cols), (0, 1), sharey=ax0)
        ax4 = plt.subplot2grid((rows, cols), (1, 1), sharey=ax1)
        ax5 = plt.subplot2grid((rows, cols), (2, 1), sharey=ax2)

        axes = (
            (ax0, ax3),
            (ax1, ax4),
            (ax2, ax5),
        )
        num_data = (
            'budget',
            'domestic_gross',
            'runtime',
        )
        y_format = (
            True,
            True,
            False,
        )
        y_labels = (
            'US Dollars',
            'US Dollars',
            'minutes',
        )
        y_limits = (
            (-5e7, 3e8),
            (-1e8, 1.02e9),
            (60, 200),
        )
        iterate = zip(
            axes,
            num_data,
            self.numerical_titles,
            y_format,
            y_labels,
            y_limits,
        )

        for ax, dat, title, y_format, y_lab, y_lim in iterate:
            self.data[dat].plot(kind='box', label='', ylim=y_lim, ax=ax[0])
            sns.violinplot(x=dat, data=self.data, inner='quartile',
                           orient='v', ax=ax[1])
            if y_format:
                ax[0].yaxis.set_major_formatter(self.millions_formatter)
            ax[0].set_title(title, fontsize=self.title_size)
            ax[1].set_title(title, fontsize=self.title_size)
            ax[0].set_ylabel(f'{y_lab}', fontsize=self.label_size)

        plt.suptitle('Movie Overview Distributions',
                     fontsize=self.sup_title_size, y=1.03)
        plt.tight_layout()

        if save:
            plt.savefig('distribution_plot.png')
        else:
            plt.show()

    def domestic_gross_vs_release_date_plot(self, save=False):
        """Domestic Gross Sales vs Release Date plot.
        
        :param bool save: if True the plot will be saved to disk
        """
        fig = plt.figure('Domestic Gross Sales vs Release Date Plot',
                         figsize=(10, 10), facecolor='white',
                         edgecolor='black')
        rows, cols = (2, 1)
        ax0 = plt.subplot2grid((rows, cols), (0, 0))
        ax1 = plt.subplot2grid((rows, cols), (1, 0), sharex=ax0)

        axes = (
            ax0,
            ax1,
        )
        style = (
            'd',
            '-',
        )

        for ax, sty in zip(axes, style):
            self.data.plot(label='', legend=None, style=sty, x='release_date',
                           y='domestic_gross', ax=ax)
            ax.yaxis.set_major_formatter(self.millions_formatter)
            ax.set_xlabel('Release Date', fontsize=self.label_size)
            ax.set_ylabel('Domestic Gross Sales\n(US Dollars)',
                          fontsize=self.label_size)

        plt.suptitle('Domestic Gross Sales vs Release Date',
                     fontsize=self.sup_title_size, y=0.92)
        fig.autofmt_xdate()

        if save:
            plt.savefig('domestic_gross_vs_release_date_plot.png')
        else:
            plt.show()

    def domestic_gross_vs_runtime_plot(self, save=False):
        """Domestic Gross Sales vs Run Time plot.

        :param bool save: if True the plot will be saved to disk
        """
        fig = plt.figure('Domestic Gross Sales vs Run Time', figsize=(10, 10),
                         facecolor='white', edgecolor='black')
        rows, cols = (3, 1)
        ax0 = plt.subplot2grid((rows, cols), (0, 0))
        ax1 = plt.subplot2grid((rows, cols), (1, 0), sharex=ax0)

        axes = (
            ax0,
            ax1,
        )
        style = (
            'd',
            '-',
        )

        for ax, sty in zip(axes, style):
            self.data.plot(label='', legend=None, style=sty, x='runtime',
                           y='domestic_gross', ax=ax)
            ax.yaxis.set_major_formatter(self.millions_formatter)
            ax.set_xlabel('Run Time\n(minutes)', fontsize=self.label_size)
            ax.set_ylabel('Domestic Gross Sales\n(US Dollars)',
                          fontsize=self.label_size)

        plt.suptitle('Domestic Gross Sales vs Run Time',
                     fontsize=self.sup_title_size, y=1.03)
        plt.tight_layout()

        if save:
            plt.savefig('domestic_gross_vs_runtime_plot.png')
        else:
            plt.show()

    def kde_plot(self, n_bins=50, save=False):
        """Kernel Density Estimate plot of numerical data types.
        
        :param int n_bins: number of histogram bins
        :param bool save: if True the plot will be saved to disk
        """
        fig = plt.figure('KDE Plot', figsize=(10, 15),
                         facecolor='white', edgecolor='black')
        rows, cols = (3, 1)
        ax0 = plt.subplot2grid((rows, cols), (0, 0))
        ax1 = plt.subplot2grid((rows, cols), (1, 0))
        ax2 = plt.subplot2grid((rows, cols), (2, 0))

        axes = (
            ax0,
            ax1,
            ax2,
        )
        num_data = (
            self.data.budget[self.data.budget.notnull()],
            self.data.domestic_gross[self.data.domestic_gross.notnull()],
            self.data.runtime[self.data.runtime.notnull()],
        )

        for ax, dat, title in zip(axes, num_data, self.numerical_titles):
            sns.distplot(dat, bins=n_bins, ax=ax,
                         hist_kws={'alpha': 0.5, 'edgecolor': 'black'},
                         kde_kws={'color': 'darkblue', 'label': 'KDE'})
            ax.set_xlabel(title, fontsize=self.label_size)
            ax.set_ylabel('Density', fontsize=self.label_size)

        plt.suptitle('Kernel Density Estimate Plots',
                     fontsize=self.sup_title_size, y=1.03)
        plt.tight_layout()

        if save:
            plt.savefig('kde_plot.png')
        else:
            plt.show()

    def rating_plot(self, save=False):
        """Plot movie rating with respect to run time and domestic gross.
        
        :param bool save: if True the plot will be saved to disk
        """
        fig = plt.figure('Rating Plot', figsize=(10, 10),
                         facecolor='white', edgecolor='black')
        rows, cols = (2, 2)
        ax0 = plt.subplot2grid((rows, cols), (0, 0))
        ax1 = plt.subplot2grid((rows, cols), (0, 1))
        ax2 = plt.subplot2grid((rows, cols), (1, 0))
        ax3 = plt.subplot2grid((rows, cols), (1, 1))

        rating = self.data.groupby(['rating'])['runtime',
                                               'domestic_gross'].mean()
        axes_bar = (
            ax2,
            ax3,
        )
        axes_pie = (
            ax0,
            ax1,
        )
        column = (
            'runtime',
            'domestic_gross',
        )
        titles = (
            'Run Time',
            'Domestic Gross Sales',
        )
        y_format = (
            False,
            True,
        )
        y_labels = (
            'Minutes',
            'US Dollars',
        )

        for ax, dat, title in zip(axes_pie, column, titles):
            rating[dat].plot(kind='pie', autopct='%i%%',
                             explode=[0.05] * rating[dat].shape[0],
                             pctdistance=0.88, legend=None, shadow=True, ax=ax)
            ax.set_title(title, fontsize=self.title_size, y=0.97)
            ax.set_ylabel('')

        for ax, dat, title, y_for, y_lab in zip(axes_bar, column, titles,
                                                y_format, y_labels):
            rating[dat].plot(kind='bar', alpha=0.5, ax=ax)
            ax.set_title(title, fontsize=self.title_size)
            ax.set_xlabel('Movie Rating', fontsize=self.label_size)
            ax.set_ylabel(y_lab, fontsize=self.label_size)
            if y_for:
                ax.yaxis.set_major_formatter(self.millions_formatter)

        plt.suptitle('Movie Rating Plots', fontsize=self.sup_title_size,
                     y=1.03)
        plt.tight_layout()

        if save:
            plt.savefig('rating_plot.png')
        else:
            plt.show()

        return rating
