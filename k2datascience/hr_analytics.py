#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Human Resources Analytics Module

.. moduleauthor:: Timothy Helton <timothy.j.helton@gmail.com>
"""

from itertools import product
import logging
import os.path as osp

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


log_format = ('%(asctime)s  %(levelname)8s  -> %(name)s <- '
              '(line: %(lineno)d) %(message)s\n')
date_format = '%m/%d/%Y %I:%M:%S'
logging.basicConfig(format=log_format, datefmt=date_format,
                    level=logging.INFO)

current_dir = osp.dirname(osp.realpath(__file__))
data_dir = osp.realpath(osp.join(current_dir, '..', 'data', 'human_resources'))
data_file = osp.join(data_dir, 'hr.csv')


class HR:
    """Attributes and methods related to human resourse analytics.
    
    :Attributes:
    """
    def __init__(self, data=data_file):
        self.data = pd.read_csv(data)
        self.data.columns = [
            'satisfaction',
            'evaluation',
            'projects_qty',
            'hours_avg',
            'service_qty',
            'accident',
            'left',
            'promotion_5yr',
            'sales',
            'salary',
        ]
        self.label_size = 14
        self.sup_title_size = 24
        self.title_size = 20
        self._p_left_company = None
        self._p_left_and_accident = None
        self._p_work_accident = None

    @property
    def p_left_company(self):
        self.calc_p_left_company()
        return self._p_left_company

    @property
    def p_left_and_accident(self):
        self.calc_p_left_and_accident()
        return self._p_left_and_accident

    @property
    def p_work_accident(self):
        self.calc_p_work_accident()
        return self._p_work_accident

    def __repr__(self):
        return (f'HR('
                f'data={self.data})')

    def box_plot(self, save=False):
        """Create box plot of quantitative fields."""
        plt.rcParams['xtick.labelsize'] = 14
        fig = plt.figure('Distribution Plot', figsize=(10, 5),
                         facecolor='white', edgecolor='black')
        ax1 = plt.subplot2grid((1, 1), (0, 0))

        self.data.loc[:, 'satisfaction':'service_qty'].plot(
            kind='box', subplots=True, ax=ax1)

        plt.suptitle('Data Distributions',
                     fontsize=self.sup_title_size, y=1.08)
        plt.tight_layout()

        if save:
            plt.savefig('data_distributions.png')
        else:
            plt.show()

    def calc_p_left_company(self):
        """Probability an employee left the company."""
        self._p_left_company = self.data.left.mean()

    def calc_p_left_and_accident(self):
        """Probability an employee left and experienced an accident."""
        mask = 'left == 1 & accident == 1'
        self._p_left_and_accident = (
            self.data.query(mask).left.count() / self.data.size
        )

    def calc_p_work_accident(self):
        """Probability an employee experienced a work accident."""
        self._p_work_accident = self.data.accident.mean()

    def calc_percentile_satisfaction(self, percentile, left=False):
        """Return percentile of job satisfaction.
        
        :param float percentile: desired percentile
        :param bool left: if True the employees that left the company will \
            be considered; if False the current employees will be considered
        :return: value of requested percentile of job satisfaction
        :rtrype: float
        """
        if left:
            data = self.data.query('left == 1').satisfaction
        else:
            data = self.data.query('left == 0').satisfaction

        return data.quantile(percentile)

    def compare_satisfaction(self):
        """Compare job satisfaction between those who left and who stayed."""
        percentiles = (0.25, 0.50, 0.90)
        titles = ('ex_employees', 'current_employees', 'delta')
        sat_pct = pd.DataFrame(index=percentiles, columns=titles)

        for pct, left in product(percentiles, (True, False)):
            quartile = self.calc_percentile_satisfaction(pct, left)
            if left:
                sat_pct.loc[pct, titles[0]] = quartile
            else:
                sat_pct.loc[pct, titles[1]] = quartile

        sat_pct['delta'] = sat_pct[titles[1]] - sat_pct[titles[0]]

        fig = plt.figure('Job Satisfaction Comparision', figsize=(12, 6),
                         facecolor='white', edgecolor='black')
        rows, cols = (1, 3)
        ax0 = plt.subplot2grid((rows, cols), (0, 0))
        ax1 = plt.subplot2grid((rows, cols), (0, 1))
        ax2 = plt.subplot2grid((rows, cols), (0, 2), sharey=ax1)

        sns.heatmap([sat_pct.delta.as_matrix().astype(float)],
                    annot=True, cbar_kws={'orientation': 'horizontal'},
                    cmap='Blues', linewidths=5,
                    xticklabels=percentiles, yticklabels=' ',
                    vmin=0, vmax=1, ax=ax0)
        ax0.set_title('Comparison Deltas', fontsize=self.title_size)
        ax0.set_xlabel('Percentiles', fontsize=self.label_size)
        ax0.set_ylabel('Delta', fontsize=self.label_size)

        self.data.query('left == 1').satisfaction.plot(
            kind='box', ylim=[0, 1], yticks=np.linspace(0, 1, 5), ax=ax1)
        ax1.set_title('Ex-Employees', fontsize=self.title_size)

        self.data.query('left == 0').satisfaction.plot(kind='box', ax=ax2)
        ax2.set_title('Current Employees', fontsize=self.title_size)

        plt.suptitle('Job Satisfaction Comparision',
                     fontsize=self.sup_title_size, y=1.08)
        plt.tight_layout()

        return sat_pct
