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
        self.pop_tot = self.data.shape[0]
        self.salaries = ('low', 'medium', 'high')
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

    def calc_hours_stats(self):
        """Hours worked statistics.
        
        :returns: average hours worked variance and standard deviation
        :rtype: tuple
        """
        return self.data.hours_avg.var(), self.data.hours_avg.std()

    def calc_p_hours_salary(self):
        """Probability average hours exceeds 2 std for each salary.
        
        :returns: probability employees work hours exceeds two standard \
            deviations for the given their salary.
        :rtype: pd.DataFrame
        """
        p_hours_salary = pd.DataFrame(columns=pd.Index(data=self.salaries,
                                                       name='Salaries'))
        for pay in self.salaries:
            hr_hours = self.data.hours_avg
            hr_2_std = hr_hours.mean() + 2 * hr_hours.std()

            hours_n = self.data.query(f'hours_avg > '
                                      f'{hr_2_std}').hours_avg.count()
            p_hours = hours_n / self.pop_tot

            p_salary = (self.data.query(f'salary == "{pay}"').salary.count()
                        / self.pop_tot)

            p_salary_hours = (
                self.data.query(f'salary == "{pay}"'
                                f'& hours_avg > {hr_2_std}').hours_avg.count()
                / hours_n)

            p_hours_salary[pay] = [p_salary_hours * p_hours / p_salary]
            p_hours_salary.index = ['hours_over_2_std']

        return p_hours_salary

    def calc_p_left_company(self):
        """Probability an employee left the company."""
        self._p_left_company = self.data.left.mean()

    def calc_p_left_and_accident(self):
        """Probability an employee left and experienced an accident."""
        mask = 'left == 1 & accident == 1'
        self._p_left_and_accident = (
            self.data.query(mask).left.count() / self.pop_tot
        )

    def calc_p_salary_promotion(self):
        """Probability of promotion given a salary. """
        cols = ('candidates', 'promoted', 'P_promotion')
        promotions = pd.DataFrame(index=self.salaries)
        for pay in self.salaries:
            candidates = self.data.query(f'salary == "{pay}"').salary.count()
            promoted = self.data.query(f'salary == "{pay}" '
                                       f'& promotion_5yr == 1').salary.count()
            promotions[pay] = [candidates, promoted, promoted / candidates]
        promotions.columns = cols
        return promotions

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

    def calc_satisfaction_salary(self):
        """Compare job satisfaction related to salary.
        
        :returns: satisfaction means for all available salaries
        :rtype: pd.DataFrame
        """
        return self.data.groupby('salary').satisfaction.mean()

    def calc_satisfaction_random_sample(self, n):
        """Calculate the satisfaction mean from a sample size of n.
        
        :returns: satisfaction mean from a sample size of n
        :rtype: float
        """
        sample = self.data.sample(n=n)
        return sample.satisfaction.mean()

    def compare_satisfaction_variance(self):
        """Compare variance of job satisfaction between ex and current.
        
        :returns: variance for ex and current employees
        :rtype: tuple
        """
        current = self.data.query('left == 0').satisfaction.var()
        ex = self.data.query('left == 1').satisfaction.var()
        return ex, current

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
