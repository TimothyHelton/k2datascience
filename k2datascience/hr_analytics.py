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
from scipy import stats
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
        self.bernoulli_vars = ('accident', 'left', 'promotion_5yr')
        self.central_limit_vars = ('satisfaction', 'evaluation')
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
        self.normal_vars = ('satisfaction', 'evaluation', 'hours_avg')
        self.poisson_vars = ('projects_qty', 'service_qty')
        self.pop_tot = self.data.shape[0]
        self.p_salary = None
        self.salaries = ('low', 'medium', 'high')
        self.sup_title_size = 24
        self.title_size = 20
        self._norm_stats = None
        self._p_left_company = None
        self._p_left_and_accident = None
        self._p_work_accident = None

    @property
    def norm_stats(self):
        self.calc_gaussian_stats()
        return self._norm_stats

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

    def bernoulli_plot(self, save=False):
        """Plot PMF and CDF for Bernoulli variables.
        
        :param bool save: if True the plot will be saved to disk
        """
        ber = pd.DataFrame(list(range(self.pop_tot)), columns=['k'])
        probs = self.calc_p_bernoulli()

        for var in self.bernoulli_vars:
            dist = stats.binom(n=self.pop_tot, p=probs[var])
            ber[f'pmf_{var}'] = ber.k.apply(lambda x: dist.pmf(x))
            ber[f'cdf_{var}'] = ber.k.apply(lambda x: dist.cdf(x))

        fig = plt.figure('Bernoulli Plot', figsize=(10, 7),
                         facecolor='white', edgecolor='black')
        rows, cols = (2, 1)
        ax0 = plt.subplot2grid((rows, cols), (0, 0))
        ax1 = plt.subplot2grid((rows, cols), (1, 0))

        mask = [x for x in ber.columns if 'pmf' in x]
        ber.loc[:, mask].plot(ax=ax0)

        ax0.set_title('Probability Mass Functions', fontsize=self.title_size)
        ax0.set_xlabel('Employees')
        ax0.set_ylabel('Probability')
        ax0.legend(self.bernoulli_vars)

        mask = [x for x in ber.columns if 'cdf' in x]
        ber.loc[:, mask].plot(ax=ax1)

        ax1.set_title('Cumulative Distribution Functions',
                      fontsize=self.title_size)
        ax1.set_xlabel('Employees')
        ax1.set_ylabel('Probability')
        ax1.legend(self.bernoulli_vars)

        plt.suptitle('Bernoulli Distributions',
                     fontsize=self.sup_title_size, y=1.08)
        plt.tight_layout()

        if save:
            plt.savefig('bernoulli.png')
        else:
            plt.show()

    @staticmethod
    def bootstrap(variable, n, sets):
        """Bootstrap the median of given variable.
        
        :param pd.Series variable: variable to be bootstrapped 
        :param int n: number of samples per set
        :param int sets: number of sets
        :return: 
        """
        median_values = []
        for _ in range(sets):
            median_values.append(variable.sample(n=n, replace=True).median())
        boot = np.array(median_values)
        return {
            'boot_median': boot.mean(),
            'actual_median': variable.median(),
            'boot_std': boot.std(),
            'actual_std': variable.std(),
        }

    def box_plot(self, save=False):
        """Create box plot of quantitative fields.
        
        :param bool save: if True the plot will be saved to disk
        """
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

    def central_limit_plot(self, save=False):
        """Create a Central Limit histogram plot.
        
        :param bool save: if True the plot will be saved to disk
        """
        fig = plt.figure('Central Limit Plot', figsize=(10, 10),
                         facecolor='white', edgecolor='black')
        rows, cols = (4, 1)
        ax0 = plt.subplot2grid((rows, cols), (0, 0))
        ax1 = plt.subplot2grid((rows, cols), (1, 0))
        ax2 = plt.subplot2grid((rows, cols), (2, 0))
        ax3 = plt.subplot2grid((rows, cols), (3, 0))

        for sample_n, axes in zip((10, 100, 500, 1000), (ax0, ax1, ax2, ax3)):
            cl = self.central_limit_distributions(sample_n)
            cl.loc[:, self.central_limit_vars].plot(
                kind='hist', alpha=0.5, bins=75, edgecolor='black',
                normed=True, title=f'{sample_n} Samples', ax=axes
            )
            for var, color in zip(self.central_limit_vars, ('r', 'k')):
                rv = stats.norm(loc=cl[var].mean(), scale=cl[var].std())
                x = np.linspace(rv.ppf(0.01), rv.ppf(0.99), 100)
                axes.plot(x, rv.pdf(x), color=color, linestyle='-',
                          linewidth=2)

        plt.suptitle('Central Limit Theorem',
                     fontsize=self.sup_title_size, y=1.08)
        plt.tight_layout()

        if save:
            plt.savefig('central_limit.png')
        else:
            plt.show()

    def calc_bernoulli_variance(self):
        """Calculate Bernoulli variables variance.

        :returns: variance for each Bernoulli variable        
        :rtype: pd.DataFrame
        """
        return self.data.loc[:, self.bernoulli_vars].var()

    def central_limit_distributions(self, n):
        """Generate central limit distributions.
        
        :param int n: sample size
        :returns: satisfaction and evaluation means for 1000 distribution \
            of sample size n
        :rtype: pd.DataFrame
        """
        cl = pd.DataFrame(list(range(1000)), columns=['sample_n'])
        cl['satisfaction'] = cl.sample_n.map(
            lambda x: self.calc_satisfaction_random_sample(n))
        cl['evaluation'] = cl.sample_n.map(
            lambda x: self.calc_evaluation_random_sample(n))
        return cl

    def calc_evaluation_random_sample(self, n):
        """Calculate the evaluation mean from a sample size of n.

        :returns: evaluation mean from a sample size of n
        :rtype: float
        """
        sample = self.data.sample(n=n)
        return sample.evaluation.mean()

    def calc_gaussian_stats(self):
        """Calculate mean and variance for the Gaussian variables."""
        self._norm_stats = pd.DataFrame(index=['mean', 'variance'])
        for var in self.normal_vars:
            mean = self.data[var].mean()
            variance = self.data[var].var()
            self._norm_stats[var] = [mean, variance]

    def calc_hours_stats(self):
        """Hours worked statistics.
        
        :returns: average hours worked variance and standard deviation
        :rtype: tuple
        """
        return self.data.hours_avg.var(), self.data.hours_avg.std()

    def calc_p_bernoulli_k(self, k=3500, cumulative=False):
        """Probability of k positive outcomes for Bernoulli variables.
        
        :param int k: number of positive outcomes
        :param bool cumulative: if True calculate the cumulative probability \ 
            of k positive outcomes
        :returns: probability of k positive outcomes
        :rtype: pd.DataFrame
        """
        p_bernoulli = self.calc_p_bernoulli()
        p_k = pd.DataFrame(columns=pd.Index(data=self.bernoulli_vars,
                                            name=f'p_{k}'))
        for var in self.bernoulli_vars:
            dist = stats.binom(n=self.pop_tot, p=p_bernoulli[var])
            if cumulative:
                prob = dist.cdf(k)
            else:
                prob = dist.pmf(k)
            p_k[var] = [prob]
        return p_k

    def calc_p_bernoulli(self):
        """Probability of True for Bernoulli variables.
        
        :returns: probability of success for each Bernoulli variable
        :rtype: pd.DataFrame
        """
        return self.data.loc[:, self.bernoulli_vars].mean()

    def calc_p_hours_salary(self):
        """Probability average hours exceeds 2 std for each salary.
        
        :returns: probability employees work hours exceeds two standard \
            deviations for the given their salary.
        :rtype: pd.DataFrame
        """
        if self.p_salary is None:
            self.calc_p_salary()

        p_hours_salary = pd.DataFrame(columns=pd.Index(data=self.salaries,
                                                       name='Salaries'))
        for pay in self.salaries:
            hr_hours = self.data.hours_avg
            hr_2_std = hr_hours.mean() + 2 * hr_hours.std()

            hours_n = self.data.query(f'hours_avg > '
                                      f'{hr_2_std}').hours_avg.count()
            p_hours = hours_n / self.pop_tot

            p_salary_hours = (
                self.data.query(f'salary == "{pay}"'
                                f'& hours_avg > {hr_2_std}').hours_avg.count()
                / hours_n)

            p_hours_salary[pay] = [p_salary_hours * p_hours
                                   / self.p_salary[pay]]
            p_hours_salary.index = ['hours_over_2_std']

        return p_hours_salary

    def calc_p_left_and_accident(self):
        """Probability an employee left and experienced an accident."""
        mask = 'left == 1 & accident == 1'
        self._p_left_and_accident = (
            self.data.query(mask).left.count() / self.pop_tot
        )

    def calc_p_left_company(self):
        """Probability an employee left the company."""
        left_n = self.data.query('left == 1').left.count()
        self._p_left_company = left_n / self.pop_tot

    def calc_p_left_salary(self):
        """Probability employee left given a specific salary."""
        p_left_salary = pd.DataFrame(columns=pd.Index(data=self.salaries,
                                                      name='Salaries'))
        for pay in self.salaries:
            p_salary_left_n = self.data.query(f'salary == "{pay}"'
                                              f'& left == 1').left.count()
            p_salary_left = (p_salary_left_n
                             / self.data.query('left == 1').left.count())

            p_left_salary[pay] = [p_salary_left * self.p_left_company
                                  / self.p_salary[pay]]
        return p_left_salary

    def calc_p_salary(self):
        """Probability of having a given salary.
        
        :returns: probability of having a given salary.
        :rtype: dict
        """
        self.p_salary = {}
        for pay in self.salaries:
            prob = (self.data.query(f'salary == "{pay}"').salary.count()
                    / self.pop_tot)
            self.p_salary[pay] = prob

    def calc_p_salary_promotion(self):
        """Probability of promotion given a salary.
        
        :return: probability of receiving a promotion based on salary.
        :rtype: pd.DataFrame
        """
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

    @staticmethod
    def calc_power(dataset_1, dataset_2):
        """Statistical Power of two datasets using T distributions.
        
        .. note:: This method should work, but does not yield the same result \
            as the solution guide.
        
        :param pd.Series dataset_1: first dataset
        :param pd.Series dataset_2: second dataset
        :returns: statistical power of both datasets
        :rtype: float
        """
        df_1, med_1, scale_1 = stats.t.fit(dataset_1)
        df_2, med_2, scale_2 = stats.t.fit(dataset_2)

        dt_1 = stats.t(df=df_1, loc=dataset_1.mean(),
                       scale=dataset_1.std(ddof=1) / dataset_1.size**0.5)
        dt_2 = stats.t(df=df_2, loc=dataset_2.mean(),
                       scale=dataset_2.std(ddof=1) / dataset_2.size ** 0.5)

        if dataset_1.mean() < dataset_2.mean():
            lower = dt_1
            upper = dt_2
        else:
            lower = dt_2
            upper = dt_1

        x_bar = lower.ppf(0.95)
        return upper.sf(x_bar)

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

    def compare_confidence(self, dataset_1, name_1, dataset_2, name_2,
                           confidence):
        """Compare confidence intervals between Gaussian and T distributions.
        
        :param pd.Series dataset_1: first dataset
        :param str name_1: name of first dataset
        :param pd.Series dataset_2: second dataset
        :param str name_2: name of second dataset
        :param float confidence: confidence level
        :returns: T and Gaussian confidence intervals
        :rtype: dict
        """
        t = {}
        normal = {}

        for dataset, name in zip((dataset_1, dataset_2), (name_1, name_2)):
            t[name] = self.t_confidence(dataset, confidence)
            normal[name] = self.gaussian_confidence(dataset, confidence)

        return {'t': t, 'normal': normal}

    @staticmethod
    def gaussian_confidence(dataset, confidence=0.95):
        """Confidence interval for a Gaussian distribution at given confidence.
        
        :param pd.Series dataset: dataset to evaluate
        :param float confidence: desired confidence
        :returns: upper and lower confidence limits
        :rtype: tuple
        """
        distribution = stats.norm(
            loc=dataset.mean(), scale=dataset.std(ddof=1) / dataset.size**0.5)
        return distribution.interval(confidence)

    def gaussian_plot(self, normal_overlay=False, save=False):
        """Create histogram plots for Gaussian variables.

        :param bool normal_overlay: if True a Gaussian curve will be overlaid \
            the histogram for each variable
        :param bool save: if True the plot will be saved to disk. 
        """
        if self._norm_stats is None:
            self.calc_gaussian_stats()
        fig = plt.figure(f'Gaussian Variables', figsize=(10, 10),
                         facecolor='white', edgecolor='black')
        rows, cols = (3, 1)
        ax0 = plt.subplot2grid((rows, cols), (0, 0))
        ax1 = plt.subplot2grid((rows, cols), (1, 0))
        ax2 = plt.subplot2grid((rows, cols), (2, 0))

        for var, ax in zip(self.normal_vars, (ax0, ax1, ax2)):
            self.data[var].plot(
                kind='hist', alpha=0.5, bins=50, edgecolor='black',
                normed=True,
                title=' '.join([x.title() for x in var.split('_')]), ax=ax)
            if normal_overlay:
                rv = stats.norm(loc=self.data[var].mean(),
                                scale=self.data[var].std())
                x = np.linspace(rv.ppf(0.01), rv.ppf(0.99), 100)
                ax.plot(x, rv.pdf(x), color='indianred', linestyle='-',
                        linewidth=2)

        plt.suptitle('Gaussian Variables',
                     fontsize=self.sup_title_size, y=1.08)
        plt.tight_layout()

        if save:
            plt.savefig('gaussian.png')
        else:
            plt.show()

    def poisson_distributions(self):
        """Create distributions for the Poisson variables."""
        poisson = self.data.loc[:, ['salary', *self.poisson_vars]]
        poisson_means = poisson.groupby('salary').mean()
        poisson_means.columns = [f'{x}_mean' for x in poisson_means.columns]
        poisson_dists = poisson_means.applymap(lambda x: stats.poisson(x))
        poisson_dists.columns = [x.replace('_mean', '_poisson')
                                 for x in poisson_dists.columns]
        poisson = pd.concat([poisson_means, poisson_dists], axis=1)
        for var in self.poisson_vars:
            mask = [x for x in poisson.columns if var in x]
            poisson[f'p_{var}'] = poisson[mask].apply(
                lambda x: x[1].cdf(x[0]), axis=1)

        return poisson

    @staticmethod
    def t_confidence(dataset, confidence=0.95):
        """Confidence interval for a T distribution at given confidence.

        :param pd.Series dataset: dataset to evaluate
        :param float confidence: desired confidence
        :returns: upper and lower confidence limits
        :rtype: tuple
        """
        df, loc, scale = stats.t.fit(dataset)
        distribution = stats.t(
            df=df,
            loc=dataset.mean(),
            scale=dataset.std(ddof=1) / dataset.size**0.5)
        return distribution.interval(confidence)

    def t_test(self, dataset_1, name_1, dataset_2, name_2, dataset_3=None,
               name_3=None, save=False, independent_vars=True):
        """T-test to determine if two datasets have identical values.
        
        :param pd.Series dataset_1: first dataset
        :param name_1: first dataset name
        :param pd.Series dataset_2: second dataset
        :param name_2: second dataset name
        :param pd.Series dataset_3: third dataset
        :param name_3: third dataset name
        :param bool save: if True figure will be saved to disk
        :param bool independent_vars: if True the two datasets will be \ 
            assumed to be independent
        """
        fig = plt.figure('Bernoulli Plot', figsize=(10, 7),
                         facecolor='white', edgecolor='black')
        rows, cols = (2, 1)
        ax0 = plt.subplot2grid((rows, cols), (0, 0))

        norm_1 = stats.norm(loc=dataset_1.mean(),
                            scale=dataset_1.std() / dataset_1.size**0.5)
        x_1 = np.linspace(norm_1.ppf(0.01), norm_1.ppf(0.99), 100)
        norm_2 = stats.norm(loc=dataset_2.mean(),
                            scale=dataset_2.std() / dataset_2.size**0.5)
        x_2 = np.linspace(norm_2.ppf(0.01), norm_2.ppf(0.99), 100)

        ax0.plot(x_1, norm_1.pdf(x_1), color='indianred', label=name_1,
                 linestyle='-', linewidth=2)
        ax0.plot(x_2, norm_2.pdf(x_2), color='k', label=name_2,
                 linestyle='-', linewidth=2)

        if dataset_3 is not None:
            norm_3 = stats.norm(loc=dataset_3.mean(),
                                scale=dataset_3.std() / dataset_3.size ** 0.5)
            x_3 = np.linspace(norm_3.ppf(0.01), norm_3.ppf(0.99), 100)
            ax0.plot(x_3, norm_3.pdf(x_3), color='g', label=name_3,
                     linestyle='-', linewidth=2)

        ax0.legend()
        ax0.set_xlabel('')
        ax0.set_ylabel('')
        plt.suptitle('T-test Comparision',
                     fontsize=self.sup_title_size, y=1.08)
        plt.tight_layout()

        if save:
            plt.savefig('ttest.png')
        else:
            plt.show()

        if independent_vars:
            test = stats.ttest_ind(a=dataset_1, b=dataset_2, equal_var=False)
        else:
            test = stats.ttest_1samp(a=dataset_1, popmean=dataset_2.mean())

        return test
