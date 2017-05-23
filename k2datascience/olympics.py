#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Olympics Medal Module

.. moduleauthor:: Timothy Helton <timothy.j.helton@gmail.com>
"""
import datetime as dt
import logging
import os.path as osp

from dateutil import relativedelta
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns


log_format = ('%(asctime)s  %(levelname)8s  -> %(name)s <- '
              '(line: %(lineno)d) %(message)s\n')
date_format = '%m/%d/%Y %I:%M:%S'
logging.basicConfig(format=log_format, datefmt=date_format,
                    level=logging.INFO)

current_dir = osp.dirname(osp.realpath(__file__))
data_dir = osp.realpath(osp.join(current_dir, '..', 'data', 'olympics'))
data_files = (
    'athletes.csv',
    'countries.csv',
)
data_paths = {osp.splitext(x)[0]: osp.join(data_dir, x) for x in data_files}


class Medals:
    """Attributes and methods related to the 2016 Olympic Medals.
    
    :Attributes:
    
    
    """
    def __init__(self, data_sources=data_paths):
        self.athletes_columns = (
            'id',
            'name',
            'country',
            'sex',
            'dob',
            'height',
            'weight',
            'sport',
            'gold',
            'silver',
            'bronze',
        )
        self.athletes_dtypes = (
            int,
            str,
            str,
            str,
            str,
            float,
            float,
            str,
            int,
            int,
            int,
        )
        self.countries_columns = (
            'country',
            'code',
        )
        self.countries_dtypes = (
            str,
            str,
        )
        self.data_sources = data_sources
        self.athletes = pd.read_csv(
            self.data_sources['athletes'],
            dtype={x: y for x, y in zip(self.athletes_columns,
                                        self.athletes_dtypes)},
            header=0,
            names=self.athletes_columns,
            parse_dates=[4],
        )
        self.countries = pd.read_csv(
            self.data_sources['countries'],
            dtype={x: y for x, y in zip(self.countries_columns,
                                        self.countries_dtypes)},
            header=0,
            names=self.countries_columns,
        )

        self._ages = None
        self.start_date = dt.date(2016, 8, 5)

        # plot attributes
        self.label_size = 14
        self.title_size = 18
        self.sup_title_size = 24

        self._country_medals = None

    @property
    def ages(self):
        self.calc_ages()
        return self._ages

    @property
    def country_medals(self):
        self.get_country_medals()
        return self._country_medals

    def __repr__(self):
        return (f'Medals('
                f'data_files={self.data_sources})')

    def calc_age_means(self):
        """Determine the average age of male and female athletes.
        
        :returns: average age of male and female athletes
        :rtype: DataFrame
        """
        self.calc_ages()
        age_avg = (self.ages
                   .groupby('sex')['years',
                                   'months',
                                   'days']
                   .mean())
        return age_avg.applymap(np.floor)

    def calc_ages(self):
        """Determine the age of athletes."""
        ages = self.athletes.loc[:, ['id', 'sex', 'dob']].dropna()
        ages['age'] = (ages.dob.map(
            lambda x: relativedelta.relativedelta(self.start_date, x))
        )
        years = ages.age.map(lambda x: x.years)
        months = ages.age.map(lambda x: x.months)
        days = ages.age.map(lambda x: x.days)
        self._ages = pd.concat([ages, years, months, days], axis=1)
        self._ages.columns = ['id', 'sex', 'dob', 'age', 'years', 'months',
                              'days']

    def common_full_birthday(self):
        """Find most common birthday for the athletes."""
        birthday = self.athletes.dob
        birthday_count = birthday.value_counts()
        top_birthdays = birthday_count[birthday_count == birthday_count.max()]
        top_birthday_list = '\n'.join([str(x.date())
                                       for x in top_birthdays.index])

        print('Most Common Athlete Birthdays\n')
        print(f'{top_birthdays.values[0]} Athletes were born on the '
              f'following dates:\n{top_birthday_list}')

    def common_month_day_birthday(self):
        """Find most common birthday month and day for the athletes."""
        birthday = self.athletes.dob.to_frame()
        birthday['month'] = birthday.dob.map(lambda x: x.month)
        birthday['day'] = birthday.dob.map(lambda x: x.day)
        common_bday = (birthday.loc[:, ['month', 'day']]
                       .mean()
                       .map(np.floor)
                       .to_frame())
        common_bday.columns = ['Most Common Birthday']
        return common_bday

    def country_medals_plot(self, total_medals=100, save=False):
        """Plot the country_medals.
        
        :param int total_medals: display countries with total medals greater \
            than given argument
        :param bool save: if True the plot will be saved to disk
        """
        if self._country_medals is None:
            self.get_country_medals()

        font = {
                'size': self.label_size,
        }

        plt.rc('font', **font)

        fig = plt.figure('Domestic Gross Sales vs Rating Plot',
                         figsize=(10, 10), facecolor='white',
                         edgecolor='black')
        rows, cols = (2, 1)
        ax0 = plt.subplot2grid((rows, cols), (0, 0))
        ax1 = plt.subplot2grid((rows, cols), (1, 0))

        mask = self._country_medals.query(f'total > {total_medals}')

        (mask.total
         .plot(kind='pie', autopct='%i%%',
               explode=[0.05] * mask.shape[0],
               pctdistance=0.80, legend=None, shadow=True, ax=ax0))
        ax0.set_aspect('equal')
        ax0.set_ylabel('')

        (mask
         .plot(kind='bar', alpha=0.5, edgecolor='black', ax=ax1)
         )
        ax1.set_xlabel('Country', fontsize=self.label_size)
        ax1.set_ylabel('Total Medal Count', fontsize=self.label_size)
        ax1.set_xticklabels(ax1.xaxis.get_majorticklabels(), rotation=45)

        plt.suptitle('2016 Olympic Medal Count by Country',
                     fontsize=self.sup_title_size, y=1.05)
        plt.tight_layout()

        if save:
            plt.savefig('country_medals.png')
        else:
            plt.show()

    def get_country_medals(self):
        """Find medals based on country."""
        self._country_medals = (self.athletes
                                .groupby('country')['gold', 'silver', 'bronze']
                                .agg('sum'))
        self._country_medals['total'] = self._country_medals.sum(axis=1)
