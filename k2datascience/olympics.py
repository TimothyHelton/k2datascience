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
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
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
    
    - **athletes**: *DataFrame* athletes data
    - **athletes_columns**: *tuple* column names for athletes dataset
    - **athletes_dtypes**: *tuple* data types for athletes dataset
    - **countries**: *DataFrame* countries data
    - **countries_columns**: *tuple* column names for countries dataset
    - **countries_dtypes**: *tuple* data types for countries dataset
    - **data_sources**: *list* paths to source data files
    - **heights**: *DataFrame* athletes heights and sex
    - **label_size**: *int* plot label font size
    - **start_date**: *Date* date for 2016 Olympics opening ceremony
    - **sup_title_size**: *int* plot super title font size
    - **title_size**: *int* plot title font size 
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
        self.heights = self.athletes.loc[:, ['sex', 'height']].set_index('sex')
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
        return np.floor(age_avg)

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
        """Find most common birthday month and day for the athletes.
        
        :returns: athlete's most common birthday month and day
        :rtype: DataFrame
        """
        birthday = self.athletes.dob.to_frame()
        birthday['month'] = birthday.dob.map(lambda x: x.month)
        birthday['day'] = birthday.dob.map(lambda x: x.day)
        common_bday = (birthday.loc[:, ['month', 'day']]
                       .mean()
                       .to_frame())
        common_bday.columns = ['Most Common Birthday']
        return np.floor(common_bday)

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

        plt.tight_layout()
        plt.suptitle('2016 Olympic Medal Count by Country',
                     fontsize=self.sup_title_size, y=1.05)

        if save:
            plt.savefig('country_medals.png', bbox_inches='tight',
                        bbox_extra_artists=[self.sup_title_size])
        else:
            plt.show()

    def get_country_medals(self):
        """Find medals based on country."""
        self._country_medals = (self.athletes
                                .groupby('country')['gold', 'silver', 'bronze']
                                .agg('sum'))
        self._country_medals['total'] = self._country_medals.sum(axis=1)

    def height_boxplot(self, save=False):
        """Generate box plot of male and female height values.

        :param bool save: if True the plot will be saved to disk
        """
        height_by_sex = self.heights.reset_index()

        fig = plt.figure('Height Boxplot',
                         figsize=(8, 5), facecolor='white',
                         edgecolor='black')
        rows, cols = (1, 2)
        ax0 = plt.subplot2grid((rows, cols), (0, 0))
        ax1 = plt.subplot2grid((rows, cols), (0, 1), sharey=ax0)

        sns.boxplot(x='sex', y='height', data=height_by_sex, width=0.3, ax=ax0)
        sns.violinplot(x='sex', y='height', data=height_by_sex, cut=0,
                       inner='quartile', ax=ax1)

        for ax in (ax0, ax1):
            ax.set_xlabel('Sex', fontsize=self.label_size)
            ax.set_ylabel('Height (m)', fontsize=self.label_size)

        plt.tight_layout()
        plt.suptitle("2016 Olympic Athlete's Height",
                     fontsize=self.sup_title_size, y=1.05)

        if save:
            plt.savefig('height_box.png', bbox_inches='tight',
                        bbox_extra_artists=[self.sup_title_size])
        else:
            plt.show()

    def height_histograms(self, save=False):
        """Generate histograms of male and female height values.
        
        :param bool save: if True the plot will be saved to disk
        """
        male = self.heights.loc['male'].dropna()
        female = self.heights.loc['female'].dropna()

        bins = 50
        fig = plt.figure('Height Histograms',
                         figsize=(10, 10), facecolor='white',
                         edgecolor='black')
        rows, cols = (3, 1)
        ax0 = plt.subplot2grid((rows, cols), (0, 0))
        ax1 = plt.subplot2grid((rows, cols), (1, 0), sharex=ax0)
        ax2 = plt.subplot2grid((rows, cols), (2, 0), sharex=ax0)

        color = (
            (0.29803921568627451, 0.44705882352941179, 0.69019607843137254),
            (0.33333333333333331, 0.6588235294117647, 0.40784313725490196),
        )

        # combined plot
        male.plot(kind='hist', alpha=0.5, bins=bins, color=color[0],
                  edgecolor='black', ax=ax0)
        female.plot(kind='hist', alpha=0.5, bins=bins, color=color[1],
                    edgecolor='black', ax=ax0)

        ax0.legend(['Male', 'Female'])
        ax0.set_title("Male and Female Athlete's Height",
                      fontsize=self.title_size)
        ax0.set_ylabel('Frequency', fontsize=self.label_size)

        # male KDE
        sns.distplot(male, ax=ax1, bins=bins, color=color[0],
                     hist_kws={'alpha': 0.5, 'edgecolor': 'black'},
                     kde_kws={'color': 'darkblue', 'label': 'KDE'})
        ax1.axvline(male.height.mean(), color='crimson', label='Mean',
                    linestyle='--')
        ax1.axvline(male.height.median(), color='black', label='Median',
                    linestyle='-.')
        ax1.legend()
        ax1.set_title("Male Athlete's Height", fontsize=self.title_size)
        ax1.set_ylabel('Density', fontsize=self.label_size)

        # female KDE
        sns.distplot(female, ax=ax2, bins=bins, color=color[1],
                     hist_kws={'alpha': 0.5, 'edgecolor': 'black'},
                     kde_kws={'color': 'darkblue', 'label': 'KDE'})
        ax2.axvline(female.height.mean(), color='crimson', label='Mean',
                    linestyle='--')
        ax2.axvline(female.height.median(), color='black', label='Median',
                    linestyle='-.')
        ax2.legend()
        ax2.set_title("Female Athlete's Height", fontsize=self.title_size)
        ax2.set_xlabel('Height (m)', fontsize=self.label_size)
        ax2.set_ylabel('Density', fontsize=self.label_size)

        plt.tight_layout()
        plt.suptitle("2016 Olympic Athlete's Height",
                     fontsize=self.sup_title_size, y=1.05)

        if save:
            plt.savefig('height_hist.png', bbox_inches='tight',
                        bbox_extra_artists=[self.sup_title_size])
        else:
            plt.show()

    def height_sport(self, save=False):
        """Compare height vs sport with respect to sex.
        
        :param bool save: if True the plot will be saved to disk
        """
        fig = plt.figure('Height vs Sport',
                         figsize=(15, 7), facecolor='white',
                         edgecolor='black')
        rows, cols = (1, 1)
        ax0 = plt.subplot2grid((rows, cols), (0, 0))

        sns.violinplot(x='sport', y='height', hue='sex',
                       data=(self.athletes
                             .sort_values(by=['sport', 'sex'],
                                          ascending=[True, False])),
                       cut=0,
                       split=True, ax=ax0)

        ax0.set_xlabel('Sport', fontsize=self.label_size)
        ax0.set_ylabel('Height (m)', fontsize=self.label_size)
        ax0.set_xticklabels(ax0.xaxis.get_majorticklabels(), rotation=80)

        plt.tight_layout()
        plt.suptitle("2016 Olympic Athlete's Height by Sport",
                     fontsize=self.sup_title_size, y=1.05)

        if save:
            plt.savefig('height_sport.png', bbox_inches='tight',
                        bbox_extra_artists=[self.sup_title_size])
        else:
            plt.show()

    def weightlifting_classes(self, save=False):
        """Determine weightlifting classes based on data.
        
        :param bool save: if True the plot will be saved to disk
        """
        weightlifters = self.athletes.query('sport == "weightlifting"'
                                            '& sex == "male"')

        iterations = 100
        n_groups = 7
        weight_cls = np.zeros((n_groups, iterations))
        x = weightlifters.query('weight < 110')[['weight', 'height']].values
        for n in range(iterations):
            kmeans = KMeans(n_clusters=n_groups).fit(x)
            kmeans.predict(x)
            weight_cls[:, n] = np.sort(kmeans.cluster_centers_, axis=0)[:, 0]

        weight_classes = weight_cls.mean(axis=1)

        fig = plt.figure('Weightlifter Scatter',
                         figsize=(10, 5), facecolor='white',
                         edgecolor='black')
        rows, cols = (1, 1)
        ax0 = plt.subplot2grid((rows, cols), (0, 0))

        weightlifters.plot(kind='scatter', x='weight', y='height', ax=ax0)

        text_box_props = {
            'boxstyle': 'round',
            'edgecolor': 'red',
            'facecolor': 'white',
        }

        for predicted_weight in weight_classes:
            ax0.axvline(predicted_weight, color='crimson', linestyle='--')
            ax0.text(predicted_weight - 2.5, 1.95,
                     f'{predicted_weight:.1f}', bbox=text_box_props,
                     fontsize=10)

        ax0.text(140, 1.95, f'Over {weight_classes[-1]:.1f}',
                 bbox=text_box_props, fontsize=10)

        ax0.set_xlabel('Weight (Kg)', fontsize=self.label_size)
        ax0.set_ylabel('Height (m)', fontsize=self.label_size)

        plt.tight_layout()
        plt.suptitle("2016 Olympic Weightlifter Height vs Weight",
                     fontsize=self.sup_title_size, y=1.05)

        if save:
            plt.savefig('Weightlifter.png', bbox_inches='tight',
                        bbox_extra_artists=[self.sup_title_size])
        else:
            plt.show()
