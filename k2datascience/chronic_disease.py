#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Chronic Disease Indicators Module

.. moduleauthor:: Timothy Helton <timothy.j.helton@gmail.com>
"""

from collections import OrderedDict
import logging
import os.path as osp

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


log_format = ('%(asctime)s  %(levelname)8s  -> %(name)s <- '
              '(line: %(lineno)d) %(message)s\n')
date_format = '%m/%d/%Y %I:%M:%S'
logging.basicConfig(format=log_format, datefmt=date_format,
                    level=logging.INFO)


data_file = osp.join('..', 'data', 'chronic_disease_indicators',
                     'US_Chronic_Disease_Indicators__CDI.csv')
data_dtype = OrderedDict({
    'YearStart': int,
    'YearEnd': int,
    'LocationAbbr': str,
    'LocationDesc': str,
    'DataSource': str,
    'Topic': str,
    'Question': str,
    'Response': str,
    'DataValueUnit': str,
    'DataValueTypeID': str,
    'DataValueType': str,
    'DataValue': str,
    'DataValueAlt': float,
    'DataValueFootnoteSymbol': str,
    'DatavalueFootnote': str,
    'LowConfidenceLimit': float,
    'HighConfidenceLimit': float,
    'StratificationCategory1': str,
    'Stratification1': str,
    'StratificationCategory2': str,
    'Stratification2': str,
    'StratificationCategory3': str,
    'Stratification3': str,
    'GeoLocation': str,
    'TopicID': str,
    'QuestionID': str,
    'ResponseID': str,
    'LocationID': str,
    'StratificationCategoryID1': str,
    'StratificationID1': str,
    'StratificationCategoryID2': str,
    'StratificationID2': str,
    'StratificationCategoryID3': str,
    'StratificationID3': str,
})

col_names = (
    'yr_start',
    'yr_end',
    'loc_abbr',
    'loc_desc',
    'data_src',
    'topic',
    'question',
    'response',
    'data_unit',
    'data_type_id',
    'data_type',
    'data_value',
    'data_value_alt',
    'footnote_symbol',
    'footnote',
    'low_conf',
    'high_conf',
    'strat_cat_1',
    'strat_1',
    'strat_cat_2',
    'strat_2',
    'strat_cat_3',
    'strat_3',
    'geo_loc',
    'topic_id',
    'question_id',
    'response_id',
    'loc_id',
    'strat_cat_1_id',
    'strat_1_id',
    'strat_cat_2_id',
    'strat_2_id',
    'strat_cat_3_id',
    'strat_3_id',
)


class CDI:
    """Actions related to the US Chronic Disease Indicators dataset.
    
    :param str data_path: path to data file
    
    :Attributes:
    
    **data**: *pandas.DataFrame* us chronic disease indicators data
    **data_path**: *str* path to data file 
    
    """
    def __init__(self, data_path=data_file, data_cols=col_names,
                 dtype=data_dtype):
        self.data = None
        self.data_cols = data_cols
        self.data_path = osp.realpath(data_path)
        self.diseases = None
        self.dtype = dtype

    def __repr__(self):
        return f"CDI()"

    def load_data(self):
        """Load data into a Pandas DataFrame."""
        self.data = pd.read_csv(self.data_path, dtype=self.dtype)
        self.data.columns = self.data_cols
        self.data.topic = self.data.topic.str.lower()
        logging.debug(f'Data Load Complete: {self.data_path}')

    def get_diseases(self):
        """Get diseases from data."""
        self.diseases = self.data.groupby('topic')['topic'].count()

    def plot_diseases(self, save=False):
        """Create plot with bar plot above two pie plots that share categories. 

        :param bool save: if True the plot will be saved as .png
        """
        label_size = 14
        title_size = 24
        fig = plt.figure('Diseases Figure', figsize=(10, 16),
                         facecolor='white', edgecolor='black')
        rows, cols = (3, 1)
        ax1 = plt.subplot2grid((rows, cols), (0, 0))
        ax2 = plt.subplot2grid((rows, cols), (1, 0), rowspan=2)

        self.diseases.plot(kind='bar', alpha=0.5, ax=ax1)
        ax1.set_xlabel('Diseases', fontsize=label_size)
        ax1.set_ylabel('Records', fontsize=label_size)

        self.diseases.plot(kind='pie', autopct='%i%%',
                           labeldistance=1.05, pctdistance=0.9, shadow=True,
                           startangle=90, ax=ax2)
        ax2.set_title('Percentage of Data by Disease', fontsize=title_size,
                      y=0.94)
        ax2.set_ylabel('')

        plt.suptitle('Diseases in Data', fontsize=title_size, y=1.03)
        plt.tight_layout()

        if save:
            plt.savefig('diseases_data.png')
        else:
            plt.show()
