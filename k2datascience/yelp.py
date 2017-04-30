#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Yelp Dataset Challenge Module

.. moduleauthor:: Timothy Helton <timothy.j.helton@gmail.com>
"""

from collections import Counter
import logging
import os
import os.path as osp
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


log_format = ('%(asctime)s  %(levelname)8s  -> %(name)s <- '
              '(line: %(lineno)d) %(message)s\n')
date_format = '%m/%d/%Y %I:%M:%S'
logging.basicConfig(format=log_format, datefmt=date_format,
                    level=logging.INFO)

current_dir = osp.dirname(osp.realpath(__file__))
data_dir = osp.realpath(osp.join(current_dir, '..', 'data', 'yelp'))
data_files = (
    'business',
    # 'checkin',
    # 'review',
    # 'tip',
    # 'user',
)


def convert_boolean(series):
    """Convert Pandas Series boolean values True to 1 and False to 0.

    :param pandas.Series series: Pandas Series to be converted
    """

    def conversion(conversion_dict):
        """Recursively convert boolean objects to integers.

        :param dict conversion_dict: dictionary to be converted
        :returns: original dictionary with boolean object replaced with \
            integers
        :rtype: dict
        """
        for k, v in conversion_dict.items():
            if isinstance(v, dict):
                conversion(v)
            elif isinstance(v, bool):
                conversion_dict[k] = v * 1
        return conversion_dict

    return series.map(conversion)


class YDC:
    """Yelp Dataset Challenge Class
    
    :Attributes:
    
    **data_dir**: *str* path to directory containing data files
    **file_data**: *Dict[str, pandas.DataFrame]* data frames of the loaded \ 
        data files
    **file_paths**: *Dict[str, str]* paths to the data files
    **file_types**: *Tuple(str)* names of the files data files to be loaded
    
    """
    def __init__(self,
                 data_path: str=data_dir,
                 file_types: Tuple[str]=data_files):
        self.data_dir = data_path
        self.file_types = file_types
        self.file_paths = {}
        self.file_data = {}

    def __repr__(self) -> str:
        return (
            f'YDC('
            f'data_path={self.data_dir}, '
            f'file_types={self.file_types}'
        )

    def load_data(self):
        """Load original data from json files."""
        for file_type in self.file_types:
            path, data = self.data_files(file_type)
            self.file_paths[file_type] = path
            self.file_data[file_type] = data
            logging.info(f'File Loaded: {osp.basename(path)}')

    def data_files(self, descriptor: str):
        """Return path to data file for a specific descriptor.

        :param str descriptor:
        :returns: full path to data file
        :rtype: str
        """
        prefix = 'yelp_academic_dataset_'
        extension = '.json'
        real_path = osp.realpath(osp.join(self.data_dir,
                                          f'{prefix}{descriptor}{extension}'))
        data = pd.read_json(real_path, lines=True)
        return real_path, data

    def get_restaurant_type(self):
        """Count the categories and save under a new column restaurant type."""
        restaurant_type = self.file_data['business'].categories.map(Counter)
        self.file_data['business']['restaurant_type'] = restaurant_type

    def get_zip_codes_canada(self):
        """Look for zip codes from Canada."""
        bus = self.file_data['business']
        mask = bus.zip_code.isnull()
        codes = bus[mask].full_address.str.extract(r"(\w\d\w(\s\d\w\d)?$)",
                                                   expand=True)
        bus.loc[mask, 'zip_code'] = codes[0]

    def get_zip_codes_scotland(self):
        """Look for zip codes from Scotland."""
        bus = self.file_data['business']
        mask = bus.zip_code.isnull()
        codes = bus[mask].full_address.str.extract(r"(\w\w\d+\s?\d?(\w\w)?$)",
                                                   expand=True)
        bus.loc[mask, 'zip_code'] = codes[0]

    def get_zip_codes_usa(self):
        """Look for zip codes from the United States."""
        bus = self.file_data['business']
        bus['zip_code'] = bus.full_address.str.extract(r"(\d{5})", expand=True)

    def get_zip_codes(self):
        """Population the zip code column."""
        self.get_zip_codes_usa()
        self.get_zip_codes_canada()
        self.get_zip_codes_scotland()
