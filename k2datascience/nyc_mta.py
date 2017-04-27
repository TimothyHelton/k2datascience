#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" New York City MTA Turnstile Module

.. moduleauthor:: Timothy Helton <timothy.j.helton@gmail.com>
"""

import glob
import logging
import os
import os.path as osp
import re

import pandas as pd
import requests


log_format = ('%(asctime)s  %(levelname)8s  -> %(name)s <- '
              '(line: %(lineno)d) %(message)s\n')
date_format = '%m/%d/%Y %I:%M:%S'
logging.basicConfig(format=log_format, datefmt=date_format,
                    level=logging.INFO)


class TurnstileData:
    """Methods related to acquiring New York City MTA subway turnstile data.
    
    :Attributes:
    
    **data**: *pandas.DataFrame* NYC turnstile data
    **data_dir**: *str* path to the local data directory
    **data_files**: *list* names of all available data files to download from \
        the url attribute
    **request**: *requests.models.Response* response object from scraping \
        the url attribute
    **station_daily**: *pandas.DataFrame* sum of entries and exits by station
    **turnstile_daily**: *pandas.DataFrame* sum of entries and exits by \
        turnstile
    **url**: *str* web address for turnstile data
    """
    def __init__(self):
        self.url = 'http://web.mta.info/developers/turnstile.html'
        self.request = requests.get(self.url)
        self.data = None
        self.data_dir = osp.realpath(osp.join(osp.dirname(__file__), '..',
                                              'data', 'nyc_mta_turnstile'))
        self.data_files = None

        self.get_data()

        self._turnstile_daily = None
        self._station_daily = None

    @property
    def station_daily(self):
        self.get_station_daily()
        return self._station_daily

    @property
    def turnstile_daily(self):
        self.get_turnstile_daily()
        return self._turnstile_daily

    def __repr__(self):
        return f'TurnstileData()'

    def available_data_files(self):
        """Find all available turnstile data files to retrieve."""
        self.data_files = re.findall(pattern=r'href="(data.*?.txt)"',
                                     string=self.request.text)
        self.data_files = [f'http://web.mta.info/developers/{x}'
                           for x in self.data_files]

    def get_data(self):
        """Retrieve data from raw files."""
        raw_files = glob.glob(osp.join(self.data_dir, '*'))
        frames = (pd.read_csv(x) for x in raw_files)
        self.data = pd.concat(frames, ignore_index=True)
        self.data.columns = [x.strip().lower().replace('/', '_')
                             for x in self.data.columns]

    def get_turnstile_daily(self):
        """Filter data to find daily entries and exits per turnstile."""
        mask = ['c_a', 'unit', 'station', 'scp',
                pd.Series([x.date() for x in self.data.time_stamp],
                          name='date')]
        self._turnstile_daily = self.data.groupby(mask)['entries',
                                                        'exits'].sum()

    def get_station_daily(self, control_area=False, unit=False):
        """Filter data to find entries and exits per station.
        
        .. note:: Data will always be filtered by station, date, week and \ 
        weekday.
        
        :param bool control_area: if True the data will be additionally \ 
            filtered based on control area
        :param bool unit: if True data will be additionally filtered based \ 
            on unit
        """
        mask = ['station',
                pd.Series([x.date() for x in self.data.time_stamp],
                          name='date'),
                pd.Series([x.week for x in self.data.time_stamp],
                          name='week'),
                pd.Series([x.weekday() for x in self.data.time_stamp],
                          name='weekday')]

        if unit:
            mask = ['unit'] + mask

        if control_area:
            mask = ['c_a'] + mask

        self._station_daily = self.data.groupby(mask)['entries', 'exits'].sum()

    def get_time_stamp(self):
        """Add Series to data that is date_time object."""
        time_stamp = pd.to_datetime(self.data.date.str.cat(self.data.time,
                                                           sep=' '))
        self.data['time_stamp'] = time_stamp

    def write_data_files(self, qty=None, overwrite=False):
        """Retrieve and write requested data files to the data directory.
        
        :param int qty: number of records to retrieve beginning with the most \
            recent data (default of None will retrieve all available data \
            files)
        :param bool overwrite: if True existing files will be overwritten \
            with new data
        """
        os.makedirs(self.data_dir, exist_ok=True)

        if self.data_files is None:
            self.available_data_files()

        if qty is not None:
            retrieve = self.data_files[:qty]
        else:
            retrieve = self.data_files

        remaining_files = len(retrieve)
        for path in retrieve:
            file_name = path.split('_')[-1]
            file_path = osp.join(self.data_dir, file_name)

            if osp.isfile(file_path) and not overwrite:
                logging.info(f'Using Existing File: {file_name}')
                continue

            logging.info(f'Scraping: {path}')
            r = requests.get(path)
            if r.status_code >= 400:
                logging.error(f'Error in File: {path}')
                continue

            with open(file_path, 'w') as f:
                f.write(r.text)
            remaining_files -= 1
            logging.info(f'Remaining Files: {remaining_files}\n\n')

