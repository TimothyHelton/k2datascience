#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Job Satisfaction Module

.. moduleauthor:: Timothy Helton <timothy.j.helton@gmail.com>
"""
import logging
import os
import os.path as osp

from bs4 import BeautifulSoup
import pandas as pd
import requests


log_format = ('%(asctime)s  %(levelname)8s  -> %(name)s <- '
              '(line: %(lineno)d) %(message)s\n')
date_format = '%m/%d/%Y %I:%M:%S'
logging.basicConfig(format=log_format, datefmt=date_format,
                    level=logging.INFO)

current_dir = osp.dirname(osp.realpath(__file__))
data_dir = osp.realpath(osp.join(current_dir, '..', 'data', 'roots'))
census_cities = osp.join(data_dir, '2016_Gaz_zcta_national.txt')


class LoadData:
    """Load Data 
    
    :param str zipcode_file: data file with US zip codes
    
    :Attributes:
    
    """
    def __init__(self, zip_code_file=census_cities):
        self.zip_code_data = None
        self.zip_code_file = zip_code_file

    def __repr__(self):
        return (f'LoadData('
                f'zip_code_file={self.zip_code_file}')

    def load_data(self):
        """Load US city data."""
        data_types = {
            'zip_code': str,
            'latitude': float,
            'longitude': float,
        }
        self.zip_code_data = pd.read_table(
            self.zip_code_file,
            dtype=data_types,
            encoding='ISO-8859-1',
            header=0,
            names=data_types.keys(),
            usecols=[0, 5, 6],
        )


class Indeed(LoadData):
    """Attributes and methods related to querying indeed.com.
    
    :Attributes:
    
    - **url**: *str* url address of target page to scrape
    """
    def __init__(self):
        super().__init__()
        self.base_url = 'https://www.indeed.com/'
        self.jobs = None
        self.url = None
        self._soup = None

    @property
    def soup(self):
        self.scrape_page(self.url)
        return self._soup

    def __repr__(self):
        return 'Jobs()'

    def get_company(self, company):
        """Get indeed url for a company.
        
        :param str company: company name
        """
        self.url = f'{self.base_url}cmp/{company}/'

    def get_jobs(self, what, location):
        """Search for jobs.
        
        :param str what: what job to search for (ie: 'Data Scientist')
        :param str location: city and state of desired job location \
            (ie: Austin, TX)
        """
        what = what.replace(' ', '+')
        location = location.replace(', ', '%2C+')
        self.url = f'{self.base_url}/jobs?q={what}&l={location}'
        self.scrape_page(self.url)

    def get_salary(self, company):
        """Get indeed url for a company salary data.
        
        :param str company: company name 
        """
        self.url = f'{self.base_url}cmp/{company}/salaries'

    def scrape_page(self, url):
        """Scrape requested url.
        
        :param str url: url of web page to be scraped.
        """
        r = requests.get(url)
        self._soup = BeautifulSoup(r.text, 'lxml')
