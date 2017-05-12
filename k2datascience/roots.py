#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Job Satisfaction Module

.. moduleauthor:: Timothy Helton <timothy.j.helton@gmail.com>
"""
from bs4 import BeautifulSoup
import pandas as pd
import requests


class Indeed:
    """Attributes and methods related to querying indeed.com.
    
    :Attributes:
    
    - **url**: *str* url address of target page to scrape
    """
    def __init__(self):
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
