#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from pkg_resources import get_distribution, DistributionNotFound
import os.path as osp

from . import chronic_disease
from . import hr_analytics
from . import movies
from . import nyc_mta
from . import olympics
from . import pca
from . import plotting
from . import yelp


__version__ = '1.0.0'

try:
    _dist = get_distribution('k2datascience')
    dist_loc = osp.normcase(_dist.location)
    here = osp.normcase(__file__)
    if not here.startswith(osp.join(dist_loc, 'k2datascience')):
        raise DistributionNotFound
except DistributionNotFound:
    __version__ = 'Please install this project with setup.py'
else:
    __version__ = _dist.version
