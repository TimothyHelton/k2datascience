#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Utilities Module

.. moduleauthor:: Timothy Helton <timothy.j.helton@gmail.com>
"""
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter


ax_formatter = {
    'billions': FuncFormatter(lambda x, position: f'{x * 1e-9:.0f}'),
    'millions': FuncFormatter(lambda x, position: f'{x * 1e-6:.0f}'),
    'percent_convert': FuncFormatter(lambda x, position: f'{x * 100:.0f}%'),
    'percent': FuncFormatter(lambda x, position: f'{x * 100:.0f}%'),
    'thousands': FuncFormatter(lambda x, position: f'{x * 1e-3:.0f}'),
}

size = {
    'label': 14,
    'legend': 12,
    'title': 20,
    'super_title': 24,
}


def save_fig(name=None, save=False):
    """
    Helper function to save or display figure.

    :param str name: file name
    :param bool save: if True the figure will be saved
    """
    if save:
        plt.savefig(f'{name}.png', bbox_inches='tight',
                    bbox_extra_artists=[size['super_title']])
    else:
        plt.show()
