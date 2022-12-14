#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file countains the utilitary functions.
"""

import os, sys
import numpy as np
from tabulate import tabulate

class HiddenPrints:
    """
    This class is used to hide the prints of a function.
    """	
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def get_folder_names(path):
    """
    Get a list of the folders names located at a given path.

    Parameters
    ----------
    path : str
        Path to the folder.

    Returns
    -------
    folder_names : list
        List of the folders names.
    """

    folder_names = []
    for folder in os.listdir(path):

        if os.path.isdir(os.path.join(path, folder)):
            folder_names.append(folder)
            
    return folder_names


def find_lower_upper_bound(ls, x):
    """
    Find the lower and upper bound indexes of a given value in a list.

    Parameters
    ----------
    ls : list
        List to search in.
    x : object of the same type as the list
        Value to search.

    Returns
    -------
    lower_bound : int
        Lower bound index.
    upper_bound : int
        Upper bound index.
    """
    lower_bound = 0
    upper_bound = len(ls) - 1
    while upper_bound - lower_bound > 1:
        mid = (upper_bound + lower_bound) // 2
        if ls[mid] < x:
            lower_bound = mid
        else:
            upper_bound = mid
    return lower_bound, upper_bound


def get_vector_summary_table(vector, vector_name=''):
    """
    Get a table with the summary of a vector.

    Parameters
    ----------
    vector : list
        Vector to summarize.

    vector_name : str, optional
        Name of the vector. The default is ''.

    Returns
    -------
    table : str
        PSQL table with the summary of the vector.
    """

    table = []

    table.append([vector_name, np.mean(vector), np.std(vector), np.median(vector), np.min(vector), np.max(vector)])

    return tabulate(table, headers=['Name', 'Mean', 'Std', 'Median', 'Min', 'Max'], tablefmt='psql')