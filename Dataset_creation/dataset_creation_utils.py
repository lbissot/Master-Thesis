#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file countains the functions used to create the dataset.
"""

import os
import pandas as pd
from astropy.io import fits
from tabulate import tabulate


def write_headers_in_file(fits_headers, file_path, filename='headers.txt', table_format='psql'):
    """
    Write fits headers in a file.
    """

    if not os.path.exists(path):
        exit('ERROR! Folder {} does not exist.'.format(path))
    
    if not file_path[-1] == '/':
        file_path += '/'

    table = []

    with open(os.path.join(file_path, filename), 'w') as f:
        for card in fits_headers.cards:
            table.append([card.keyword, card.value, card.comment])
            
        f.write(tabulate(table, headers=['Keyword', 'Value', 'Comment'], tablefmt=table_format))


def get_folder_names(path):
    """
    Get a list of the folders names located in a given path.
    """

    folder_names = []
    for folder in os.listdir(path):

        if os.path.isdir(os.path.join(path, folder)):
            folder_names.append(folder)
            
    return folder_names


def get_df_with_headers(path, header_list=[], filename='ird_specal_dc-IRD_SPECAL_CONTRAST_CURVE_TABLE-contrast_curve_tab.fits'):
    """
    Get a dataframe with the separation, contrast and headers specified in the list.
    """

    if not os.path.exists(path):
        exit('ERROR! Folder {} does not exist.'.format(path))

    folder_names = get_folder_names(path)

    data_dict_list = []

    for folder in folder_names:
        folder = folder + '/'

        with fits.open(os.path.join(path, folder, filename)) as hdul:

            fits_data = hdul[1].data
            fits_headers = hdul[1].header
            data_dict = {}

            for header in header_list:
                data_dict[header] = fits_headers[header]
            
            data_dict['SEPARATION'] = fits_data['SEPARATION']
            data_dict['NSIGMA_CONTRAST'] = fits_data['NSIGMA_CONTRAST']

            data_dict_list.append(data_dict)

    return pd.DataFrame(data_dict_list)