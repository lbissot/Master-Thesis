#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file countains the functions used to create the dataset.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from tabulate import tabulate


def get_folder_names(path):
    """
    Get a list of the folders names located in a given path.
    """

    folder_names = []
    for folder in os.listdir(path):

        if os.path.isdir(os.path.join(path, folder)):
            folder_names.append(folder)
            
    return folder_names


def write_headers_in_file(path, fits_headers, filename='headers.txt', table_format='psql'):
    """
    Write fits headers in a file.
    """

    if not os.path.exists(path):
        exit('ERROR! Folder {} does not exist.'.format(path))
    
    if not path[-1] == '/':
        path += '/'

    table = []

    with open(os.path.join(path, filename), 'w') as f:
        for card in fits_headers.cards:

            # If the card is unparsable (it does not follow the FITS standard), it has to be fixed.
            try:
                table.append([card.keyword, card.value, card.comment])
            except:
                # print('WARNING! Card {} is unparsable.'.format(card.keyword))
                try:
                    card.verify('fix')
                    table.append([card.keyword, card.value, card.comment])
                except:
                    print('Card {} cannot be fixed, it will not be written in the file.'.format(card.keyword))

            
        f.write(tabulate(table, headers=['Keyword', 'Value', 'Comment'], tablefmt=table_format))


def get_vector_summary_table(vector, vector_name=''):
    """
    Get a table with the summary of a vector.
    """

    table = []

    table.append([vector_name, np.mean(vector), np.std(vector), np.median(vector), np.min(vector), np.max(vector)])

    return tabulate(table, headers=['Name', 'Mean', 'Std', 'Median', 'Min', 'Max'], tablefmt='psql')


def plot_contrast(path, separation, contrast, object=''):
    """
    Plot the contrast curve.
    """

    if not os.path.exists(path):
        exit('ERROR! Folder {} does not exist.'.format(path))

    if not path.endswith('/'):
        path += '/'

    # Log transform of the contrast and warning suppression
    # IDK if this is necessary if we use plt.yscale(value='log')

    # with np.errstate(divide='ignore' , invalid='ignore'):
    #     contrast = np.log10(contrast) # There are some negative and zero values in the contrast curves, maybe process them before plotting

    # Plot the contrast curve for each object and save it in separate files
    plt.figure()
    plt.plot(separation, contrast, label=object)
    plt.xlabel('Separation (arcsec)')
    plt.ylabel('Contrast (5-sigma)')
    plt.yscale(value='log')
    # Set limits for the y axis
    plt.ylim(1e-7, 1e-1)
    plt.title('Contrast curve for {}'.format(object))
    filename = os.path.join(path, 'contrast_curve.png')
    # print('Saving figure {}...'.format(filename))
    plt.plot()
    plt.savefig(filename)
    plt.close()


# Maybe interresting to specify the type of the data in order to manipulate it later.
def get_df_with_headers(path, header_list=[], filename='ird_specal_dc-IRD_SPECAL_CONTRAST_CURVE_TABLE-contrast_curve_tab.fits', \
    max_sep=25 ,interpolate=True, compute_summary=False, write_headers=False, compute_plots=False):
    """
    Get a dataframe with the separation, contrast and headers specified in the list.
    - max_sep: maximum separation to consider in the dataframe.
    - If interpolate is True, the contrast is interpolated to have the same number of points for each contrast curve.
    - If compute_summary is True, the summary of the contrast is computed and written in separate files (one file per contrast curve).
    - If write_headers is True, the headers are written in separate files (one file per contrast curve).
    - If compute_plots is True, the contrast curves are plotted and saved in separate files (one file per contrast curve).
    """

    if not os.path.exists(path):
        exit('ERROR! Folder {} does not exist.'.format(path))

    folder_names = get_folder_names(path)

    # List of dictionnaries whose keys will be the same among all the dictionnaries.
    # It will then be converted into a dataframe.
    data_dict_list = []

    for folder in folder_names:
        folder = folder + '/'

        with fits.open(os.path.join(path, folder, filename)) as hdul:

            fits_data = hdul[1].data
            fits_headers = hdul[1].header

            # Write the headers in a file.
            if write_headers:
                write_headers_in_file(os.path.join(path, folder), fits_headers)

            data_dict = {}

            # Save the folder name for potential use.
            data_dict['folder'] = folder

            for header in header_list:
                data_dict[header] = fits_headers[header]

            separation = fits_data['SEPARATION'][2] # Combination of the two cameras (I think)
            contrast = fits_data['NSIGMA_CONTRAST'][2]

            # Cut the contrast curve at a given separation.
            separation = separation[separation < max_sep]
            contrast = contrast[:len(separation)]

            data_dict['SEPARATION'] = separation
            data_dict['NSIGMA_CONTRAST'] = contrast

            # Write the summary of the contrast in a file.
            if compute_summary:
                with open(os.path.join(path, folder, 'contrast_summary.txt'), 'w') as f:
                    f.write(get_vector_summary_table(data_dict['NSIGMA_CONTRAST'], 'NSIGMA_CONTRAST'))

            data_dict_list.append(data_dict)

    df = pd.DataFrame(data_dict_list)

    # Interpolate the contrast curves to have the same number of points for each observation
    if interpolate:
        # Get the median length and use it as the number of points for the interpolation.
        n_points = int(np.median((df['SEPARATION'].apply(lambda x: len(x)))))

        # Get the min and max separation of all the observations.
        # Might not be the best way to do it. Maybe I should use max(min(x)) and min(max(x)).
        min_sep = np.min(df['SEPARATION'].apply(lambda x: np.min(x)))
        max_sep = np.max(df['SEPARATION'].apply(lambda x: np.max(x)))

        # Create the new separation array
        new_sep = np.linspace(min_sep, max_sep, n_points)

        # Interpolate the contrast curves
        new_contrast = df.apply(lambda x: np.interp(new_sep, x['SEPARATION'], x['NSIGMA_CONTRAST']), axis=1)
        new_sep = df.apply(lambda x: new_sep, axis = 1)

        # Replace columns in the dataframe
        df['SEPARATION'] = new_sep
        df['NSIGMA_CONTRAST'] = new_contrast

    # Plot the contrast curves
    if compute_plots:

        dict_df = df.to_dict()

        for i in range(len(dict_df['folder'])):
            plot_contrast(os.path.join(path, dict_df['folder'][i]), dict_df['SEPARATION'][i], dict_df['NSIGMA_CONTRAST'][i], dict_df['OBJECT'][i])

    return df


def plot_contrast_curves_summary(path, df):
    """	
    Plot the contrast curves mean, median and quartiles.
    """
    # Get a 2D array of the contrast curves
    contrast_curves = np.array(df['NSIGMA_CONTRAST'].tolist())

    # Get the mean, median, first and third quartiles of the contrast curves
    mean = np.mean(contrast_curves, axis=0)
    median = np.median(contrast_curves, axis=0)
    q1 = np.quantile(contrast_curves, 0.25, axis=0)
    q3 = np.quantile(contrast_curves, 0.75, axis=0)

    # Plot the mean, median and fill between the quartiles
    plt.plot(df['SEPARATION'][0], mean, label='Mean')
    plt.plot(df['SEPARATION'][0], median, label='Median')
    plt.fill_between(df['SEPARATION'][0], q1, q3, alpha=0.5, label='Quartiles')
    plt.legend()
    plt.xlabel('Separation (arcsec)')
    plt.ylabel('Contrast 5-sigma')
    plt.yscale('log')
    plt.savefig(os.path.join(path, "contrast_curves_summary_plot.png"), dpi=300)
