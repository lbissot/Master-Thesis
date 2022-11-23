#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file countains the functions used to create the dataset.
"""

import os, sys
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from tabulate import tabulate
import sparta.query_eso_archive as qea
import astropy.coordinates as coord
import astropy.units as u
from astropy.time import Time

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


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

    # Plot the contrast curve for each object and save it in separate files
    plt.figure()
    plt.plot(separation, contrast, label=object)
    plt.xlabel('Separation (arcsec)')
    plt.ylabel('Contrast (5-sigma)')
    plt.yscale(value='log')
    # Set limits for the y axis
    # plt.ylim(1e-7, 1e-1)
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

    # Create dictionnary with the headers as keys and 0 as values.
    missings = {}
    for header in header_list:
        missings[header] = 0

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
                try:
                    data_dict[header] = fits_headers[header]
                except:
                    # print('WARNING! Header {} not found in {}.'.format(header, folder))
                    data_dict[header] = np.nan
                    missings[header] += 1

            separation = fits_data['SEPARATION'][2] # Combination of the two cameras (I think)
            contrast = fits_data['NSIGMA_CONTRAST'][2]

            # Cut the contrast curve at a given separation.
            separation = separation[separation < max_sep]
            contrast = contrast[:len(separation)]

            # Replace the negative values with 0
            contrast[contrast < 0] = 0

            data_dict['SEPARATION'] = separation
            data_dict['NSIGMA_CONTRAST'] = contrast

            # Now we will query simbad and retrieve the flux in G and H bands.
            date = Time(fits_headers['DATE-OBS'])
            name = fits_headers['OBJECT']
            coords = coord.SkyCoord(fits_headers['RA']*u.degree, fits_headers['DEC']*u.degree)

            # Remove Julien's prints
            with HiddenPrints():
                # Remove the warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    simbad_dico = qea.query_simbad(date, coords, name)
            
            # Add the fluxes to the dictionnary.
            if 'simbad_FLUX_G' in simbad_dico.keys():
                data_dict['SIMBAD_FLUX_G'] = simbad_dico['simbad_FLUX_G']
            else:
                data_dict['SIMBAD_FLUX_G'] = np.nan
            
            if 'simbad_FLUX_H' in simbad_dico.keys():
                data_dict['SIMBAD_FLUX_H'] = simbad_dico['simbad_FLUX_H']
            else:
                data_dict['SIMBAD_FLUX_H'] = np.nan

            # Write the summary of the contrast in a file.
            if compute_summary:
                with open(os.path.join(path, folder, 'contrast_summary.txt'), 'w') as f:
                    f.write(get_vector_summary_table(data_dict['NSIGMA_CONTRAST'], 'NSIGMA_CONTRAST'))

            data_dict_list.append(data_dict)

    # Print the percentage of missing values for each header.
    if missings:
        print('Percentage of missing values:')
        for header in missings:
            print('{}: {:.2f}%'.format(header, missings[header]/len(folder_names)*100))
    
    # Create the dataframe.
    df = pd.DataFrame(data_dict_list)

    # If the df has columns ESO TEL PARANG START and ESO TEL PARANG END, compute the absolute delta parang
    if 'ESO TEL PARANG START' in df.columns and 'ESO TEL PARANG END' in df.columns:
        df['DELTA_PARANG'] = np.abs(df['ESO TEL PARANG END'] - df['ESO TEL PARANG START'])
        # Remove the columns ESO TEL PARANG START and ESO TEL PARANG END
        df = df.drop(columns=['ESO TEL PARANG START', 'ESO TEL PARANG END'])

    # Interpolate the contrast curves to have the same number of points for each observation
    if interpolate:
        # Get the median length and use it as the number of points for the interpolation.
        n_points = int(np.median((df['SEPARATION'].apply(lambda x: len(x)))))

        # Get the min and max separation of all the observations.
        # We will use the max(min(x)) and min(max(x)) in order not to have extrapolation.
        min_sep = np.max(df['SEPARATION'].apply(lambda x: np.min(x)))
        max_sep = np.min(df['SEPARATION'].apply(lambda x: np.max(x)))

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


def plot_contrast_curves_summary(path, df, filename='contrast_curves_summary.png', mode='all'):
    """	
    Plot the contrast curves mean, median and quartiles.
    """
    separation = df['SEPARATION'][0]
    # Get a 2D array of the contrast curves
    contrast_curves = np.array(df['NSIGMA_CONTRAST'].tolist())

    # Get the mean, median, first and third quartiles of the contrast curves
    mean = np.mean(contrast_curves, axis=0)
    # Use the log(x+1) values to compute the mean, then take the inverse transformation to get the mean in linear scale.
    mean_log = np.power(10, np.mean(np.log10(contrast_curves + 1), axis=0)) - 1

    median = np.median(contrast_curves, axis=0)
    q1 = np.quantile(contrast_curves, 0.25, axis=0)
    q3 = np.quantile(contrast_curves, 0.75, axis=0)

    # Plot all the curves in transparent grey
    if mode == 'all':
        for i in range(len(contrast_curves)):
            plt.plot(separation, contrast_curves[i], color='grey', alpha=0.2)

    # Plot the mean, median and fill between the quartiles above all the gray curves
    plt.plot(separation, mean, color='orange', label='mean')
    plt.plot(separation, mean_log, color='yellow', label='mean (log)')
    plt.plot(separation, median, color='red', label='median')
    if mode != 'all':
        plt.fill_between(separation, q1, q3, color='lightblue', alpha=0.3, label='quartiles')
    # Plot legend in upper right corner
    plt.legend(loc='upper right')
    plt.xlabel('Separation (arcsec)')
    plt.ylabel('Contrast (sigma)')
    plt.yscale('log')
    plt.title('Contrast curves summary')
    plt.savefig(os.path.join(path, filename), dpi=300)


def get_abs_deviations_from_median(df, log_values=True):
    """
    Get the absolute deviations from the median of the contrast curves.
    """
    # Get a 2D array of the contrast curves
    contrast_curves = np.array(df['NSIGMA_CONTRAST'].tolist())

    if log_values:
        # Get the log(x + 1) values of the contrast curves
        contrast_curves = np.log10(contrast_curves + 1)

    # Compute the median of the contrast curves
    median = np.median(contrast_curves, axis=0)

    deviations = []

    for curve in contrast_curves:
        deviation = np.sum(np.abs(curve - median))
        deviations.append(deviation)

    return np.array(deviations)


def remove_contrast_anomalies(path, df, filename='paths_to_remove.txt', log_values=True, threshold=50):
    """
    Remove the contrast curves that are too different from the median and write their paths in a file.
    """
    abs_deviations = get_abs_deviations_from_median(df, log_values=log_values)

    df['ABS DEV'] = abs_deviations

    # Get the indices of the contrast curves that are too different from the median.
    # The threshold is set to 50.
    indices = np.where(abs_deviations > threshold)[0]

    # Write in a file the paths of the observations to remove.
    with open(path + filename, 'w') as f:
        for index in indices:
            f.write(path + df.loc[index]['folder'] + '\n')

    # Remove the contrast curves that are too different from the median.
    df = df.drop(indices)

    return df
