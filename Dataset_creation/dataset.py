#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file countains the dataset object.
"""

import os, sys
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from tabulate import tabulate
from tqdm import tqdm
import sparta.query_eso_archive as qea
import astropy.coordinates as coord
import astropy.units as u
from astropy.time import Time, TimeDelta
from dataset_creation_utils import HiddenPrints, get_folder_names, get_vector_summary_table, find_lower_upper_bound


class Dataset:

    def get_dataframe(self):
        """
        Get the dataframe of the dataset.

        Returns
        -------
        pandas.DataFrame
            Dataset.
        """
        return self.__df

    def print_missings_percentages(self):
        """
        Print the percentages of missing values in the dataset.

        Returns
        -------
        None.
        """
        if self.__missings is None:
            print('No missing values.')
        else:
            for header in self.__missings:
                print('{}: {:.2f}%'.format(header, self.__missings[header]/len(self.__folder_names)*100))

    def get_contrast_abs_deviations_from_median(self, log_values=True):
        """
        Get the absolute deviations from the median of the contrast curves.

        Parameters
        ----------
        log_values : bool, optional
            If True, the log(x + 1) values of the contrast curves are used. The default is True.

        Returns
        -------
        numpy.ndarray
            Absolute deviations from the median of the contrast curves.
        """
        # Get a 2D array of the contrast curves
        contrast_curves = np.array(self.__df['NSIGMA_CONTRAST'].tolist())

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

    def remove_contrast_anomalies(self, filename='paths_to_remove.txt', log_values=True, threshold=50):
        """
        Remove the contrast curves that are too different from the median.

        Parameters
        ----------
        filename : str, optional
            Name of the file where the paths of the observations to remove are written. The default is 'paths_to_remove.txt'.

        log_values : bool, optional
            If True, the log(x + 1) values of the contrast curves are used. The default is True.

        threshold : int, optional
            Threshold of the absolute deviation from the median. The default is 50.

        Returns
        -------
        None.
        """
        
        abs_deviations = self.get_contrast_abs_deviations_from_median(log_values=log_values)
        # Get the indices of the contrast curves that are too different from the median.
        # The threshold is set to 50.
        indices = np.where(abs_deviations > threshold)[0]

        # Write in a file the paths of the observations to remove.
        with open(self.__path + filename, 'w') as f:
            for index in indices:
                f.write(self.__path + self.__df.loc[index]['folder'] + '\n')

        # Remove the contrast curves that are too different from the median.
        self.__df = self.__df.drop(indices)

    def plot_contrast_curves_summary(self, filename='contrast_curves_summary.png', mode='all'):
        """	
        Plot the contrast curves along with their mean and median curves.

        Parameters
        ----------
        filename : str, optional
            Name of the file where the plot is saved. The default is 'contrast_curves_summary.png'.

        mode : str, optional
            If 'all', all the contrast curves are plotted. If 'summary', only the mean, median and quartiles are plotted. The default is 'all'.

        Returns
        -------
        None.
        """
        
        separation = self.__df['SEPARATION'][0]
        # Get a 2D array of the contrast curves
        contrast_curves = np.array(self.__df['NSIGMA_CONTRAST'].tolist())
        # Get the mean, median, first and third quartiles of the contrast curves
        mean = np.mean(contrast_curves, axis=0)
        # Use the log(x+1) values to compute the mean, then take the inverse transformation to get the mean in linear scale.
        # mean_log = np.power(10, np.mean(np.log10(contrast_curves + 1), axis=0)) - 1

        median = np.median(contrast_curves, axis=0)
        q1 = np.quantile(contrast_curves, 0.25, axis=0)
        q3 = np.quantile(contrast_curves, 0.75, axis=0)

        # Plot all the curves in transparent grey
        if mode == 'all':
            for i in range(len(contrast_curves)):
                plt.plot(separation, contrast_curves[i], color='grey', alpha=0.2)

        # Plot the mean, median and fill between the quartiles above all the gray curves
        plt.plot(separation, mean, color='orange', label='mean')
        # plt.plot(separation, mean_log, color='yellow', label='mean (log)')
        plt.plot(separation, median, color='red', label='median')
        if mode != 'all':
            plt.fill_between(separation, q1, q3, color='lightblue', alpha=0.3, label='quartiles')
        # Plot legend in upper right corner
        plt.legend(loc='upper right')
        plt.xlabel('Separation (arcsec)')
        plt.ylabel('Contrast (sigma)')
        plt.yscale('log')
        plt.title('Contrast curves summary')
        plt.savefig(os.path.join(self.__path, filename), dpi=300)

    def __init__(self, path, max_sep=25, plot=False, write=False):
        """
        This class is used to create a dataset from a database.

        Parameters
        ----------
        path : str
            Path to the database files from the sphere DC data.
        max_sep : float, optional
            Maximum separation between the target and the star in the dataset.
            The default is 25.
        plot : bool, optional
            If True, the contrast curves are plotted and saved in the database folder.
            The default is False.
        write : bool, optional
            If True, the headers are written in a file in the database folder.  
            The default is False.

        Returns
        -------
        None.

        """

        if not os.path.exists(path):
            exit('ERROR! Folder {} does not exist.'.format(path))

        self.__path = path

        # List of fits headers to be used in the dataset.
        self.__header_list = ['ESO OBS ID', 'DATE-OBS', 'ESO OBS START', 'OBJECT', 'ESO TEL AMBI FWHM MEAN', 'ESO TEL TAU0 MEAN', \
            'ESO TEL AIRM MEAN', 'EFF_NFRA', 'EFF_ETIM', 'SR_AVG', 'ESO INS4 FILT3 NAME', \
                'ESO INS4 OPTI22 NAME', 'ESO AOS VISWFS MODE', 'ESO TEL AMBI WINDSP', 'SCFOVROT', 'SC MODE', \
                    'ESO TEL AMBI RHUM', 'HIERARCH ESO INS4 TEMP422 VAL', 'HIERARCH ESO TEL TH M1 TEMP', \
                        'HIERARCH ESO TEL AMBI TEMP']
        
        self.__folder_names = get_folder_names(path)
        self.__filename='ird_specal_dc-IRD_SPECAL_CONTRAST_CURVE_TABLE-contrast_curve_tab.fits'

        # List of dictionnaries whose keys will be the same among all the dictionnaries.
        # It will then be converted into a dataframe.
        data_dict_list = []

        # Create dictionnary with the headers as keys and 0 as values.
        self.__missings = {}
        for header in self.__header_list:
            self.__missings[header] = 0
        self.__missings['SIMBAD_FLUX_G'] = 0
        self.__missings['SIMBAD_FLUX_H'] = 0

        print('Creating the dataset...')
        for folder in tqdm(self.__folder_names):
            folder = folder + '/'

            with fits.open(os.path.join(path, folder, self.__filename)) as hdul:
                fits_data = hdul[1].data
                fits_headers = hdul[1].header  

                # Write the headers in a file.
                if write:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        self.__write_headers_in_file(os.path.join(path, folder), fits_headers)

                data_dict = {}
                
                data_dict['folder'] = str(folder)

                # Add the wanted fits headers to the dictionnary.
                for header in self.__header_list:
                    try:
                        # Precise the type of the data.
                        if header == 'DATE-OBS':
                            data_dict[header] = Time(fits_headers[header])

                        else:
                            data_dict[header] = fits_headers[header]
                    except:
                        # Try to recover the exposure time using other headers.
                        if header == 'EFF_ETIM':
                            try:
                                data_dict[header] = fits_headers['ESO DET SEQ1 EXPTIME']
                            except:
                                try:
                                    data_dict[header] = fits_headers['ESO DET NDIT'] * fits_headers['ESO DET SEQ1 DIT']
                                except:
                                    data_dict[header] = np.nan
                                    self.__missings[header] += 1
                        else:  
                            data_dict[header] = np.nan
                            self.__missings[header] += 1

                # Compute ESO OBS STOP = ESO OBS START + EFF_ETIM
                if 'ESO OBS START' in self.__header_list and 'EFF_ETIM' in self.__header_list:
                    data_dict['ESO OBS STOP'] = Time(data_dict['ESO OBS START']) + TimeDelta(data_dict['EFF_ETIM'], format='sec')

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
                    self.__missings['SIMBAD_FLUX_G'] += 1
                
                if 'simbad_FLUX_H' in simbad_dico.keys():
                    data_dict['SIMBAD_FLUX_H'] = simbad_dico['simbad_FLUX_H']
                else:
                    data_dict['SIMBAD_FLUX_H'] = np.nan
                    self.__missings['SIMBAD_FLUX_H'] += 1

                # Since we have a lot of observations lasting less than 5 minutes we probably won't find the time interval
                # in the ASM database. So we will query the ASM database with a 15 minutes addition before and after.
                start = Time(data_dict['ESO OBS START']) - TimeDelta(900, format='sec')
                stop = Time(data_dict['ESO OBS STOP']) + TimeDelta(900, format='sec')

                # Check if the observation is before april 02 2016 (Update of the ASM)
                with HiddenPrints():
                    if data_dict['ESO OBS START'] < Time('2016-04-02T00:00:00'):
                        # Query the old dimm
                        try: 
                            asm_data = qea.query_old_dimm(os.path.join(path, folder), str(start), str(stop))
                        except:
                            asm_data = pd.DataFrame()
                    else:
                        # Query mass
                        try:
                            asm_data = qea.query_mass(os.path.join(path, folder), str(start), str(stop))
                        except:
                            asm_data = pd.DataFrame()

                if len(asm_data) != 0:
                    lower, _ = find_lower_upper_bound(asm_data.to_dict()['Date time'], data_dict['ESO OBS START'])
                    _, upper = find_lower_upper_bound(asm_data.to_dict()['Date time'], data_dict['ESO OBS STOP'])

                    # Drop the rows before lower and after upper.
                    asm_data = asm_data.reset_index()
                    asm_data = asm_data.iloc[lower:upper+1]

                    try:
                        asm_data['Date time'] = asm_data['Date time'].dt.floor('T')
                        asm_data = asm_data.set_index('Date time')
                        new_dates = pd.date_range(start=asm_data.index[0], end=asm_data.index[-1], freq='1min')
                        asm_data = asm_data.reindex(new_dates)
                        asm_data = asm_data.interpolate(method='linear')
                    except:
                        print("Error while interpolating the asmd-mass data for the folder {}".format(folder))
                        print("ESO OBS START = {} and ESO OBS STOP = {}".format(data_dict['ESO OBS START'], data_dict['ESO OBS STOP']))
                        # print(asm_data)
                else:
                    print("No data for the folder {}".format(folder))
                    

                # Write the summary of the contrast in a file.
                with open(os.path.join(self.__path, folder, 'contrast_summary.txt'), 'w') as f:
                    try:
                        f.write(get_vector_summary_table(data_dict['NSIGMA_CONTRAST'], 'NSIGMA_CONTRAST'))
                    except:
                        print("Error while writing the contrast summary for the folder {} ({})".format(i, folder))
                        print(data_dict['NSIGMA_CONTRAST'])

                data_dict_list.append(data_dict)

        # Create the dataframe.
        self.__df = pd.DataFrame(data_dict_list)

        # Interpolate the contrast curves to have the same number of points for each observation
        # Get the median length and use it as the number of points for the interpolation.
        n_points = int(np.median((self.__df['SEPARATION'].apply(lambda x: len(x)))))

        # Get the min and max separation of all the observations.
        # We will use the max(min(x)) and min(max(x)) in order not to have extrapolation.
        min_sep = np.max(self.__df['SEPARATION'].apply(lambda x: np.min(x)))
        max_sep = np.min(self.__df['SEPARATION'].apply(lambda x: np.max(x)))

        # Create the new separation array
        new_sep = np.linspace(min_sep, max_sep, n_points)

        # Interpolate the contrast curves
        new_contrast = self.__df.apply(lambda x: np.interp(new_sep, x['SEPARATION'], x['NSIGMA_CONTRAST']), axis=1)
        new_sep = self.__df.apply(lambda x: new_sep, axis = 1)

        # Replace columns in the dataframe
        self.__df['SEPARATION'] = new_sep
        self.__df['NSIGMA_CONTRAST'] = new_contrast

        # Plot the contrast curves (in files)
        if plot:
            print('Creating the contrast plots...')
            dict_df = self.__df.to_dict()
            for i in tqdm(range(len(dict_df['folder']))):
                self.__plot_contrast(os.path.join(path, dict_df['folder'][i]), dict_df['SEPARATION'][i], dict_df['NSIGMA_CONTRAST'][i], dict_df['OBJECT'][i])


    def __write_headers_in_file(self, path, fits_headers, filename='headers.txt', table_format='psql'):
        """
        Write fits headers in a file.

        Parameters
        ----------
        path : str
            Path to the folder where the file will be written.

        fits_headers : astropy.io.fits.Header
            The fits headers.

        filename : str, optional
            Name of the file where the headers will be written. The default is 'headers.txt'.

        table_format : str, optional
            Format of the table. The default is 'psql'.

        Returns
        -------
        None.
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

    def __plot_contrast(self, path, separation, contrast, object=''):
        """
        Plot the contrast curve and save it in a file.

        Parameters
        ----------
        path : str
            Path to the folder where the file will be written.
        
        separation : numpy.ndarray
            Separation vector.

        contrast : numpy.ndarray
            Contrast vector.

        object : str, optional
            Name of the object. The default is ''.

        Returns
        -------
        None.
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