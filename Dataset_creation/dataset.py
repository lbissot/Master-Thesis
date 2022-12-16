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
        # 'ESO TEL AMBI FWHM MEAN', 'ESO TEL TAU0 MEAN' are not used because they are not available for all the observations.
        # We will query http://archive.eso.org/cms/eso-data/ambient-conditions/paranal-ambient-query-forms.html 
        # to recover them and thus being consistant by applying the same correction to all the observations.
        self.__header_list = ['ESO OBS ID', 'DATE-OBS', 'OBJECT', \
            'ESO TEL AIRM MEAN', 'EFF_NFRA', 'EFF_ETIM', 'SR_AVG', 'ESO INS4 FILT3 NAME', \
                'ESO INS4 OPTI22 NAME', 'ESO AOS VISWFS MODE', 'ESO TEL AMBI WINDSP', 'SCFOVROT', 'SC MODE', \
                    'ESO TEL AMBI RHUM', 'HIERARCH ESO INS4 TEMP422 VAL', 'HIERARCH ESO TEL TH M1 TEMP', \
                        'HIERARCH ESO TEL AMBI TEMP', 'OBS_STA', 'OBS_END']
        
        self.__folder_names = get_folder_names(path)
        self.__filename='ird_specal_dc-IRD_SPECAL_CONTRAST_CURVE_TABLE-contrast_curve_tab.fits'

        # List of dictionnaries whose keys will be the same among all the dictionnaries.
        # It will then be converted into a dataframe.
        data_dict_list = []

        # Create dictionnary with the headers as keys and 0 as values.
        self.__missings = {}
        # We will investiguate the missing headers to see if the missings correspond
        # to a specific update time of the database.
        self.__dates_missing_dict = {}
        self.__dates_not_missing_dict = {}

        for header in self.__header_list:
            # Counter of the number of missing headers
            self.__missings[header] = 0
            # List of the dates of the missing headers (to retrieve min and max dates)
            self.__dates_missing_dict[header] = []
            # List of the dates of the not missing headers (to retrieve min and max dates)
            self.__dates_not_missing_dict[header] = []

        # Not in header list as it is not a fits header but a query to make.
        self.__missings['SIMBAD_FLUX_G'] = 0
        self.__missings['SIMBAD_FLUX_H'] = 0
        self.__missings['SEEING_MEDIAN'] = 0
        self.__missings['SEEING_STD'] = 0
        self.__missings['COHERENCE_TIME_MEDIAN'] = 0
        self.__missings['COHERENCE_TIME_STD'] = 0

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

                        elif header == 'SR_AVG' and fits_headers[header] <= 0:
                            data_dict[header] = np.nan
                            self.__missings[header] += 1

                        else:
                            data_dict[header] = fits_headers[header]

                        self.__dates_not_missing_dict[header].append(Time(fits_headers['DATE']))
                    except:

                        self.__dates_missing_dict[header].append(Time(fits_headers['DATE']))

                        # Create a file named 'header_missing.txt' with the folders where the header is missing.
                        # If first encountered missing header, create the file and write the folder name.
                        # Else, append the folder name to the file.
                        if self.__missings[header] == 0:
                            with open(os.path.join(path, "{}_missing.txt".format(header)), 'w') as f:
                                f.write("Folder : {}\nDate: {}\n\n".format(folder, fits_headers['DATE']))
                        else:
                            with open(os.path.join(path, "{}_missing.txt".format(header)), 'a') as f:
                                f.write("Folder : {}\nDate: {}\n\n".format(folder, fits_headers['DATE']))

                        # Increment the missing counter.
                        data_dict[header] = np.nan
                        self.__missings[header] += 1

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

                # Query ASM and retrieve the seeing and coherence time.
                try:
                    # We add 15 minutes to the start and end of the observation to be sure to retrieve data.
                    # The observation can be shorter than the non-consistant delta time of the ASM.
                    start = Time(data_dict['OBS_STA']) - TimeDelta(900, format='sec')
                    stop = Time(data_dict['OBS_END']) + TimeDelta(900, format='sec')

                    # If the observation is before 2 april 2016, we use the old query.
                    # Else, we use the new query.
                    # with HiddenPrints():
                    if start < Time('2016-04-02T00:00:00.000', format='isot'):
                        print('Old query')
                        with HiddenPrints():
                            asm_data = qea.query_old_dimm(os.path.join(self.__path, folder), str(start), str(stop))
                        seeing = pd.DataFrame(asm_data['DIMM Seeing ["]'])
                        coherence_time = pd.DataFrame(asm_data['Tau0 [s]'])
                    else:
                        print('New query')
                        with HiddenPrints():
                            asm_data = qea.query_mass(os.path.join(self.__path, 'ESO ASM PARANAL/'), str(start), str(stop))
                        print('Seeing dataframe...')
                        seeing = pd.DataFrame(asm_data['MASS-DIMM Seeing ["]'])
                        print('Coherence dataframe...')
                        coherence_time = pd.DataFrame(asm_data['MASS-DIMM Tau0 [s]'])

                    print('Interpolate...')
                    # Interpolate the data to have a consistant step size of 1 minute.
                    seeing = self.__interpolate_dates(seeing, 'Date time', data_dict['OBS_STA'], data_dict['OBS_END'])
                    coherence_time = self.__interpolate_dates(coherence_time, 'Date time', data_dict['OBS_STA'], data_dict['OBS_END'])

                    print('Compute median and std...')
                    # Compute the median and std of the seeing and coherence time.
                    data_dict['SEEING_MEDIAN'] = seeing.median()
                    data_dict['SEEING_STD'] = seeing.std()
                    data_dict['COHERENCE_TIME_MEDIAN'] = coherence_time.median()
                    data_dict['COHERENCE_TIME_STD'] = coherence_time.std()

                    print("Done !")

                except:
                    self.__missings['SEEING_MEDIAN'] += 1
                    self.__missings['SEEING_STD'] += 1
                    self.__missings['COHERENCE_TIME_MEDIAN'] += 1
                    self.__missings['COHERENCE_TIME_STD'] += 1
                    data_dict['SEEING_MEDIAN'] = np.nan
                    data_dict['SEEING_STD'] = np.nan
                    data_dict['COHERENCE_TIME_MEDIAN'] = np.nan
                    data_dict['COHERENCE_TIME_STD'] = np.nan
              

                # Write the summary of the contrast in a file.
                with open(os.path.join(self.__path, folder, 'contrast_summary.txt'), 'w') as f:
                    try:
                        f.write(get_vector_summary_table(data_dict['NSIGMA_CONTRAST'], 'NSIGMA_CONTRAST'))
                    except:
                        print("Error while writing the contrast summary for the folder {}".format(folder))
                        print(data_dict['NSIGMA_CONTRAST'])

                data_dict_list.append(data_dict)

        # Write in a file the max date of the missing data along with the min date of the non missing data.
        with open(os.path.join(self.__path, 'missing_data_dates_summary.txt'), 'w') as f:
            for header in self.__header_list:
                if self.__missings[header] != 0:
                    f.write("{} : \nmax date of the missing values = {} \nmin date of the non missing values = {} \n\n".format(\
                        header, max(self.__dates_missing_dict[header]), min(self.__dates_not_missing_dict[header])))

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

    def __interpolate_dates(self, df, date_column, start, stop):
        """
        Interpolate the values of the dataframe between the start and stop dates.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe with the values to interpolate.

        date_column : str
            Name of the column with the dates.

        start : str
            Start date.

        stop : str
            Stop date.

        Returns
        -------
        df : pandas.DataFrame
            Dataframe with the interpolated values.
        """
        df = df.dropna()
        # Check if the index is already the dates
        if df.index.name == date_column:
            df = df.reset_index()
        
        lower, _ = find_lower_upper_bound(list(df[date_column]), Time(start))
        _, upper = find_lower_upper_bound(list(df[date_column]), Time(stop))

        # Drop the rows before and after the start and stop dates
        df = df.iloc[lower:upper+1]

        # Round the dates to the nearest minute
        df[date_column] = df[date_column].dt.round('min')

        # Set the date column as the index
        df = df.set_index(date_column)
        
        # Interpolate the values
        new_dates = pd.date_range(df.index[0], df.index[-1], freq='min')
        df = df.reindex(new_dates)
        df = df.interpolate(method='linear')

        return df