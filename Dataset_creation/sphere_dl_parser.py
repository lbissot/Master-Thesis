#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script is used to remove undisired files from the sphere dl script.
"""

import os

DEBUG = True

# Path to the folder containing the sphere_dl_script
path_data = 'C:/Users/ludin/Documents/Master Thesis/Dataset_creation/Download Scripts/'
target_name = 'sphere_dl_script_contrast_curves.sh'

# List of desired files
file_name_prefix = 'ird_specal_dc-IRD_SPECAL_'
file_name_suffixes = ['JSON-reduction.json', 'PNG-reduced_image.png', \
    'MEDIAN_RESIDUAL_STACK-reduced_image_median.fits', 'CONTRAST_CURVE_TABLE-contrast_curve_tab.fits']

filenames = [file_name_prefix + suffix for suffix in file_name_suffixes]


if not os.path.exists(path_data):
    exit('ERROR! Folder path_data does not exist.')

if not os.path.exists(os.path.join(path_data, target_name)):
    exit('ERROR! target_name does not exist.')

# Open the target file and get the list of files to download
with open(os.path.join(path_data, target_name), 'r') as f:
    if DEBUG:
        print('Reading file: {}'.format(target_name))
    lines = f.readlines()

parsed_lines = []

# Keep the desired files, remove the others
for line in lines:
    if not line.startswith('wget'):
        parsed_lines.append(line)
    
    else:
        # Get the file path
        line_split = line.split('"')
        file_path = line_split[1]

        # Get the file name
        file_path_split = file_path.split('/')
        file_name = file_path_split[-1]

        # Check if the file is desired
        if file_name in filenames:
            parsed_lines.append(line)       

# Save the new file
with open(os.path.join(path_data, "parsed_sphere_dl_script_contrast_curves.sh"), 'w') as f:
    if DEBUG:
        print('Writing file: {}'.format("parsed_sphere_dl_script.sh"))

    for line in parsed_lines:
        f.write(line)