#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script is used to remove undisired files from the sphere dl script.
"""

import os

DEBUG = True
# Choose the mode : contrast_curves or timestamps
mode = 'contrast_curves'

# Path to the folder containing the sphere_dl_script
path_data = 'C:/Users/ludin/Documents/Master Thesis/Dataset_creation/Download Scripts/'

if mode == 'contrast_curves':
    target_name = 'sphere_dl_script_contrast_curves.sh'
    new_file_name = "parsed_sphere_dl_script_contrast_curves.sh"
    file_name_prefix = 'ird_specal_dc-IRD_SPECAL_'
    file_name_suffixes = ['JSON-reduction.json', 'PNG-reduced_image.png', \
        'MEDIAN_RESIDUAL_STACK-reduced_image_median.fits', 'CONTRAST_CURVE_TABLE-contrast_curve_tab.fits']

elif mode == 'timestamps':
    target_name = 'sphere_dl_script_timestamps.sh'
    new_file_name = "parsed_sphere_dl_script_timestamps.sh"
    file_name_prefix = 'ird_convert_recenter_dc5-'
    file_name_suffixes = ['IRD_TIMESTAMP-timestamp.fits']

else:
    exit('ERROR! mode is not valid.')

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
    if line.startswith('wget'):
        # Get the file path
        line_split = line.split('"')
        file_path = line_split[1]

        # Get the file name
        file_path_split = file_path.split('/')
        file_path_split.insert(1, mode)
        file_name = file_path_split[-1]

        # Check if the file is desired
        if file_name in filenames:
            # Rebuild the new path (add the mode as subfolder)
            new_path = '/'.join(file_path_split)
            # Rebuild the new line
            new_line = ''
            for i in range(len(line_split)):
                if i == 1:
                    new_line += new_path
                else:
                    new_line += line_split[i]

                if i != len(line_split) - 1:
                    new_line += '"'

            parsed_lines.append(new_line)

    elif line.startswith('echo "Directory: ') or line.startswith('mkdir'):
        # Get the file path
        line_split = line.split('"')
        file_path = line_split[1] # Careful for echo, there is "Directory: " before the file path

        # Insert the mode as subfolder in the file path
        file_path_split = file_path.split('/')
        file_path_split.insert(1, mode)

        # Rebuild the new path
        new_path = '/'.join(file_path_split)

        # Rebuild the new line
        new_line = line_split[0] + '"' + new_path + '"' + line_split[2]

        parsed_lines.append(new_line)

    else:
        parsed_lines.append(line)     

# Save the new file
with open(os.path.join(path_data, new_file_name), 'w') as f:
    if DEBUG:
        print('Writing file: {}'.format(new_file_name))

    for line in parsed_lines:
        f.write(line)