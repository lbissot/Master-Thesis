# Dataset Creation

## About this folder

The goal of this folder is to build the dataset which will eventually be fed to the deep learning model. All the files found here will serve this purpose.

## Sub-Folder(s)

### `Download Scripts`

Contains the shell scripts which when executed retrieve data from the sphere client.

## Files

### `sphere_dl_parser.py`

This file is used to remove undesired entries from the download scripts in order not to download an excessive amount of data.

### `dataset.py`

This file contains the class dataset along with its methods.

### `dataset_creation_utils.py`

This file contains all the utility functions used to build the dataset.

### `dataset_creation_toy.ipynb`

This file shows examples of how to use the dataset class.