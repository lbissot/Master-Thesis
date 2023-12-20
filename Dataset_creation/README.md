# Dataset Creation

## About this folder

The primary objective of this folder is to facilitate the construction of a comprehensive dataset, intended for training a deep learning model. All the components found in this repository collectively contribute to this crucial step in the machine learning pipeline.

## Sub-Folder(s)

### `Download Scripts`

Contained within this sub-directory are shell scripts responsible for retrieving data from the SPHERE client. Executing these scripts automates the process of fetching the necessary data for subsequent processing.

### `SPHERE_DC_DATA``

Contains the subfolders `contrast_curves` and `timestamps` which are the folders where the (raw) data from the SPHERE client is downloaded.

## Files

### `sphere_dl_parser.py`

This file is used to remove undesired entries from the download scripts in order not to download an excessive amount of data.

### `dataset.py`

This file contains the class dataset along with its methods.

### `dataset_creation_utils.py`

This file contains all the utility functions used to build the dataset.

### `dataset_creation_toy.ipynb`

For a hands-on understanding of how to use the `dataset` class, refer to this Jupyter notebook. It provides practical examples and demonstrations, making it a valuable resource for users navigating the intricacies of dataset creation.