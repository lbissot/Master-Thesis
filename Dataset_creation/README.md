# Dataset Creation

## About the folder

The goal of this folder is to build the dataset which will eventually be fed to the deep learning model. All the files found here will serve this purpose.

## Files

### `*.sh` files

Shell files which when executed retrieve data from the sphere client.

### `sphere_dl_parser.py`

This file is used to remove undesired entries from the download scripts in order not to download an excessive amount of data.

### `db_creation_utils.py`

This file contains all the utility functions used to build the dataset.

### `db_creation_toy.ipynb`

This file goal is to test implementation before putting functions in the other files thus being useless for other users.