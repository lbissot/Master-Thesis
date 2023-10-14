import sys 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import argparse
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import time

def reshape_dataframe(df, vec_column_name_list):
    """
    Reshape a dataframe containing vectors in order to have one row per element of the vectors.
    The metadata of the row is repeated for each element of the vectors.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to reshape
    vec_column_name_list : list
        The list of the columns containing vectors

    Returns
    -------
    df_reshaped : pandas.DataFrame
        The reshaped dataframe
    """

    columns = df.columns
    reshape_dataframe = {}

    # Initialize the dictionary
    for column in columns:
        reshape_dataframe[column] = []

    # Fill the dictionary
    for index, row in df.iterrows():
        # Create a dictionnary with only the metadata of the row, that is without the columns in vec_column_name_list
        meta_dict = row.drop(vec_column_name_list).to_dict()

        # Check whether the dimensions of the vectors are the same
        for column in vec_column_name_list:
            if len(row[column]) != len(row[vec_column_name_list[0]]):
                raise ValueError("The dimensions of the vectors are not the same")

        for i in range(len(row[vec_column_name_list[0]])):
            # Add the metadata to the dictionary
            for key, value in meta_dict.items():
                reshape_dataframe[key].append(value)

            # Add the vector to the dictionary
            for column in vec_column_name_list:
                reshape_dataframe[column].append(row[column][i])

    # Create a dataframe from the dictionary
    df_reshaped = pd.DataFrame(reshape_dataframe)
    return df_reshaped


def model_pipeline(run_config):

    # Config must have the dictionnary keys as attributes
    config = argparse.Namespace(**run_config)
    
    print("Building the model and the data ...")
    models = make(config)
    x_train, y_train, x_test, y_test, x_validation, y_validation = make_data(df_AD, config)

    # Train the model
    print("Training the models ...")
    # time the training
    start = time.time()
    val_loss = train(models, x_train, y_train, x_validation, y_validation, config)
    end = time.time()
    print("Training time: {}".format(end - start))

    # Load the best model
    print("Loading the best model ...")
    with open(config.model_name, 'rb') as file:
        best_model = pickle.load(file)

    # Test the model
    # print("Testing the model ...")
    # mean_mse, mean_mae, median_mse, median_mae = test(best_model, x_test, y_test, config)
    # print("Mean MSE: {}".format(mean_mse))
    # print("Mean MAE: {}".format(mean_mae))
    # print("Median MSE: {}".format(median_mse))
    # print("Median MAE: {}".format(median_mae))

    # # Plot the predictions
    # plot_predictions(best_model, x_test, y_test, 0)

    return best_model, x_train, y_train, x_test, y_test, x_validation, y_validation


def make_data(df_AD, config):

    # Check whether df_AD contains 'SEPARATION' and 'NSIGMA_CONTRAST'
    if 'SEPARATION' not in df_AD.columns:
        raise ValueError("df_AD must contain the column 'SEPARATION'")
    if 'NSIGMA_CONTRAST' not in df_AD.columns:
        raise ValueError("df_AD must contain the column 'NSIGMA_CONTRAST'")

    # The numerical features are the ones that are not categorical
    numerical_features = list(set(config.features_to_keep) - set(config.categorical_features) - set(['SEPARATION', 'NSIGMA_CONTRAST']))

    # Get a dataframe containing only the columns we want to keep
    df_AD = df_AD[config.features_to_keep]

    # Convert the strings to numbers in order to transform it into a tensor
    df_AD.loc[:, 'ESO INS4 FILT3 NAME'] = pd.factorize(df_AD['ESO INS4 FILT3 NAME'])[0]
    df_AD.loc[:, 'ESO INS4 OPTI22 NAME'] = pd.factorize(df_AD['ESO INS4 OPTI22 NAME'])[0]
    df_AD.loc[:, 'ESO AOS VISWFS MODE'] = pd.factorize(df_AD['ESO AOS VISWFS MODE'])[0]

    # Get values of separation between 0 and 1
    separation = np.array(df_AD['SEPARATION'][0])
    separation = separation / np.max(separation)

    # Replace every separation vectors with the new one
    df_AD.loc[:, 'SEPARATION'] = df_AD['SEPARATION'].apply(lambda x: np.array(x) / np.max(x))

    # Split the data into train, validation and test sets
    num_obs = len(df_AD)
    num_train = int(0.8 * num_obs)
    num_train = num_train - (num_train % 8)

    train = df_AD.sample(n=num_train, random_state=config.random_state)
    validation = df_AD.drop(train.index)
    test = validation.sample(frac=0.5, random_state=config.random_state)
    validation = validation.drop(test.index)

    # Transform the NaN values into the median of the column of the training set (only for the numerical features)
    imp = KNNImputer(n_neighbors=5, weights='uniform')
    train[numerical_features] = imp.fit_transform(train[numerical_features])
    test[numerical_features] = imp.transform(test[numerical_features])
    validation[numerical_features] = imp.transform(validation[numerical_features])

    # Reshape the dataframes (note that the reshape is done after the train/test split to avoid data leakage)
    train = reshape_dataframe(train, ['SEPARATION', 'NSIGMA_CONTRAST'])
    test = reshape_dataframe(test, ['SEPARATION', 'NSIGMA_CONTRAST'])
    validation = reshape_dataframe(validation, ['SEPARATION', 'NSIGMA_CONTRAST'])

    # Split the data into features and labels
    x_train = train.drop(['NSIGMA_CONTRAST'], axis=1)
    y_train = train['NSIGMA_CONTRAST']

    x_test = test.drop(['NSIGMA_CONTRAST'], axis=1)
    y_test = test['NSIGMA_CONTRAST']

    x_validation = validation.drop(['NSIGMA_CONTRAST'], axis=1)
    y_validation = validation['NSIGMA_CONTRAST']

    # Standardize the data
    scaler = StandardScaler()
    x_train[numerical_features] = scaler.fit_transform(x_train[numerical_features])
    x_test[numerical_features] = scaler.transform(x_test[numerical_features])
    x_validation[numerical_features] = scaler.transform(x_validation[numerical_features])

    # Convert the dataframes to numpy arrays
    x_train = (x_train.values).astype(np.float32)
    y_train = np.array(y_train.tolist()).astype(np.float32)
    x_test = (x_test.values).astype(np.float32)
    y_test = np.array(y_test.tolist()).astype(np.float32)
    x_validation = (x_validation.values).astype(np.float32)
    y_validation = np.array(y_validation.tolist()).astype(np.float32)

    # Get the number of points to keep
    n_points = config.n_obs_train * len(separation)

    return x_train[:n_points], np.log10(y_train[:n_points]), x_test, np.log10(y_test), x_validation, np.log10(y_validation)


def train(models, x_train, y_train, x_validation, y_validation, config):

    # Train the models
    for model in models:
        print("Training the model with max_features = {} ...".format(model.max_features))
        model.fit(x_train, y_train)
        print("Done")
    
    print("Selecting the best model ...")
    # Evaluate the models and keep the best one
    best_model = None
    best_mse = np.inf

    for model in models:
        y_pred = model.predict(x_validation)
        mse = mean_squared_error(y_validation, y_pred)
        print("MSE on the validation set with max_features = {} : {}".format(model.max_features, mse))

        if mse <= best_mse:
            best_mse = mse
            best_model = model
            config.max_features = model.max_features
            # Save the model
            with open(config.model_name, 'wb') as file:
                pickle.dump(best_model, file)

    print("Best MSE on the validation set : {}".format(best_mse))
    return best_mse


def test(model, x, y, config):

    # Run the model on some test examples
    predictions = model.predict(x)
    targets = y

    mse = []
    mae = []

    separation_size = config.separation_size

    # Calculate the mean squared and absolute errors for each observation
    for i in range(len(predictions) // separation_size):
        start = i * separation_size
        stop = start + separation_size
        mse.append(np.square(np.subtract(targets[start:stop], predictions[start:stop])))
        mae.append(np.abs(np.subtract(targets[start:stop], predictions[start:stop])))

    # Calculate the mean of the mean squared and absolute errors
    mean_mse = np.mean(mse)
    mean_mae = np.mean(mae)
    median_mse = np.median(mse)
    median_mae = np.median(mae)

    return mean_mse, mean_mae, median_mse, median_mae
        

def plot_predictions(model, X, y, idx):
    separation = df_AD['SEPARATION'].iloc[idx]
    
    
    start = idx * len(separation)
    stop = start + len(separation)

    prediction = model.predict(X)
    prediction = prediction[start:stop]
    contrast = y[start:stop]

    plt.plot(separation, contrast, color='blue', label='Actual')
    plt.plot(separation, prediction, color='red', label='Predicted')
    plt.xlabel('Separation (arcsec)')
    plt.ylabel('Contrast (5-sigma)')
    plt.legend()
    plt.show()


def make(config):

    # Make the model
    models = []
    for max_feature in config.max_features:
        model = RandomForestRegressor(n_estimators=config.n_estimators, max_features=max_feature, random_state=config.random_state)
        models.append(model)

    return models


if __name__ == "__main__":

    # print the python version
    print("Python version: {}".format(sys.version))

    # Load the data located in /Dataset_creation/df_AD.csv
    df_AD = pd.read_pickle('Dataset_creation/df_AD_timestamps.pkl')

    # Reset the index
    df_AD = df_AD.reset_index(drop=True)

    run_config = dict(
        # Model
        n_obs_train = 1000,
        n_estimators = 500,
        max_features = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
        loss_function = 'mse',
        architecture = 'RF_single_nsigma',
        scale = 'log',
        model_name = "",
        
        # Data
        random_state = 42,
        features_to_keep = ['ESO INS4 FILT3 NAME', 'ESO INS4 OPTI22 NAME', \
            'ESO AOS VISWFS MODE', 'ESO TEL AMBI WINDSP', 'ESO TEL AMBI RHUM', \
                'HIERARCH ESO INS4 TEMP422 VAL', 'HIERARCH ESO TEL TH M1 TEMP', 'HIERARCH ESO TEL AMBI TEMP', \
                    'ESO DET NDIT', 'ESO DET SEQ1 DIT', 'SIMBAD_FLUX_G', 'SIMBAD_FLUX_H', 'SEEING_MEDIAN', \
                        'SEEING_STD', 'COHERENCE_TIME_MEDIAN', 'COHERENCE_TIME_STD', 'SCFOVROT', 'SEPARATION', 'NSIGMA_CONTRAST'],
        categorical_features = ['ESO INS4 FILT3 NAME', 'ESO INS4 OPTI22 NAME', 'ESO AOS VISWFS MODE'],
        separation_size = 124,  
    )

    # Get the date and time of the run to name the model
    now = datetime.datetime.now()
    run_config['model_name'] = "{}_{}_{}_{}_{}_{}_{}.pth".format(run_config['architecture'], now.year, now.month, now.day, now.hour, now.minute, now.second)

    # Build, train and analyze the model with the pipeline
    model, \
        x_train, y_train, \
            x_test, y_test, \
                x_valid, y_valid = model_pipeline(run_config)

    run_config['max_features'] = model.max_features