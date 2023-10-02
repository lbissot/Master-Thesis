import pandas as pd
import numpy as np
import sys
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import argparse
import copy
import datetime

# Define the model
class MLP(nn.Sequential):
    def __init__(self, input_features, output_features, hidden_features, num_hidden_layers):
        layers = []
        layers.append(nn.Linear(input_features, hidden_features))
        layers.append(nn.ReLU())
        
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_features, hidden_features))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_features, output_features))
        
        super().__init__(*layers)


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


def model_pipeline(hyperparameters):

    # tell wandb to get started
    with wandb.init(project="master-thesis", config=run_config):
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        # make the data
        print("Making the data...")
        x_train_tensor, y_train_log_tensor, \
            x_test_tensor, y_test_log_tensor, \
                x_valid_tensor, y_valid_tensor = make_data(df_AD, config)

        if x_train_tensor.shape[0] % config.batch_size != 0:
            raise ValueError("The number of observations in the training set ({}) must be a multiple of the batch size ({})".format(x_train_tensor.shape[0], config.batch_size))

        # make the model, data, and optimization problem
        model, criterion, optimizer = make(config)
        # print(model)

        # and use them to train the model
        print("Training the model...")
        train(model, x_train_tensor, y_train_log_tensor, x_valid_tensor, y_valid_tensor, criterion, optimizer, config)
        print("Done training!")

        # load the best model according to the validation set
        best_model = MLP(
            input_features=len(config.features_to_keep) - 1, # Don't forget to remove the target feature
            output_features=1,
            hidden_features=config.hidden_size,
            num_hidden_layers=config.n_hidden_layers
            ).to(device)
        best_model.load_state_dict(torch.load(config.model_name))

        # and test its final performance
        print("Testing the model...")
        test(best_model, x_test_tensor, y_test_log_tensor, config)

    return best_model, x_train_tensor, y_train_log_tensor, x_test_tensor, y_test_log_tensor, x_valid_tensor, y_valid_tensor


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
    train = df_AD.sample(frac=0.8, random_state=config.random_state)
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

    # Convert the data to tensors
    x_train_tensor = torch.tensor(x_train[:n_points], dtype=torch.float32)
    y_train_tensor = torch.tensor(np.log10(y_train[:n_points]), dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(np.log10(y_test), dtype=torch.float32)
    x_validation_tensor = torch.tensor(x_validation, dtype=torch.float32)
    y_validation_tensor = torch.tensor(np.log10(y_validation), dtype=torch.float32)

    return x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor, x_validation_tensor, y_validation_tensor


def make(config):

    # Make the model
    model = MLP(
        input_features=len(config.features_to_keep) - 1, # Don't forget to remove the target feature
        output_features=1,
        hidden_features=config.hidden_size,
        num_hidden_layers=config.n_hidden_layers
        ).to(device)

    # Define the loss function and the optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    return model, criterion, optimizer


def shuffle(x, y, config):

    x_shuffled, y_shuffled = shuffle_sequences(x, y, config.separation_size // 4) # Shuffle observations or batches ???
    return x_shuffled, y_shuffled


def shuffle_sequences(X, y, seq_length):
    """
    Shuffle the order of the sequences of rows in matrix X and vector y.

    Parameters
    ----------
    X : torch.Tensor
        The matrix to shuffle
    y : torch.Tensor
        The target vector
    seq_length : int
        The length of the sequences
    """

    # Check whether X is a torch.Tensor
    if not isinstance(X, torch.Tensor):
        raise TypeError("X must be a torch.Tensor")

    # Check whether y is a torch.Tensor
    if not isinstance(y, torch.Tensor):
        raise TypeError("y must be a torch.Tensor")

    if not X.shape[0] % seq_length == 0:
        raise ValueError("The number of rows of X must be a multiple of seq_length")
    
    if not X.shape[0] == y.shape[0]:
        raise ValueError("X and y must have the same number of rows")
    
    seq_indices = torch.arange(0, X.shape[0], seq_length)
    seq_indices = seq_indices[np.random.permutation(len(seq_indices))]

    X_shuffled = torch.empty_like(X)
    y_shuffled = torch.empty_like(y)
    for i in range(len(seq_indices)):
        X_shuffled[i*seq_length:(i+1)*seq_length] = X[seq_indices[i]:seq_indices[i]+seq_length]
        y_shuffled[i*seq_length:(i+1)*seq_length] = y[seq_indices[i]:seq_indices[i]+seq_length]

    return X_shuffled, y_shuffled


def train_log(training_loss, validation_loss, step):
    # Where the magic happens
    wandb.log({"training loss": training_loss, "validation loss": validation_loss}, step=step)

def custom_lr_lambda_smooth(epoch, decay_rate, lr_0):
    return (1 / (1 + decay_rate * epoch)) * lr_0

def custom_lr_lambda_rough(epoch, decay_rate, lr_0):
    return lr_0 * (decay_rate ** (epoch // 10))

def train(model, x, y, x_val, y_val, criterion, optimizer, config):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=10)

    batch_ctr = 0

    custom_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: custom_lr_lambda_rough(epoch, decay_rate=config.decay_rate, lr_0=config.learning_rate))

    # Run training and track with wandb
    for epoch in range(config.epochs):

        # Shuffle the data
        x_shuffled, y_shuffled = shuffle(x, y, config)
        model.train()

        batch_losses = []
        
        for batch_start in range(0, len(x), config.batch_size):
            # Get a batch
            batch_end = batch_start + config.batch_size
            x_batch = x_shuffled[batch_start:batch_end]
            y_batch = y_shuffled[batch_start:batch_end]

            # Train the model
            batch_losses.append(train_batch(x_batch, y_batch, model, optimizer, criterion).item())

            # Increment the batch counter
            batch_ctr += 1

        # Calculate the mean loss of the epoch
        training_loss = np.mean(batch_losses)

        # Run the model on some validation examples
        model.eval()
        with torch.no_grad():
            x_val, y_val = x_val.to(device), y_val.to(device)
            outputs = model(x_val)
            val_loss = criterion(outputs, y_val.view(-1, 1)).item()
        
        # Log metrics
        if epoch % 10 == 0:
            print("Epoch: {} | Training loss: {} | Validation loss: {}".format(epoch, training_loss, val_loss))
        train_log(training_loss, val_loss, epoch)

        # Update the scheduler at the end of each epoch
        custom_scheduler.step()

        # Stopping condition
        if epoch == 0 or val_loss < min_val_loss:
            min_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), config.model_name)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement == config.stop_after_epochs_without_improvement:
                print("Early stopping")
                break


def train_batch(x, y, model, optimizer, criterion):
    x, y = x.to(device), y.to(device)
    
    # Forward pass
    outputs = model(x)
    loss = criterion(outputs, y.view(-1, 1))
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Step with optimizer
    optimizer.step()

    return loss


def test(model, x, y, config):
    model.eval()

    # Run the model on some test examples
    with torch.no_grad():
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        
        # Get the numpy versions of the predictions and the targets
        predictions = outputs.cpu().numpy()
        targets = y.cpu().numpy()

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
        
        wandb.log({"testing_mean_MSE": mean_mse, "testing_median_MSE": median_mse, "testing_mean_MAE": mean_mae, "testing_median_MAE": median_mae})


def plot_predictions(model, X, y, idx):
    separation = df_AD['SEPARATION'].iloc[idx]
    
    
    start = idx * len(separation)
    stop = start + len(separation)

    prediction = model(X).detach().numpy()
    prediction = prediction[start:stop]
    contrast = y[start:stop].detach().numpy()
    separation = X[start:stop, 1].detach().numpy()

    plt.plot(separation, contrast, color='blue', label='Actual')
    plt.plot(separation, prediction, color='red', label='Predicted')
    plt.xlabel('Separation (arcsec)')
    plt.ylabel('Contrast (5-sigma)')
    plt.legend()
    plt.show()

    
# Define the argparse setup
def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a neural network to predict the nsigma_contrast')

    # Add arguments
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='Initial learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.9, help='Decay rate of the learning rate')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size (number of observations per batch)')
    parser.add_argument('--n_obs_train', type=int, default=10, help='Number of observations to train on')
    parser.add_argument('--hidden_size', type=int, default=512, help='Hidden size')
    parser.add_argument('--n_hidden_layers', type=int, default=10, help='Number of hidden layers')
    parser.add_argument('--features_to_keep', nargs='+', default=['ESO INS4 FILT3 NAME', 'ESO INS4 OPTI22 NAME', \
        'ESO AOS VISWFS MODE', 'ESO TEL AMBI WINDSP', 'ESO TEL AMBI RHUM', \
            'HIERARCH ESO INS4 TEMP422 VAL', 'HIERARCH ESO TEL TH M1 TEMP', 'HIERARCH ESO TEL AMBI TEMP', \
                'ESO DET NDIT', 'ESO DET SEQ1 DIT', 'SIMBAD_FLUX_G', 'SIMBAD_FLUX_H', 'SEEING_MEDIAN', \
                    'SEEING_STD', 'COHERENCE_TIME_MEDIAN', 'COHERENCE_TIME_STD', 'SCFOVROT', 'SEPARATION', 'NSIGMA_CONTRAST'], 
                    help='List of features to keep')
    
    return parser.parse_args()


if __name__ == "__main__":

    # Parse command-line arguments
    args = parse_arguments()

    # print the python version
    print("Python version: {}".format(sys.version))

    print("Loading data...")
    wandb.login(key="816773503882553025c709792296c759ae384f60")

    # Load the data located in /Dataset_creation/df_AD.csv
    df_AD = pd.read_pickle('Dataset_creation/df_AD_timestamps.pkl')

    # Reset the index
    df_AD = df_AD.reset_index(drop=True)

    run_config = dict(
        # Model
        epochs = args.epochs,
        learning_rate = args.learning_rate,
        decay_rate = args.decay_rate,
        batch_size = args.batch_size, 
        n_obs_train = args.n_obs_train,
        loss_function = 'mse',
        optimizer = 'adam',
        architecture = 'MLP_single_nsigma',
        hidden_size = args.hidden_size,
        n_hidden_layers = args.n_hidden_layers,
        scale = 'log',
        stop_after_epochs_without_improvement = 20,
        model_name = "",
        
        # Data
        random_state = 42,
        features_to_keep = args.features_to_keep,
        categorical_features = ['ESO INS4 FILT3 NAME', 'ESO INS4 OPTI22 NAME', 'ESO AOS VISWFS MODE'],
        separation_size = 124,  
    )

    run_config['architecture'] += "_input_size_{}".format(len(run_config['features_to_keep']) - 1)
    run_config['architecture'] += "_hidden_size_{}".format(run_config['hidden_size'])
    run_config['architecture'] += "_n_hidden_layers_{}".format(run_config['n_hidden_layers'])

    # Get the date and time of the run to name the model
    now = datetime.datetime.now()
    run_config['model_name'] = "{}_{}_{}_{}_{}_{}_{}.pth".format(run_config['architecture'], now.year, now.month, now.day, now.hour, now.minute, now.second)

    # Change the batch size to be the number of points per batch instead of the number of observations per batch
    run_config['batch_size'] = run_config['batch_size'] * run_config['separation_size']

    # Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device : {}".format(device))

    # Build, train and analyze the model with the pipeline
    model, \
        x_train_tensor, y_train_log_tensor, \
            x_test_tensor, y_test_log_tensor, \
                x_valid_tensor, y_valid_tensor = model_pipeline(run_config)

    # Save the model
    torch.save(model.state_dict(), run_config['model_name'])