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


def model_pipeline():

    # tell wandb to get started
    with wandb.init(project="master-thesis"):
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        # Define the device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Device : {}".format(device))
        config.device = device

        # Specify the name of the model in order to be able to load it later
        config.architecture = "MLP_vector_nsigma_input_size_{}".format(len(config.features_to_keep) - 2) + "_hidden_size_{}".format(config.hidden_size) + "_n_hidden_layers_{}".format(config.n_hidden_layers)
        # Get the date and time of the run to name the model
        now = datetime.datetime.now()
        config.model_name = "{}_{}_{}_{}_{}_{}_{}.pth".format(config.architecture, now.year, now.month, now.day, now.hour, now.minute, now.second)

        # Make the data
        print("Making the data...")
        x_train_tensor, y_train_log_tensor, \
            x_test_tensor, y_test_log_tensor, \
                x_valid_tensor, y_valid_tensor = make_data(df_AD, config)

        if x_train_tensor.shape[0] % config.batch_size != 0:
            raise ValueError("The number of observations in the training set ({}) must be a multiple of the batch size ({})".format(x_train_tensor.shape[0], config.batch_size))

        # Make the model, data, and optimization problem
        model, criterion, optimizer = make(config)

        # and use them to train the model
        print("Training the model...")
        val_loss = train(model, x_train_tensor, y_train_log_tensor, x_valid_tensor, y_valid_tensor, criterion, optimizer, config)
        wandb.log({"minimum validation loss": val_loss})
        print("Done training!")


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
    num_train = num_train - (num_train % config.batch_size)

    train = df_AD.sample(n=num_train, random_state=config.random_state)
    validation = df_AD.drop(train.index)
    test = validation.sample(frac=0.5, random_state=config.random_state)
    validation = validation.drop(test.index)

    # Transform the NaN values into the median of the column of the training set (only for the numerical features)
    imp = KNNImputer(n_neighbors=5, weights="uniform")
    train[numerical_features] = imp.fit_transform(train[numerical_features])
    test[numerical_features] = imp.transform(test[numerical_features])
    validation[numerical_features] = imp.transform(validation[numerical_features])

    # Split the data into features and labels
    x_train = train.drop(['NSIGMA_CONTRAST', 'SEPARATION'], axis=1) # Drop SEPARATION in order not to have a vector as input
    y_train = train['NSIGMA_CONTRAST']

    x_test = test.drop(['NSIGMA_CONTRAST', 'SEPARATION'], axis=1)
    y_test = test['NSIGMA_CONTRAST']

    x_validation = validation.drop(['NSIGMA_CONTRAST', 'SEPARATION'], axis=1)
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

    # Get the number of observations to keep
    n_points = config.n_obs_train

    # Convert the data to tensors
    x_train_tensor = torch.tensor(x_train[:n_points], dtype=torch.float32)
    y_train_tensor = torch.tensor(np.log10(y_train[:n_points]), dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(np.log10(y_test), dtype=torch.float32)
    x_validation_tensor = torch.tensor(x_validation, dtype=torch.float32)
    y_validation_tensor = torch.tensor(np.log10(y_validation), dtype=torch.float32)

    return x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor, x_validation_tensor, y_validation_tensor


def make(config):

    device = config.device

    # Make the model
    model = MLP(
        input_features=len(config.features_to_keep) - 2, # Don't forget to remove the target and the separation
        output_features=config.separation_size,
        hidden_features=config.hidden_size,
        num_hidden_layers=config.n_hidden_layers
        ).to(device)

    # Define the loss function and the optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    return model, criterion, optimizer


def shuffle(x, y, config):

    indices = np.random.permutation(x.shape[0])
    x_shuffled = x[indices]
    y_shuffled = y[indices]

    return x_shuffled, y_shuffled


def train_log(training_loss, validation_loss, step):
    # Where the magic happens
    wandb.log({"training loss": training_loss, "validation loss": validation_loss}, step=step)


def custom_lr_lambda_smooth(epoch, decay_rate, lr_0):
    return (1 / (1 + decay_rate * epoch)) * lr_0

def custom_lr_lambda_rough(epoch, decay_rate, lr_0, step_size):
    return lr_0 * (decay_rate ** (epoch // step_size))


def train(model, x, y, x_val, y_val, criterion, optimizer, config):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=10)

    device = config.device

    batch_ctr = 0

    custom_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: custom_lr_lambda_rough(epoch, decay_rate=config.decay_rate, lr_0=config.learning_rate, step_size=config.step_size))

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
            batch_losses.append(train_batch(x_batch, y_batch, model, optimizer, criterion, config).item())

            # Increment the batch counter
            batch_ctr += 1

        # Calculate the mean loss of the epoch
        training_loss = np.mean(batch_losses)

        # Run the model on some validation examples
        model.eval()
        with torch.no_grad():
            x_val, y_val = x_val.to(device), y_val.to(device)
            outputs = model(x_val)
            val_loss = criterion(outputs, y_val).item()
        
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

    return min_val_loss


def train_batch(x, y, model, optimizer, criterion, config):
    device = config.device

    x, y = x.to(device), y.to(device)
    
    # Forward pass
    outputs = model(x)
    loss = criterion(outputs, y)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Step with optimizer
    optimizer.step()

    return loss


def test(model, x, y, config):
    device = config.device

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

        # Calculate the mean squared and absolute errors for each observation
        for i in range(len(predictions)):
            mse.append(np.square(np.subtract(targets[i], predictions[i])))
            mae.append(np.abs(np.subtract(targets[i], predictions[i])))

        # Calculate the mean of the mean squared and absolute errors
        mean_mse = np.mean(mse)
        mean_mae = np.mean(mae)
        median_mse = np.median(mse)
        median_mae = np.median(mae)
        
        wandb.log({"testing_mean_MSE": mean_mse, "testing_median_MSE": median_mse, "testing_mean_MAE": mean_mae, "testing_median_MAE": median_mae})


if __name__ == "__main__":

    # print the python version
    print("Python version: {}".format(sys.version))

    print("Loading data...")
    wandb.login(key="816773503882553025c709792296c759ae384f60")

    # Load the data located in /Dataset_creation/df_AD.csv
    df_AD = pd.read_pickle('datasets/df_AD_timestamps.pkl')

    # Reset the index
    df_AD = df_AD.reset_index(drop=True)

    sweep_config = {
        'method': 'random',
        'metric': {
            'name': 'minimum validation loss',
            'goal': 'minimize'
        },
    }

    parameters_dict = {
        'epochs': {
            'value' : 1000,
        },
        'learning_rate': {
            'distribution' : 'uniform',
            'min': 0.001,
            'max': 0.01
        },
        'decay_rate': {
            'values': [0.9, 0.7, 0.5]
        },
        'batch_size': {
            'values': [1, 2, 4, 8]
        },
        'n_obs_train': {
            'value' : 1000,
        },
        'hidden_size': {
            'values': [128, 256, 512]
        },
        'n_hidden_layers': {
            'distribution' : 'int_uniform',
            'min': 5,
            'max': 25
        },
        'step_size': {
            'distribution' : 'int_uniform',
            'min': 5,
            'max': 20
        },
        'loss_function': {
            'value': 'mse'
        },
        'optimizer': {
            'value': 'adam'
        },
        'scale': {
            'value': 'log'
        },
        'stop_after_epochs_without_improvement': {
            'value': 25
        },
        'random_state': {
            'value': 42
        },
        'features_to_keep': {
            'value': ['ESO INS4 FILT3 NAME', 'ESO INS4 OPTI22 NAME', \
                'ESO AOS VISWFS MODE', 'ESO TEL AMBI WINDSP', 'ESO TEL AMBI RHUM', \
                    'HIERARCH ESO INS4 TEMP422 VAL', 'HIERARCH ESO TEL TH M1 TEMP', 'HIERARCH ESO TEL AMBI TEMP', \
                        'ESO DET NDIT', 'ESO DET SEQ1 DIT', 'SIMBAD_FLUX_G', 'SIMBAD_FLUX_H', 'SEEING_MEDIAN', \
                            'SEEING_STD', 'COHERENCE_TIME_MEDIAN', 'COHERENCE_TIME_STD', 'SCFOVROT', 'SEPARATION', 'NSIGMA_CONTRAST']
        },
        'categorical_features': {
            'value': ['ESO INS4 FILT3 NAME', 'ESO INS4 OPTI22 NAME', 'ESO AOS VISWFS MODE']
        },
        'separation_size': {
            'value': 124
        },
    }

    sweep_config['parameters'] = parameters_dict

    sweep_id = wandb.sweep(sweep_config, project="master-thesis")

    wandb.agent(sweep_id, function=model_pipeline, count=25)