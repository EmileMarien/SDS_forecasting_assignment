from typing import Tuple
import numpy as np
import pandas as pd

# Path to the CSV file
csv_file = 'Dataset For Forecasting Assignment.csv'

def read_csv_file(csv_file):
    """
    Read the CSV file into a pandas dataframe.
    :param csv_file: The path to the CSV file.
    :return: The dataframe containing the data from the CSV file.
    """
    # Read the CSV file into a dataframe
    df = pd.read_csv(csv_file)

    # Add datetime index
    df['Date'] = pd.to_datetime(df['Date'], format="%d/%m/%Y %H:%M")
    df.set_index('Date', inplace=True)

    # Convert all columns to numeric
    df = df.apply(pd.to_numeric, errors='coerce')

    return df


def remove_outliers(data, column_name):
    """
    Remove outliers from the data based on the specified column.
    Outliers are defined as values that are more than 5 standard deviations
    away from the mean.
    
    Parameters:
    - data: the input data
    - column_name: the name of the column to remove outliers from
    
    Returns:
    - the data with outliers removed
    """
    # Calculate the mean and standard deviation
    mean = np.mean(data[column_name])
    std = np.std(data[column_name])

    # Define the number of standard deviations for outlier detection
    n_std = 5

    # Replace outliers with the mean plus/minus n_std times the standard deviation
    data.loc[data[column_name] >= mean + n_std*std, column_name] = mean + n_std*std
    data.loc[data[column_name] <= mean - n_std*std, column_name] = mean - n_std*std

    return data

def split_train_val_test(data: pd.DataFrame) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    Split the data into training, validation, and test sets in chronological order and predefined fractions.
    :param data: The data to split.
    :return: The training, validation, and test sets, and the forecast x values.
    """
    time_steps = 24
    X = data[['Load_FR', 'Gen_FR', 'Price_CH', 'Wind_BE', 'Solar_BE','Load_BE']]
    y = data['Price_BE']

    # Define the start and end dates
    start_date = pd.to_datetime('2021-01-01 00:00')
    end_date = pd.to_datetime('2024-01-22 23:00')

    # Calculate the total number of hours between the start and end dates
    total_hours = (end_date - start_date).total_seconds() / 3600

    # Calculate the number of hours for each set
    train_hours = int(total_hours * 0.7)
    val_hours = int(total_hours * 0.2)
    test_hours = int(total_hours * 0.1)

    # Adjust the hours for each set to the nearest multiple of time_steps
    train_hours = (train_hours // time_steps) * time_steps
    val_hours = (val_hours // time_steps) * time_steps
    test_hours = (test_hours // time_steps) * time_steps

    # Split the data based on the number of hours
    X_train = X[(X.index >= start_date) & (X.index < start_date + pd.Timedelta(hours=train_hours))]
    y_train = y[(y.index >= start_date) & (y.index < start_date + pd.Timedelta(hours=train_hours))]

    X_val = X[(X.index >= start_date + pd.Timedelta(hours=train_hours)) & 
                    (X.index < start_date + pd.Timedelta(hours=train_hours + val_hours))]
    y_val = y[(y.index >= start_date + pd.Timedelta(hours=train_hours)) & 
                    (y.index < start_date + pd.Timedelta(hours=train_hours + val_hours))]

    X_test = X[(X.index >= start_date + pd.Timedelta(hours=train_hours + val_hours)) & 
                    (X.index < start_date + pd.Timedelta(hours=train_hours + val_hours + test_hours))]
    y_test = y[(y.index >= start_date + pd.Timedelta(hours=train_hours + val_hours)) & 
                    (y.index < start_date + pd.Timedelta(hours=train_hours + val_hours + test_hours))]

    X_forecast = X[(X.index >= end_date)].values
    # Check the dimensions of the data
    #print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)

    # Scale the data #TODO: check if scaling is necessary
    #scaler = MinMaxScaler()
    #X_train_scaled = scaler.fit_transform(X_train)
    #X_val_scaled = scaler.transform(X_val)
    #X_test_scaled = scaler.transform(X_test)

    # Reshape the input data
    X_train_reshaped = X_train.values.reshape(-1, X_train.shape[1])
    X_val_reshaped = X_val.values.reshape(-1, X_val.shape[1])
    X_test_reshaped = X_test.values.reshape(-1, X_test.shape[1])

    # Reshape the output data
    y_train = y_train.values
    y_val = y_val.values

    #print('input_features ' + str(X_train_reshaped.shape)+str(X_val_reshaped.shape)+str(X_test_reshaped.shape))
    #print('target dimensions ' + str(y_train.shape)+str(y_val.shape))
    return (X_train_reshaped, y_train), (X_val_reshaped, y_val), (X_test_reshaped, y_test), X_forecast


def split_train_val_test_random(data: pd.DataFrame, train_frac:float=0.7,val_frac:float=0.2)-> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    Split the data into training, validation, and test sets in random order.
    :param data: The data to split.
    :param train_frac: The fraction of the data to use for training.
    :param val_frac: The fraction of the data to use for validation.
    :return: The training, validation, and test sets as tuples of numpy arrays, and the forecast x values as numpy array.
    """
    # Define the start and end dates
    start_date = pd.to_datetime('2021-01-01 00:00')
    end_date = pd.to_datetime('2024-01-22 23:00')
    
    x_forecast=data[(data.index >= end_date)][['Load_FR', 'Gen_FR', 'Price_CH', 'Wind_BE', 'Solar_BE','Load_BE']]

    data=data[(data.index >= start_date) & (data.index < end_date)]
    train, validate, test = np.split(data.sample(frac=1, random_state=42), [int(train_frac*len(data)), int((train_frac+val_frac)*len(data))])

    x_train = train[['Load_FR', 'Gen_FR', 'Price_CH', 'Wind_BE', 'Solar_BE','Load_BE']]
    y_train = train['Price_BE'].values

    x_val = validate[['Load_FR', 'Gen_FR', 'Price_CH', 'Wind_BE', 'Solar_BE','Load_BE']]
    y_val = validate['Price_BE'].values

    x_test = test[['Load_FR', 'Gen_FR', 'Price_CH', 'Wind_BE', 'Solar_BE','Load_BE']]
    y_test = test['Price_BE'].values

    return (x_train, y_train), (x_val, y_val), (x_test, y_test), x_forecast

