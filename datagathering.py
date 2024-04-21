import numpy as np
import pandas as pd

# Path to the CSV file
csv_file = 'Dataset For Forecasting Assignment.csv'

def read_csv_file(csv_file):
    # Read the CSV file into a dataframe
    df = pd.read_csv(csv_file)

    # Add datetime index
    df['Date'] = pd.to_datetime(df['Date'], format="%d/%m/%Y %H:%M")
    df.set_index('Date', inplace=True)

    # Convert all columns to numeric
    df = df.apply(pd.to_numeric, errors='coerce')

    return df


# Prepare the training and testing data
def prepare_data(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps])
        y.append(data[i+time_steps])
    return np.array(X), np.array(y)

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
