from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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

def prepare_train_test_forecast(data:pd.DataFrame, test_size:float=0.33)->Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare the training and test data for the model.
    :param data: The data to prepare.
    :param test_size: The size of the test set (<1)
    
    :return: x_train, y_train, x_test, y_test and x_forecast as Numpy arrays.
    """
    """
    # Define the start and end dates
    start_date = pd.to_datetime('2021-01-01 00:00')
    end_date = pd.to_datetime('2024-01-22 23:00')
    
    x_forecast=data[(data.index > end_date)][['Load_FR', 'Gen_FR', 'Price_CH', 'Wind_BE', 'Solar_BE','Load_BE']]

    data=data[(data.index >= start_date) & (data.index <= end_date)]
    x=data[['Load_FR', 'Gen_FR', 'Price_CH', 'Wind_BE', 'Solar_BE','Load_BE']]
    y=data['Price_BE']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42,shuffle=shuffle)

    
    """


    ## New
    data.resample('h').mean()
    n_hours=24

    # Reduce data so it is a multiple of n_hours
    data = data.iloc[(len(data) % n_hours):]
    # Define the start and end dates
    #start_date = pd.to_datetime('2021-01-02 00:00') 
    #end_date = pd.to_datetime('2024-01-6 23:00')
    #data=data[start_date:end_date]

    # Shift columns to create features


    shifts = {
        'Price_BE': {
            'Price_BE1': n_hours*1,
            'Price_BE2': n_hours*2,
            #'Price_BE3': n_hours*3,
            #'Price_BE4': n_hours*4,
            #'Price_BE5': n_hours*5,
            #'Price_BE6': n_hours*6,
            #'Price_BE7': n_hours*7,
        },
        'Load_FR': {
            #'Load_FR0': n_hours*0,
            #'Load_FR1': n_hours*1,
            #'Load_FR2': n_hours*2,
            #'Load_FR6': n_hours*6,
        },
        'Wind_BE': {
            #'Wind_BE0': n_hours*0,
            #'Wind_BE1': n_hours*1,
            #'Wind_BE2': n_hours*2,
            #'Wind_BE6': n_hours*6,
        },
        'Solar_BE': {
            #'Solar_BE0': n_hours*0,
            #'Solar_BE1': n_hours*1,
            #'Solar_BE2': n_hours*2,
            #'Solar_BE3': n_hours*3,
            #'Solar_BE4': n_hours*4,
            #'Solar_BE5': n_hours*5,
            #'Solar_BE6': n_hours*6,
            #'Solar_BE7': n_hours*7,
        },
        'Gen_FR': {
            #'Gen_FR0': n_hours*0,
            #'Gen_FR1': n_hours*1,
            #'Gen_FR2': n_hours*2,
            #'Gen_FR6': n_hours*6,
            #'Gen_FR7': n_hours*7,
        },
        'Price_CH': {            
            'Price_CH0': n_hours*0,
            'Price_CH1': n_hours*1,
            #'Price_CH2': n_hours*2,
            #'Price_CH3': n_hours*3,
            #'Price_CH4': n_hours*4,
            #'Price_CH5': n_hours*5,
            #'Price_CH6': n_hours*6,
            #'Price_CH7': n_hours*7,
        },
        'Load_BE': {
            #'Load_BE0': n_hours*0,
            #'Load_BE1': n_hours*1,
            #'Load_BE6': n_hours*6,
            #'Load_BE7': n_hours*7,
        }
    }

    new_columns = []
    # Shift columns to create
    for col, shifts in shifts.items():
        for new_col, shift in shifts.items():
            data.loc[:, new_col] = data[col].shift(shift)
            new_columns.append(new_col)
    
    features_shifted=pd.concat([data[col] for col in new_columns], axis=1)
                #'Load_FR': n_hours*2,  'Load_FR': n_hours*7, 
            #'Solar_BE': n_hours*1, 'Solar_BE': n_hours*2, 'Solar_BE': n_hours*3, 'Solar_BE': n_hours*4, 'Solar_BE': n_hours*5, 'Solar_BE': n_hours*6, 'Solar_BE': n_hours*7,   
              
    features=features_shifted.dropna()
    indices_train_test = features.index[:-n_hours]
    indices_forecast = features.index[-n_hours:]
    features_reshaped=[]
    print('features selected:  ',features.columns)
    for col in features.columns:
        features_reshaped.append(features[col].values.reshape(-1,n_hours))


    rows = features_reshaped[0].shape[0]-1 # number of entries in the training data
    col = features_reshaped[0].shape[1] # length of an entry of one feature in the training data
    X = np.zeros((rows,len(features_reshaped)*col))

    for i in range(rows):
        for j in range(len(features_reshaped)):
            X[i,j*col:(j+1)*col]=features_reshaped[j][i,:]


    X_forecast = np.zeros((1,len(features_reshaped)*col))
    for j in range(len(features_reshaped)):
        X_forecast[0,j*col:(j+1)*col]=features_reshaped[j][-1,:]
    
    Y = data['Price_BE'].loc[indices_train_test].values.reshape(-1, n_hours)
    
    
    print(X.shape)
    print(Y.shape)
    #print(X_forecast.shape)
    #print(indices_train_test.shape)

    #print(X.shape)
    split_index_X = int((1 - test_size) * len(X))
    split_index_indices= split_index_X*24#int((1 - test_size) * len(indices_train_test))
    #print(indices_train_test.shape)
    x_train, x_test = X[:split_index_X], X[split_index_X:]
    y_train, y_test = Y[:split_index_X], Y[split_index_X:]
    indices_train,indices_test = indices_train_test[:split_index_indices], indices_train_test[split_index_indices:]
    #print(x_train.shape,indices_train.shape,x_test.shape,indices_test.shape)
    #print(X_forecast)
    #print(X_forecast.shape)
    #print(X.shape,Y.shape)

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape, X_forecast.shape, indices_train.shape, indices_test.shape, indices_forecast.shape)

    return x_train, y_train, x_test, y_test, X_forecast, indices_train, indices_test, indices_forecast



data = read_csv_file('Dataset For Forecasting Assignment.csv')


start_date = data.index[0]
end_date = data.index[-1]
dates = pd.date_range(start=start_date, end=end_date, freq='1h')
prepare_train_test_forecast(data, test_size=0.33)



def preprocess_data(data, test_size=0.2):

    n_hours=24 # Number of hours to take as input and output for prices and features
    # Resample data to hourly frequency
    data_hourly = data.resample('H').mean()

    # Shift columns to create features
    shifts = {'Price_BE': n_hours*1, 
              'Load_FR': n_hours*1, 
              'Gen_FR': n_hours*1, 'Gen_FR': n_hours*7,
              'Price_CH': n_hours*1, 
              'Wind_BE': n_hours*0, 
              'Solar_BE': n_hours*1,
              'Load_BE': n_hours*1, 'Load_BE': n_hours*7
              }

    Features = pd.concat([data_hourly[col].shift(shift).rename(f'{col}_shifted') for col, shift in shifts.items()], axis=1)
    
    # Drop rows with NaN values due to shifting
    Features = Features.dropna()
    X=Features[:-1]
    print(X.shape)
    X_forecast=Features[-1]
    print(Y.shape)

    # Extract target variable
    Y = data_hourly['Price_BE'].loc[X.index].values.reshape(-1, n_hours)

    # Split data into training and testing sets
    split_index = int((1 - test_size) * len(X))
    x_train, x_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = Y[:split_index], Y[split_index:]

    # Create forecast input
    X_forecast = X.iloc[[-1]]

    return x_train, pd.DataFrame(y_train), x_test, pd.DataFrame(y_test), X_forecast


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