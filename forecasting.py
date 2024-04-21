
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from datagathering import prepare_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam, RMSprop # type: ignore
from tensorflow.keras.losses import MeanSquaredError # type: ignore
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def forecasting_model(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    ## Prepare the data
    # Select the relevant columns for input and output
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
    time_steps = 24  # Assuming each day has 24 hours

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

    print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)
    # Scale the data
    scaler = MinMaxScaler()
    #X_train_scaled = scaler.fit_transform(X_train)
    #X_val_scaled = scaler.transform(X_val)
    #X_test_scaled = scaler.transform(X_test)

    # Reshape the input data
    X_train_reshaped = X_train.values.reshape(-1, X_train.shape[1])

    # Reshape the output data
    y_train = y_train.values
    y_val = y_val.values

    print('input_features ' + str(X_train_reshaped.shape))
    print('target dimensions ' + str(y_train.shape))
    # Build the model
    #model = ()
    #model.add(LSTM(64, activation='relu', input_shape=(time_steps, X_train_scaled.shape[1])))
    #model.add(Dense(1))
    #model.compile(optimizer=Adam(), loss=MeanSquaredError())
    neurons = [24, 24 ]
    activation_functions = ['relu', 'linear']

    model = Sequential()
    model.add(Dense(neurons[0], input_dim=X.shape[1], activation=activation_functions[0]))
    model.add(Dense(neurons[1], activation=activation_functions[1]))

    rprop = RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-6) # type: ignore
    model.compile(loss='mean_squared_error', optimizer=rprop)
    
    # Train the model
    model.fit(X_train_reshaped, y_train, epochs=50, batch_size=32, verbose=1, validation_data=(X_val_reshaped, y_val))

    # Evaluate the model
    mse = model.evaluate(X_test_reshaped, y_test, verbose=0)
    print('Mean Squared Error:', mse)

    # Make predictions
    predictions = model.predict(X_test_reshaped)

    # Inverse scale the predictions
    predictions = scaler.inverse_transform(predictions)

    # Plot the test results
    plt.plot(X_test.index, y_test.values, label='Actual')
    plt.plot(X_test.index, predictions.flatten(), label='Predicted')
    plt.xlabel('Time')
    plt.ylabel('Price_BE')
    plt.legend()
    plt.show()

    # Print the predictions
    print(predictions)
    return predictions
