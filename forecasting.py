
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from datagathering import prepare_data
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Input, LSTM # type: ignore
from tensorflow.keras.optimizers import Adam, RMSprop # type: ignore
from tensorflow.keras.losses import MeanSquaredError # type: ignore
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def forecasting_model(data: pd.DataFrame, time_steps: int=24, neurons: List[int]=[24,24], activation_functions: List[str]=['relu', 'linear'], learning_rate: float=0.001, rho:int=0.9, epochs: int=24, batch_size: int=24,epsilon: int=1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """
        This function trains a forecasting model on the given data and returns the predictions.
        :param data: The data to train the model on.
        :param time_steps: The number of time steps to use for each input sample.
        :param neurons: The number of neurons in each layer of the model.
        :param activation_functions: The activation functions for each layer of the model.
        :param learning_rate: The learning rate for the optimizer.
        :param rho: The rho value for the optimizer.
        :param epochs: The number of epochs to train the model for.
        :param batch_size: The batch size to use for training.
        :param epsilon: The epsilon value for the optimizer.

        :return: The predictions made by the model.
        """
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

        X_forecast=X[(X.index >= end_date)]
        # Check the dimensions of the data
        print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)
        # Scale the data
        scaler = MinMaxScaler()
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

        print('input_features ' + str(X_train_reshaped.shape)+str(X_val_reshaped.shape)+str(X_test_reshaped.shape))
        print('target dimensions ' + str(y_train.shape)+str(y_val.shape))
        # Build the model
        #model = ()
        #model.add(LSTM(64, activation='relu', input_shape=(time_steps, X_train_scaled.shape[1])))
        #model.add(Dense(1))
        #model.compile(optimizer=Adam(), loss=MeanSquaredError())
        model = Sequential()
        model.add(Input(shape=(X_train_reshaped.shape[1],)))
        model.add(Dense(neurons[0], input_dim=X.shape[1], activation=activation_functions[0]))
        model.add(Dense(neurons[1], activation=activation_functions[1]))

        rprop = RMSprop(learning_rate=learning_rate, rho=rho, epsilon=epsilon) # type: ignore
        model.compile(loss='mean_squared_error', optimizer=rprop)
        
        # Train the model
        model.fit(X_train_reshaped, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=(X_val, y_val))

        # Evaluate the model
        print(X_test_reshaped.shape)
        print(y_test.shape)
        mse = model.evaluate(X_test_reshaped, y_test, verbose=0,batch_size=batch_size)
        print('Mean Squared Error:', mse)

        # Make predictions
        predictions = model.predict(X_forecast.values.reshape(-1, X_forecast.shape[1]))

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
