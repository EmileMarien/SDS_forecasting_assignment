
from sklearn.metrics import mean_squared_error
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
from sklearn.model_selection import GridSearchCV
from datagathering import split_train_val_test

def build_model(input_shape:Tuple(),hidden_layers: int, hidden_neurons: int, activation_function: str, learning_rate: float, rho: int, epsilon: float) -> Sequential: # type: ignore
                """
                Build a sequential model with the specified architecture and parameters.
                :param input_shape: The shape of the input data features.
                :param hidden_layers: The number of hidden layers in the model.
                :param hidden_neurons: The number of neurons in each hidden layer.
                :param activation_function: The activation function for the hidden layers.
                :param learning_rate: The learning rate for the optimizer.
                :param rho: The rho value for the optimizer.
                :param epsilon: The epsilon value for the optimizer.
                :return: The built model.
                """
                model = Sequential() # build a sequential model
                model.add(Input(shape=input_shape)) # add an input layer with the shape of the input data features
                for i in range(hidden_layers):
                        model.add(Dense(units=hidden_neurons, activation=activation_function)) # add a hidden layer

                model.add(Dense(units=1, activation='linear')) # add an output layer

                rprop = RMSprop(learning_rate=learning_rate, rho=rho, epsilon=epsilon) # type: ignore
                model.compile(loss='mean_squared_error', optimizer=rprop) # compile the model

                return model

def forecasting_model(data: pd.DataFrame, time_steps: int=24, hidden_layers: int=1, hidden_neurons:int=6, activation_function: str='relu', learning_rate: float=0.001, rho:int=0.9, epochs: int=24, batch_size: int=24,epsilon: int=1e-6) -> Tuple[np.ndarray, np.ndarray]:
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
        x_train=split_train_val_test(data)[0][0]
        y_train=split_train_val_test(data)[0][1]

        x_val=split_train_val_test(data)[1][0]
        y_val=split_train_val_test(data)[1][1]

        x_test=split_train_val_test(data)[2][0]
        y_test=split_train_val_test(data)[2][1]

        x_forecast=split_train_val_test(data)[3]

        # Build the model
        #model = ()
        #model.add(LSTM(64, activation='relu', input_shape=(time_steps, X_train_scaled.shape[1])))
        #model.add(Dense(1))
        #model.compile(optimizer=Adam(), loss=MeanSquaredError())
        

        #model = build_model(input_shape=(X_train_reshaped.shape[1],), hidden_layers=hidden_layers, hidden_neurons=hidden_neurons, activation_function=activation_function, learning_rate=learning_rate, rho=rho, epsilon=epsilon)
        
        # Define the hyperparameters to search
        param_grid = {
            'hidden_layers': [1, 2, 3],
            'hidden_neurons': [3, 6, 12, 24],
            'activation_function': ['relu', 'tanh', 'sigmoid'],
            'learning_rate': [0.001, 0.01, 0.1],
            'rho': [0.9, 0.99, 0.999],
            'epsilon': [1e-6, 1e-7, 1e-8]
        }

        # Optimize hyperparameters
        grid_search = GridSearchCV(build_model(input_shape=(x_train.shape[1],)), param_grid, cv=5, scoring='accuracy')
        grid_search.fit(x_train, y_train)
        best_params = grid_search.best_params_
        print(best_params)
        best_score = grid_search.best_score_

        # Train the final model
        final_model = build_model(input_shape=(x_train.shape[1],), **best_params)
        output_training=final_model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=(x_val, y_val))

        mse_train=output_training.history['loss']
        mse_val = output_training.history['val_loss']
        print('- mse_train is %.4f' % mse_train[-1] + ' @ ' + str(len(output_training.history['loss'])))
        print('- mse_val is %.4f' % mse_val[-1] + ' @ ' + str(len(output_training.history['val_loss'])))

        # Evaluate the model
        #print(X_test_reshaped.shape)
        #print(y_test.values.reshape(-1, 1).shape)

        #mse = model.evaluate(X_test_reshaped, y_test.values.reshape(-1, 1), verbose=0,batch_size=batch_size)
        test_pred = final_model.predict(x_test).flatten()
        mse_test = mean_squared_error(y_test, test_pred)
        print('Mean Squared Error test:', mse_test)

        # Plot the test results
        """
        plt.plot(X_test.index, y_test, label='Actual')
        plt.plot(X_test.index, test_pred.flatten(), label='Predicted')
        plt.xlabel('Time')
        plt.ylabel('Price_BE')
        plt.legend()
        plt.show()
        """

        # Make predictions
        predictions = final_model.predict(x_forecast.values.reshape(-1, x_forecast.shape[1]))

        # Inverse scale the predictions
        #predictions = scaler.inverse_transform(predictions)


        # Print the predictions
        #print(predictions)
        return predictions, mse_train, mse_val, mse_test

