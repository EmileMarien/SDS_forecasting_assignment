
import inspect
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler
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
from scikeras.wrappers import KerasRegressor 
from scipy.stats import randint as sp_randint


from datagathering import split_train_val_test, prepare_train_test_forecast


def create_model(hidden_layers=1, hidden_neurons=6, activation='relu', learning_rate=0.001,rho=0.9, epsilon=1e-6):
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
        input_shape=(6,)
        model = Sequential() # build a sequential model
        model.add(Input(shape=input_shape)) # add an input layer with the shape of the input data features
        for i in range(hidden_layers):
                model.add(Dense(units=hidden_neurons, activation=activation)) # add a hidden layer for each hidden layer specified

        model.add(Dense(units=1, activation='linear')) # add an output layer (1 neuron since we are predicting a single value each time)

        rprop = RMSprop(learning_rate=learning_rate, rho=rho, epsilon=epsilon) # type: ignore
        model.compile(loss='mean_squared_error', optimizer=rprop) # compile the model

        return model

def create_model2(hidden_layers=1, hidden_neurons=6, activation='relu', learning_rate=0.001,rho=0.9, epsilon=1e-6):
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
        #TODO: change to other model (LSTM)
        input_shape=(6,)
        model = Sequential() # build a sequential model
        model.add(Input(shape=input_shape)) # add an input layer with the shape of the input data features
        for i in range(hidden_layers):
                model.add(Dense(units=hidden_neurons, activation=activation)) # add a hidden layer for each hidden layer specified

        model.add(Dense(units=1, activation='linear')) # add an output layer (1 neuron since we are predicting a single value each time)

        rprop = RMSprop(learning_rate=learning_rate, rho=rho, epsilon=epsilon) # type: ignore
        model.compile(loss='mean_squared_error', optimizer=rprop) # compile the model

        return model

def optimized_model(data: pd.DataFrame) -> Tuple[np.ndarray, List[float], List[float], float]:
        """
        This function trains a forecasting model on the given data and returns the predictions. It optimizes the hyperparameters.
        :param data: The data to train the model on.

        :param epochs: The number of epochs to train the model for.
        :param batch_size: The batch size to use for training.

        :return: The predictions made by the model.
        :return: The training loss.
        :return: The validation loss.
        :return: The test loss.
        """
        ## Prepare the data
        x_train, y_train, x_test, y_test, x_forecast = prepare_train_test_forecast(data)
        print(x_train.shape, y_train.shape, x_test.shape, y_test.shape, x_forecast.shape)
        #print(x_train, y_train, x_test, y_test, x_forecast)
        
        # Define the model
        selected_model=create_model

        # Get the hyperparameters for the selected model
        model_params = inspect.signature(selected_model).parameters
        print(model_params)
        # Define the hyperparameters and their values
        hyperparameters = {
        'epsilon': [1e-6],  #, 1e-7, 1e-8
        'batch_size': [24],  #, 32, 64
        'epochs': [24,48,72], #, 48, 72
        'hidden_layers': [1,2,3],  # , 2, 3
        'hidden_neurons': [6],  #sp_randint(3, 12) 6, 12, 24
        'activation': ['relu','tanh'],   #, 'tanh', 'sigmoid'
        'learning_rate': [0.01,0.001],  
        'rho': [0.99],  
        }

        # Iterate over hyperparameters and add them to param_grid only if they are present in model_params
        param_grid = {}
        for param, values in hyperparameters.items():
                if param in model_params:
                        param_grid[param] = values
                if param =='batch_size':
                      param_grid[param] = values
                if param =='epochs':
                        param_grid[param] = values
        print(param_grid)

        # Optimize hyperparameters
        model=KerasRegressor(model=create_model,**param_grid,verbose=2) #Wrap the model in a KerasRegressor 

        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error',verbose=2) # cv: the number of cross-validation folds (means the data is split into 2 parts, 1 for training and 1 for testing)

        #grid_search= RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=10, cv=2, scoring='neg_mean_squared_error',verbose=2) 
        
        grid_search.fit(x_train, y_train,verbose=0)
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        print("Best: %s using %f" % (best_params, best_score))
        
        # Train the final model
        final_model = create_model(hidden_layers=best_params['hidden_layers'], hidden_neurons=best_params['hidden_neurons'], activation=best_params['activation'], learning_rate=best_params['learning_rate'], rho=best_params['rho'], epsilon=best_params['epsilon'])

        output_training=final_model.fit(x_train, y_train, epochs=best_params['epochs'], batch_size=best_params['batch_size'], verbose=1)
                                        #,validation_data=(x_val, y_val)) not needed anymore since hyperparameters are already optimized

        # Print the training loss
        mse_train=output_training.history['loss']
        print('- mse_train is %.4f' % mse_train[-1] + ' @ ' + str(len(output_training.history['loss'])))

        # Evaluate the model
        #mse = model.evaluate(X_test_reshaped, y_test.values.reshape(-1, 1), verbose=0,batch_size=batch_size) #TODO: check why not working
        test_pred = final_model.predict(x_test).flatten()
        mse_test = mean_squared_error(y_test, test_pred)
        print('Mean Squared Error test:', mse_test) #Alternative way to evaluate the model

        # Re-order the sets
        x_train = x_train.sort_index()        
        x_test = x_test.sort_index()
        y_train = y_train.sort_index()
        y_test = y_test.sort_index()
        #print(x_train, y_train, x_test, y_test)

        # Plot the test results
        plt.plot(x_train.index, y_train, label='Train')
        plt.plot(x_test.index, y_test, label='Actual')
        plt.plot(x_test.index, test_pred, label='Predicted')
        plt.xlabel('Time')
        plt.ylabel('Price_BE')
        plt.legend()
        plt.show()
        

        # Make predictions
        predictions = final_model.predict(x_forecast)

        # Inverse scale the predictions #TODO: check if necessary
        #predictions = scaler.inverse_transform(predictions)


        # Print the predictions
        #print(predictions)
        return predictions, mse_train, mse_test


def play_model(data: pd.DataFrame,hidden_layers: int=1, hidden_neurons: int=6, activation: str='relu', learning_rate: float=0.001, rho: int=0.9, epsilon: float=1e-6, epochs: int=24, batch_size: int=24):
        """
        This function trains a forecasting model on the given data and returns the predictions. It allows to play with the hyperparameters.
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
        model = create_model(hidden_layers=hidden_layers, hidden_neurons=hidden_neurons, activation=activation, learning_rate=learning_rate, rho=rho, epsilon=epsilon)

        # Train the model
        output_training=model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=(x_val, y_val))

        # Print the training and validation loss
        mse_train=output_training.history['loss']
        mse_val = output_training.history['val_loss']
        print('- mse_train is %.4f' % mse_train[-1] + ' @ ' + str(len(output_training.history['loss'])))
        print('- mse_val is %.4f' % mse_val[-1] + ' @ ' + str(len(output_training.history['val_loss'])))
              
        # Evaluate the model
        test_pred = model.predict(x_test).flatten()
        mse_test = mean_squared_error(y_test, test_pred)
        print('Mean Squared Error test:', mse_test) #Alternative way to evaluate the model

        # Plot the test results
        plt.plot(x_train.index, y_train, label='Train')
        plt.plot(x_val.index, y_val, label='Validate')
        plt.plot(x_test.index, y_test, label='Actual')
        plt.plot(x_test.index, test_pred.flatten(), label='Predicted')
        plt.xlabel('Time')
        plt.ylabel('Price_BE')
        plt.legend()
        plt.show()

        # Make predictions
        predictions = model.predict(x_forecast)

        return predictions, mse_train, mse_val, mse_test

# TODO: check if below model can be used too
        # Build the model
        #model = ()
        #model.add(LSTM(64, activation='relu', input_shape=(time_steps, X_train_scaled.shape[1])))
        #model.add(Dense(1))
        #model.compile(optimizer=Adam(), loss=MeanSquaredError())
        

        #model = build_model(input_shape=(X_train_reshaped.shape[1],), hidden_layers=hidden_layers, hidden_neurons=hidden_neurons, activation_function=activation_function, learning_rate=learning_rate, rho=rho, epsilon=epsilon)