
import inspect
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Input, LSTM, BatchNormalization, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam, RMSprop, SGD # type: ignore
from tensorflow.keras.losses import MeanSquaredError # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasRegressor 
from scipy.stats import randint as sp_randint

#from plot import plot_gridsearch_results
from datagathering import split_train_val_test, prepare_train_test_forecast


def create_model_dense(hidden_layers=1, input_length=24, output_length=24,hidden_neurons=6, activation='relu', learning_rate=0.001,rho=0.9, epsilon=1e-6):
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
        input_shape=(input_length,)
        model = Sequential() # build a sequential model
        model.add(Input(shape=input_shape)) # add an input layer with the shape of the input data features
        for i in range(hidden_layers):
                model.add(Dense(units=hidden_neurons, activation=activation)) # add a hidden layer for each hidden layer specified

        model.add(Dense(units=output_length, activation='linear')) # add an output layer (1 neuron since we are predicting a single value each time)

        rprop = RMSprop(learning_rate=learning_rate, rho=rho, epsilon=epsilon) # type: ignore
        model.compile(loss='mean_squared_error', optimizer=rprop) # compile the model

        return model

def play_model(data: pd.DataFrame,hidden_layers: int=1, hidden_neurons: int=6, activation: str='relu', learning_rate: float=0.001, rho: int=0.9, epsilon: float=1e-6, epochs: int=24, batch_size: int=24,model:str='Dense') -> Tuple[np.ndarray, List[float], List[float], float]:
        """
        This function trains a forecasting model on the given data and returns the predictions. It allows to play with the hyperparameters.
        """
        x_train, y_train, x_test, y_test, x_forecast,indices_train,indices_test,indices_forecast = prepare_train_test_forecast(data,test_size=0.05)
        print(x_train.shape, y_train.shape, x_test.shape, y_test.shape, x_forecast.shape,indices_train.shape,indices_test.shape,indices_forecast.shape)

        # Build the model
        model = create_model_dense(hidden_layers=hidden_layers, hidden_neurons=hidden_neurons, activation=activation, learning_rate=learning_rate, rho=rho, epsilon=epsilon, input_length=x_train.shape[1], output_length=y_train.shape[1])

        # Train the model
        #output_training=model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=(x_val, y_val)) # With self chosen validation set
        output_training=model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_split=0.2) # With validation split
        # Print the training and validation loss
        mse_train=output_training.history['loss']
        mse_val = output_training.history['val_loss']
        print('- mse_train is %.4f' % mse_train[-1] + ' @ ' + str(len(output_training.history['loss'])))
        print('- mse_val is %.4f' % mse_val[-1] + ' @ ' + str(len(output_training.history['val_loss'])))
              
        # Evaluate the model
        test_pred = model.predict(x_test).flatten()
        mse_test = mean_squared_error(y_test.flatten(), test_pred)
        print('Mean Squared Error test:', mse_test) #Alternative way to evaluate the model


        plt.plot(indices_train, y_train.flatten(), label='Train')
        plt.plot(indices_test, y_test.flatten(), label='Actual')
        plt.plot(indices_test, test_pred.flatten(), label='Predicted')
        plt.xlabel('Time')
        plt.ylabel('Price_BE')
        plt.legend()
        plt.show()

        # Make predictions
        predictions = model.predict(x_forecast)

        return predictions, mse_train, mse_val, mse_test

def optimized_model(data: pd.DataFrame,model:str='Dense') -> Tuple[np.ndarray, List[float], List[float], float]:
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
        x_train, y_train, x_test, y_test, x_forecast,indices_train,indices_test,indices_forecast = prepare_train_test_forecast(data,test_size=0.05)
        #print(x_train.shape, y_train.shape, x_test.shape, y_test.shape, x_forecast.shape,indices_train.shape,indices_test.shape,indices_forecast.shape)
        #print(x_train, y_train, x_test, y_test, x_forecast)
        
        # Normalize input data #TODO: check if normalizing has effect
        

        # Define the model
        if model =='Dense':
                selected_model=create_model_dense
        elif model =='LSTM':
                selected_model=create_model_LSTM
        else:
                print('the provided model is not valid')

        # Get the hyperparameters for the selected model
        model_params = inspect.signature(selected_model).parameters
        # Define the hyperparameters and their values
        hyperparameters = {
        'output_length': [y_train.shape[1]],  #, 24, 48
        'input_length': [x_train.shape[1]],  #, 24, 48
        'epsilon': [1e-6],#1e-7,1e-8,1e-5],  #, 1e-7, 1e-8
        'batch_size': [8],#16,32,48],  #, 32, 64
        'epochs': [200],#300,400], #, 48, 72
        'hidden_layers': [2,3,4,5],#2,3,4,5,6],  # , 2, 3
        'hidden_neurons': [24,48,64,128],#48,62,78,94,32],  #sp_randint(3, 12) 6, 12, 24
        'activation': ['relu'],   #, 'tanh', 'sigmoid'
        'learning_rate': [0.001],  #, 0.01, 0.1
        'rho': [0.9],  #, 0.99, 0.999
        'beta_1': [0.99],  #, 0.99, 0.999
        'beta_2': [0.999],  #, 0.99, 0.999
        'momentum': [0.95],  #, 0.95, 0.99
        'nesterov': [True]  #, True, False

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
        model=KerasRegressor(model=selected_model,**param_grid,verbose=2) #Wrap the model in a KerasRegressor 
        ea = EarlyStopping(monitor='loss', patience=100)

        KerasModel=KerasRegressor(model=selected_model,**param_grid,verbose=2,callbacks=[ea]) #Wrap the model in a KerasRegressor 

        #grid_search = GridSearchCV(estimator=KerasModel, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error',verbose=2,n_jobs=-1) # cv: the number of cross-validation folds (means the data is split into 2 parts, 1 for training and 1 for testing)

        grid_search= RandomizedSearchCV(estimator=KerasModel, param_distributions=param_grid, n_iter=100, cv=2, scoring='neg_mean_squared_error',verbose=2,n_jobs=1) 
        print('test')
        grid_search.fit(x_train, y_train,verbose=0)
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        #plot_gridsearch_results(grid_search.cv_results_)
        print("Best: %s using %f" % (best_params, best_score))
        
        # Train the final model
        final_model = selected_model(input_length=best_params['input_length'], output_length=best_params['output_length']
                                     ,hidden_layers=best_params['hidden_layers'], hidden_neurons=best_params['hidden_neurons'], activation=best_params['activation'], learning_rate=best_params['learning_rate'], rho=best_params['rho'], epsilon=best_params['epsilon'])
        print(final_model.summary())

        output_training=final_model.fit(x_train, y_train, epochs=best_params['epochs'], batch_size=best_params['batch_size'], verbose=0)
                                        #,validation_data=(x_val, y_val)) not needed anymore since hyperparameters are already optimized

        # Print the training loss
        mse_train=output_training.history['loss']
        print('- mse_train is %.4f' % mse_train[-1] + ' @ ' + str(len(output_training.history['loss'])))

        # Evaluate the model
        #mse = model.evaluate(X_test_reshaped, y_test.values.reshape(-1, 1), verbose=0,batch_size=batch_size) #TODO: check why not working
        test_pred = final_model.predict(x_test)
        print(test_pred.shape, y_test.shape)
        mse_test = mean_squared_error(y_test, test_pred)
        print('Mean Squared Error test:', mse_test) #Alternative way to evaluate the model

        # Re-order the sets
        #x_train = x_train.sort_index()        
        #x_test = x_test.sort_index()
        #y_train = y_train.sort_index()
        #y_test = y_test.sort_index()
        #print(x_train, y_train, x_test, y_test)

        # Plot the test results
        plt.plot(indices_train, y_train.flatten(), label='Train')
        plt.plot(indices_test, y_test.flatten(), label='Actual')
        plt.plot(indices_test, test_pred.flatten(), label='Predicted')
        plt.xlabel('Time')
        plt.ylabel('Price_BE')
        plt.legend()
        plt.show()
        

        # Make predictions
        predictions = final_model.predict(x_forecast).flatten()

        # Inverse scale the predictions #TODO: check if necessary
        #predictions = scaler.inverse_transform(predictions)


        # Print the predictions
        #print(predictions)
        return predictions, mse_train, mse_test

def create_model_LSTM(hidden_layers=1, hidden_neurons=6, activation='relu', optimizer='rmsprop', learning_rate=0.001, rho=0.9, epsilon=1e-6, beta_1=0.99, beta_2=0.999,momentum = 0.95, nesterov = True):
    """
    Build an LSTM model with Batch Normalization and the specified architecture and parameters.
    :param hidden_layers: The number of LSTM layers in the model.
    :param hidden_neurons: The number of neurons in each LSTM layer.
    :param activation: The activation function for the LSTM layers.
    :param optimizer: The optimizer choice ('rmsprop', 'adam', or 'sgd').
    :param learning_rate: The learning rate for the optimizer.
    :param rho: The rho value for RMSprop optimizer.
    :param epsilon: The epsilon value for the optimizer.
    :return: The built model.
    """
    input_shape = (24, 1)  # Adjust input shape to match the input data shape
    model = Sequential()  # Build a sequential model
    model.add(Input(shape=input_shape))  # Add an input layer with the shape of the input data features

    for i in range(hidden_layers):
        # Add LSTM layers
        model.add(LSTM(hidden_neurons, activation=activation, return_sequences=True if i < hidden_layers - 1 else False))
        # Add Batch Normalization after each LSTM layer
        model.add(BatchNormalization())

    model.add(Dense(units=1, activation='linear'))  # Add an output layer (1 neuron for regression)

    # Choose optimizer
    if optimizer.lower() == 'rmsprop':
        chosen_optimizer = RMSprop(learning_rate=learning_rate, rho=rho, epsilon=epsilon)
    elif optimizer.lower() == 'adam':
        chosen_optimizer = Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)
    elif optimizer.lower() == 'sgd':
        chosen_optimizer = SGD(learning_rate=learning_rate, momentum=momentum, nesterov=nesterov)
    else:
        raise ValueError("Invalid optimizer choice. Choose between 'rmsprop', 'adam', or 'sgd'.")

    model.compile(loss='mean_squared_error', optimizer=chosen_optimizer)  # Compile the model

    return model


def play_model_LSTM(data: pd.DataFrame, hidden_layers: int = 1, hidden_neurons: int = 6, activation: str = 'relu', optimizer='rmsprop', learning_rate: float = 0.001, rho: float = 0.9, epsilon: float = 1e-6, beta_1:float=0.99, beta_2:float=0.999,momentum:float= 0.95, nesterov:bool = True, epochs: int = 24, batch_size: int = 24):
    """
    This function trains an LSTM forecasting model on the given data and returns the predictions. It allows to play with the hyperparameters.
    """
    ## Prepare the data
    x_train = split_train_val_test(data)[0][0]
    y_train = split_train_val_test(data)[0][1]

    x_val = split_train_val_test(data)[1][0]
    y_val = split_train_val_test(data)[1][1]

    x_test = split_train_val_test(data)[2][0]
    y_test = split_train_val_test(data)[2][1]

    x_forecast = split_train_val_test(data)[3]

    # Reshape input data for LSTM
    x_train_reshaped = np.expand_dims(x_train, axis=2)
    x_val_reshaped = np.expand_dims(x_val, axis=2)
    x_test_reshaped = np.expand_dims(x_test, axis=2)
    x_forecast_reshaped = np.expand_dims(x_forecast, axis=2)

    # Build the model
    model = create_model_LSTM(hidden_layers=hidden_layers, hidden_neurons=hidden_neurons, activation=activation, optimizer=optimizer, learning_rate=learning_rate, rho=rho, epsilon=epsilon, beta_1=beta_1, beta_2=beta_2, momentum=momentum, nesterov=nesterov)

    # Train the model
    output_training = model.fit(x_train_reshaped, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=(x_val_reshaped, y_val))

    # Print the training and validation loss
    mse_train = output_training.history['loss']
    mse_val = output_training.history['val_loss']
    print('- mse_train is %.4f' % mse_train[-1] + ' @ ' + str(len(output_training.history['loss'])))
    print('- mse_val is %.4f' % mse_val[-1] + ' @ ' + str(len(output_training.history['val_loss'])))

    # Evaluate the model
    test_pred = model.predict(x_test_reshaped).flatten()
    mse_test = mean_squared_error(y_test, test_pred)
    print('Mean Squared Error test:', mse_test)  # Alternative way to evaluate the model
 
    # Make predictions
    predictions = model.predict(x_forecast_reshaped)

    return predictions, mse_train, mse_val, mse_test

def optimized_model_LSTM(data: pd.DataFrame) -> Tuple[np.ndarray, List[float], float]:
    """
    This function trains a forecasting model on the given data and returns the predictions. It optimizes the hyperparameters.
    :param data: The data to train the model on.

    :return: The predictions made by the model.
    :return: The training loss.
    :return: The test loss.
    """

    ## Prepare the data
    x_train, y_train, x_test, y_test, x_forecast, indices_train, indices_test, indices_forecast = prepare_train_test_forecast(data, test_size=0.05)

    # Reshape input data for LSTM
    x_train_reshaped = np.expand_dims(x_train, axis=2)
    x_test_reshaped = np.expand_dims(x_test, axis=2)
    x_forecast_reshaped = np.expand_dims(x_forecast, axis=2)

    # Get the hyperparameters for the selected model
    model_params = inspect.signature(create_model_LSTM).parameters

    # Define the hyperparameters and their values
    hyperparameters = {
        'output_length': [y_train.shape[1]],
        'input_length': [x_train_reshaped.shape[1]],  # Include input_length here
        'epsilon': [1e-6],
        'batch_size': [8],
        'epochs': [16],
        'hidden_layers': [1],
        'hidden_neurons': [24],
        'activation': ['relu'],
        'learning_rate': [0.001],
        'rho': [0.9],
        'beta_1': [0.99],
        'beta_2': [0.999],
        'momentum': [0.95],
        'nesterov': [True],
        'optimizer': ['rmsprop'], #'sgd', 'adam'
    }


    # Iterate over hyperparameters and add them to param_grid only if they are present in model_params
    param_grid = {}
    for param, values in hyperparameters.items():
        if param in model_params:
            param_grid[param] = values
        if param == 'batch_size':
            param_grid[param] = values
        if param == 'epochs':
            param_grid[param] = values

    # Optimize hyperparameters
    KerasModel = KerasRegressor(model=create_model_LSTM, **param_grid, verbose=2)
    ea = EarlyStopping(monitor='loss', patience=100)

    grid_search = RandomizedSearchCV(estimator=KerasModel, param_distributions=param_grid, n_iter=10, cv=2,
                                     scoring='neg_mean_squared_error', verbose=2, n_jobs=1)
    print('test')
    grid_search.fit(x_train_reshaped, y_train, verbose=1)
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print("Best: %s using %f" % (best_params, best_score))

    # Train the final model
    final_model = create_model_LSTM(hidden_layers=best_params['hidden_layers'], hidden_neurons=best_params['hidden_neurons'],
                             activation=best_params['activation'], optimizer = best_params['optimizer'], learning_rate=best_params['learning_rate'],
                             rho=best_params['rho'], epsilon=best_params['epsilon'], beta_1=best_params['beta_1'], beta_2=best_params['beta_2'], 
                             momentum=best_params['momentum'], nesterov=best_params['nesterov'])
    print(final_model.summary())

    output_training = final_model.fit(x_train_reshaped, y_train, epochs=best_params['epochs'],
                                      batch_size=best_params['batch_size'], verbose=0)

    # Print the training loss
    mse_train = output_training.history['loss']
    print('- mse_train is %.4f' % mse_train[-1] + ' @ ' + str(len(output_training.history['loss'])))

    # Evaluate the model
    test_pred = final_model.predict(x_test_reshaped).reshape(-1, 1)  # Reshape predictions to match y_test shape
    mse_test = mean_squared_error(y_test, test_pred)
    print('Mean Squared Error test:', mse_test)

    # Plot the test results
    plt.plot(indices_train, y_train.flatten(), label='Train')
    plt.plot(indices_test, y_test.flatten(), label='Actual')
    plt.plot(indices_test, test_pred.flatten(), label='Predicted')
    plt.xlabel('Time')
    plt.ylabel('Price_BE')
    plt.legend()
    plt.show()

    # Make predictions
    predictions = final_model.predict(x_forecast_reshaped).flatten()

    return predictions, mse_train, mse_test




# TODO: check if below model can be used too
        # Build the model
        #model = ()
        #model.add(LSTM(64, activation='relu', input_shape=(time_steps, X_train_scaled.shape[1])))
        #model.add(Dense(1))
        #model.compile(optimizer=Adam(), loss=MeanSquaredError())
        

        #model = build_model(input_shape=(X_train_reshaped.shape[1],), hidden_layers=hidden_layers, hidden_neurons=hidden_neurons, activation_function=activation_function, learning_rate=learning_rate, rho=rho, epsilon=epsilon)