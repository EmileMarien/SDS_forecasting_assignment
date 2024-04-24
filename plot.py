import datetime
import matplotlib.pyplot as plt
import pandas as pd
from forecasting import forecasting_model
import seaborn as sns # type: ignore

def plot_lists(x_values, y_values, legend=None, xlabel=None, ylabel=None, title=None):
    plt.plot(x_values, y_values)
    if legend:
        plt.legend(legend)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.title(title)
    
    # Set the limits of the y-axis
    plt.ylim([min(y_values), max(y_values)])
    plt.show()

def subplot(data):
    start = datetime.datetime(2021, 1, 1, 0, 0)
    end = datetime.datetime(2021, 1, 14, 23, 45)

    plt.figure()
    plt.subplot(311)
    plt.plot(data.belpex[start:end], label='belpex')
    plt.legend(frameon=False)
    plt.subplot(312)
    plt.plot(data.solar[start:end], label='solar')
    plt.legend(frameon=False)
    plt.subplot(313)
    plt.plot(data.wind[start:end], label='wind')
    plt.legend(frameon=False)
    plt.show()

def boxplot(data):

    data['week_days']=data.index.weekday
    data['month_days']=data.index.day
    data['hours']=data.index.hour
    data['months']=data.index.month

    data.boxplot(column='belpex', by='month_days')
    data.boxplot(column='belpex', by='week_days')
    data.boxplot(column='belpex', by='hours')
    data.boxplot(column='belpex', by='months')

def plot_training_validation_loss_lr(data, learning_rates):
    # Create a new figure
    plt.figure()

    for lr in learning_rates:
        predictions, mse_train, mse_val, mse_test = forecasting_model(data, time_steps=24, neurons=[24,1], activation_functions=['relu', 'linear'], learning_rate=lr, rho=0.9, epochs=48, batch_size=24,epsilon=1e-6)

        # Plot mse_train
        plt.plot(mse_train[1:], label=f'Training loss (lr={lr})')

        # Plot mse_val
        plt.plot(mse_val[1:], label=f'Validation loss (lr={lr})')

    # Add a title
    plt.title('Training and Validation Loss for different learning rates')

    # Add x and y label
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # Add a legend
    plt.legend()

    # Show the plot
    plt.show()

def plot_training_validation_loss(mse_train, mse_val):
    # Create a new figure
    plt.figure()

    # Plot mse_train
    plt.plot(mse_train[1:], label='Training loss')

    # Plot mse_val
    plt.plot(mse_val[1:], label='Validation loss')

    # Add a title
    plt.title('Training and Validation Loss')

    # Add x and y label
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # Add a legend
    plt.legend()

    # Show the plot
    plt.show()



def plot_training_validation_loss_rho(data, rhos):
    # Create a new figure
    plt.figure()

    for rho in rhos:
        predictions, mse_train, mse_val, mse_test = forecasting_model(data, time_steps=24, neurons=[24,1], activation_functions=['relu', 'linear'], learning_rate=0.01, rho=rho, epochs=48, batch_size=24, epsilon=1e-6)

        # Plot mse_train
        plt.plot(mse_train[1:], label=f'Training loss (rho={rho})')

        # Plot mse_val
        plt.plot(mse_val[1:], label=f'Validation loss (rho={rho})')

    # Add a title
    plt.title('Training and Validation Loss for different rho values')

    # Add x and y label
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # Add a legend
    plt.legend()

    # Show the plot
    plt.show()


def plot_gridsearch_results(results):
    # Access the grid search results
    results = results

    # Extract relevant data into a DataFrame
    data = pd.DataFrame({
        'C': results['param_C'],
        'kernel': results['param_kernel'],
        'mean_score': results['mean_test_score']
    })

    # Reshape the data into a pivot table format for the heatmap
    heatmap_data = data.pivot(index='C', columns='kernel', values='mean_score')

    # Create a heatmap using seaborn
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True)
    plt.xlabel('Kernel')
    plt.ylabel('C')
    plt.title('Grid Search Results Heatmap')
    plt.show()