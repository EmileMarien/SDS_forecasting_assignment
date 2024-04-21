import datetime
import matplotlib.pyplot as plt

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