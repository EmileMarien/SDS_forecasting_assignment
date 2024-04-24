


from matplotlib import pyplot as plt
import pandas as pd
from datagathering import read_csv_file, remove_outliers
from forecasting import forecasting_model
from plot import plot_lists, subplot, boxplot, plot_training_validation_loss, plot_training_validation_loss_lr, plot_training_validation_loss_rho

data = read_csv_file('Dataset For Forecasting Assignment.csv')

start_date = data.index[0]
end_date = data.index[-1]

dates = pd.date_range(start=start_date, end=end_date, freq='1h')
for d in dates:
    try:
        p = data.loc[d]
    except KeyError:
        print(d)
print(data.head(10))

#plot_lists(data.index, data['Price_BE'], legend=['Price_BE'], xlabel='Date', ylabel='Price', title='Price_BE')
#data=remove_outliers(data,'Price_BE')

predictions,mse_train,mse_val,mse_test=forecasting_model(data, time_steps=24, neurons=[24,1], activation_functions=['relu', 'linear'], learning_rate=0.001, rho=0.9, epochs=72, batch_size=24,epsilon=1e-6)


#plot_training_validation_loss(mse_train, mse_val)
#plot_lists(predictions.index,predictions, legend=['Price_BE'], xlabel='Date', ylabel='Price', title='Price_BE')

#plot_training_validation_loss_lr(data, [0.001, 0.01, 0.1])
#plot_training_validation_loss_rho(data, [0.9, 0.99, 0.999])

