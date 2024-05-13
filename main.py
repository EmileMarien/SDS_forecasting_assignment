


from matplotlib import pyplot as plt
import pandas as pd
from datagathering import read_csv_file, remove_outliers
from forecasting import optimized_model, play_model, play_model_LSTM, play_model
from export import write_forecasted_values
from plot import plot_lists, subplot, boxplot, plot_training_validation_loss, plot_training_validation_loss_lr, plot_training_validation_loss_rho
from scikeras.wrappers import KerasRegressor 

data = read_csv_file('Dataset For Forecasting Assignment.csv')

start_date = data.index[0]
end_date = data.index[-1]

dates = pd.date_range(start=start_date, end=end_date, freq='1h')
""" CHECK FOR MISSING DATES 
for d in dates:
    try:
        p = data.loc[d]
    except KeyError:
        print(d)
print(data.head(10))
"""

#predictions,mse_train,mse_test=play_model_LSTM(data, optimizer='adam')

#predictions,mse_train,mse_test=optimized_model(data,model='Dense')

predictions,mse_train,mse_val,mse_test=play_model(data,model='Dense',learning_rate=0.001,rho=0.99,epochs=200,hidden_neurons=24,batch_size=16,hidden_layers=2,epsilon=1e-6)

write_forecasted_values(predictions)


