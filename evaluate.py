
# Compare the performance of the two optimized models
from datagathering import read_csv_file, remove_outliers
from forecasting import optimized_model, play_model


data = read_csv_file('Dataset For Forecasting Assignment.csv')
predictions,mse_train,mse_test=optimized_model(data,model='LSTM')

