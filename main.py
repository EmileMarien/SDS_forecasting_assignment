


import pandas as pd
from datagathering import read_csv_file, remove_outliers
from forecasting import forecasting_model
from plot import plot_lists, subplot, boxplot

data = read_csv_file('Dataset For Forecasting Assignment.csv')

start_date = data.index[0]
end_date = data.index[-1]

dates = pd.date_range(start=start_date, end=end_date, freq='1h')
for d in dates:
    try:
        p = data.loc[d]
    except KeyError:
        print(d)
data.head()

#plot_lists(data.index, data['Price_BE'], legend=['Price_BE'], xlabel='Date', ylabel='Price', title='Price_BE')
data=remove_outliers(data,'Price_BE')

predictions=forecasting_model(data)




