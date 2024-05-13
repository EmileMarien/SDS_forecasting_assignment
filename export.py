import csv
import csv
import pandas as pd
import numpy as np

def write_forecasted_values(forecasted_values: np.ndarray):
    """
    Write the forecasted values to a CSV file.
    :param forecasted_values: The forecasted values to write to the CSV file.
    """
    forecasted_values = forecasted_values.reshape(-1)
    forecasted_values_df = pd.DataFrame(forecasted_values, columns=['Price_BE'])

    with open('predictions.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Date', 'Forecasted Price'])
        for index, row in forecasted_values_df.iterrows():
            writer.writerow([row['Price_BE']])

