import csv
import csv
import pandas as pd

# Define a function to write forecasted values to a CSV file
def write_forecasted_values(forecasted_values: pd.DataFrame):
    """
    Write the forecasted values to a CSV file.
    :param forecasted_values: The forecasted values to write to the CSV file.
    """
    with open('predictions.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Date', 'Forecasted Price'])
        for index, row in forecasted_values.iterrows():
            writer.writerow([index, row['Price_BE']])