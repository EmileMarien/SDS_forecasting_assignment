import csv
import csv


# Define a function to write forecasted values to a CSV file
def write_forecasted_values(forecasted_values: list[int]=[]):
    """
    Write the forecasted values to a CSV file.
    :param forecasted_values: The forecasted values to write to the CSV file.
    """
    with open('predictions.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for value in forecasted_values:
            writer.writerow([value])