import csv

# Generate the forecasted values
forecasted_values = [1.2, 2.5, 3.1, ...]  # Replace with your actual forecasted values

# Write the forecasted values to the CSV file
with open('predictions.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for value in forecasted_values:
        writer.writerow([value])