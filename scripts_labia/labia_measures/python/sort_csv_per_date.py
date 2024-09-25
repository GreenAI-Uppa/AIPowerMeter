import os
import csv

# Directory containing the CSV files
path = '/mnt/beegfs/projects/coca4ai/metrics/summary_per_user/'

# Get all the CSV files in the current directory
csv_files = [file for file in os.listdir(path) if file.endswith('.csv')]

# Sort each CSV file
for file in csv_files:
    # Read the CSV file
    print(file)
    with open(path + file, 'r') as csv_file:
        reader = csv.reader(csv_file)
        lines = list(reader)

    # Sort the lines by "start date" (in the third column, format : 2024-06-16T05:21:25)
    sorted_lines = sorted(lines[1:], key=lambda x: x[2])

    # Write the sorted lines back to the CSV file
    with open(path + file, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(lines[0])  # Write the header line
        writer.writerows(sorted_lines)