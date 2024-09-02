import json
import os
import subprocess
from datetime import datetime

def load_data(file_name):
    try:
        with open(file_name, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"File {file_name} not found.")
        return []

def check_null_values(data):
    dates_with_null = set()
    for record in data:
        if record.get('taux_co2') is None:
            print(f"null value on date: {record.get('date_heure')}")
            # Convert date to "mm_dd_yyyy" format and add to the set
            date_str = record.get('date_heure').split(' ')[0]  # Extract the date part
            date_formatted = datetime.strptime(date_str, "%Y-%m-%d").strftime("%m_%d_%Y")
            dates_with_null.add(date_formatted)
    return list(dates_with_null)

def save_data(data, file_name):
    with open(file_name, 'w') as file:
        json.dump(data, file, indent=4)
    print(f"Data saved to {file_name}")

def main():
    file_name = "co2.json"
    data = load_data(file_name)
    if data:
        dates_with_null = check_null_values(data)
        if dates_with_null:
            print("Dates with null values:")
            for date in dates_with_null:
                print(date)
                user_input = input(f"Do you want to (1) re-fetch data or (2) delete null records for date {date}? (1/2): ")
                if user_input == '1':
                    # Execute the get_co2.py script for the specified date
                    subprocess.run(["python", "get_co2.py", date, date])
                elif user_input == '2':
                    # Remove records with null values for the specified date
                    date_to_remove = datetime.strptime(date, "%m_%d_%Y").strftime("%Y-%m-%d")
                    data = [record for record in data if not (record['date_heure'].startswith(date_to_remove) and record['taux_co2'] is None)]
            save_data(data, file_name)
    else:
        print("No data to process.")

if __name__ == "__main__":
    main()
