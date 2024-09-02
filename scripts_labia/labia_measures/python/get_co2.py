import requests
import json
import sys
from datetime import datetime, timedelta

BASE_URL = "https://odre.opendatasoft.com"

DATA_PATH = "/mnt/beegfs/projects/coca4ai/metrics/co2.json"

def fetch_data(date):
    date = date.split(' ')[0]  # Split the date to get only "YYYY-MM-DD"
    url = f'{BASE_URL}/api/explore/v2.1/catalog/datasets/eco2mix-national-tr/records?select=date%2C%20taux_co2%2Cheure&where=date%3D%22{date}%22&order_by=heure%20desc&limit=100'
    response = requests.get(url)
    if response.status_code == 200:
        response_json = response.json()
        data = response_json.get('results', [])
        return data
    else:
        print(f"Error fetching data for {date}: {response.status_code}")
        return None

def fetch_data_by_datetime(datetime_str):
    date, time = datetime_str.split(' ')
    url = f'{BASE_URL}/api/explore/v2.1/catalog/datasets/eco2mix-national-tr/records?select=date%2C%20taux_co2%2Cheure&where=date%3D%22{date}%22%20AND%20heure%3D%22{time}%22&limit=1'
    response = requests.get(url)
    if response.status_code == 200:
        response_json = response.json()
        data = response_json.get('results', [])
        return data[0] if data else None
    else:
        print(f"Error fetching data for {datetime_str}: {response.status_code}")
        return None

def load_existing_data(file_name):
    try:
        with open(file_name, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return []

def save_data(data, file_name):
    with open(file_name, 'w') as file:
        json.dump(data, file, indent=4)
    print(f"Data saved to {file_name}")

def date_range(start_date, end_date):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    delta = timedelta(days=1)
    current = start
    while current <= end:
        yield current.strftime("%Y-%m-%d")
        current += delta

def format_datetime(date_str, time_str):
    dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
    return dt.strftime("%Y-%m-%d %H:%M")


def main(start_date, end_date):
    file_name = DATA_PATH
    existing_data = load_existing_data(file_name)
    print(f"Loaded {len(existing_data)} existing records")

    new_records_count = 0

    for date in date_range(start_date, end_date):
        new_data = fetch_data(date)
        if new_data is None:
            continue  # Skip this date if there was an error fetching data
        
        for record in new_data:
            # Check if taux_co2 is None and try to fetch again by datetime
            if record['taux_co2'] is None:
                datetime_str = format_datetime(record['date'], record['heure'])
                print("None data on date: ", datetime_str)
                new_record = fetch_data_by_datetime(datetime_str)
                if new_record and new_record['taux_co2'] is not None:
                    record['taux_co2'] = new_record['taux_co2']
                else:
                    continue  # Skip this record if taux_co2 is still None
            
            new_record = {
                'date': record['date'],
                'heure': record['heure'],
                'taux_co2': record['taux_co2']
            }
            
            # Remove duplicates with the same date and heure
            existing_data = [r for r in existing_data if not (r['date'] == new_record['date'] and r['heure'] == new_record['heure'])]
            
            # Append the new or updated record
            existing_data.append(new_record)
            new_records_count += 1

    # Sorting data by date and then by heure
    existing_data.sort(key=lambda x: (x['date'], x['heure']))
    save_data(existing_data, file_name)
    print(f"Total records after update: {len(existing_data)}")
    print(f"New records added: {new_records_count}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python get_co2.py <start_date> <end_date>")
        print("Where date format is : 'mm_dd_yyyy'")
        sys.exit(1)
    
    # Reformatting the date from mm_dd_yyyy to yyyy-mm-dd
    start_date = datetime.strptime(sys.argv[1], "%m_%d_%Y").strftime("%Y-%m-%d")
    end_date = datetime.strptime(sys.argv[2], "%m_%d_%Y").strftime("%Y-%m-%d")
    
    main(start_date, end_date)
