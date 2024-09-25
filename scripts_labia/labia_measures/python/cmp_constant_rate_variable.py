import datetime
import statistics
import pandas as pd
import matplotlib.pyplot as plt
import json
import os

def get_average_co2_rate_of_the_day(co2_data, date_str):
    """
    Given co2_data as a dictionary with keys 'date' and 'heure',
    and date_str in the format 'YYYY-MM-DDTHH:MM:SS',
    return the average CO2 rate for the whole day.
    """
    date = datetime.datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S').strftime('%Y-%m-%d')
    
    # Filter the CO2 data for the given date
    daily_co2_data = {key: value for key, value in co2_data.items() if key[0] == date}
    
    # Calculate the average CO2 rate for the day
    average_co2_rate = round(statistics.mean(daily_co2_data.values()), 2)

    return average_co2_rate

def convert_duration_to_seconds(duration_str):
    """
    Convert a duration string in the format '1h20min56s' or similar into seconds.
    """
    h, m, s = 0, 0, 0
    if 'h' in duration_str:
        h = int(duration_str.split('h')[0])
        duration_str = duration_str.split('h')[1]
    if 'min' in duration_str:
        m = int(duration_str.split('min')[0])
        duration_str = duration_str.split('min')[1]
    if 's' in duration_str:
        s = int(duration_str.split('s')[0])
    return h * 3600 + m * 60 + s

def plot_and_save(df, title, output_image_path):
    """Helper function to plot and save the graph"""
    plt.figure(figsize=(10, 6))
    plt.scatter(df.index, df['Real time Emission (gCO2eq)'], marker='o', linestyle='-', color='b', label='Real time Emission (gCO2eq)')
    plt.scatter(df.index, df['Daily Emission'], marker='o', linestyle='-', color='r', label='Daily Emission (gCO2eq)')
    plt.scatter(df.index, df['Annual Emission'], marker='o', linestyle='-', color='g', label='Annual Emission (gCO2eq)')
    plt.xlabel('Index')
    plt.ylabel('Emission (gCO2eq)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(output_image_path, bbox_inches='tight')
    plt.close()

def process_csv(file_path, co2_data, output_dir):
    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    df = pd.read_csv(file_path)

    # Convert job duration to seconds and add as a column
    df['Job Duration (s)'] = df['Total Duration'].apply(convert_duration_to_seconds)

    # Calculate total emission RTE and add it as a column
    df['Real time Emission (gCO2eq)'] = df['GPU Emission RTE (gCO2eq)'] + df['CPU Emission RTE (gCO2eq)']
    
    # Calculate total consumption in kWh and emissions
    df['Total Consumption (kWh)'] = (df['GPU Consumption (WH)'] + df['CPU Consumption (WH)']) / 1000
    
    # Daily emission using the average rate for the day
    df['Daily Emission'] = df.apply(lambda row: get_average_co2_rate_of_the_day(co2_data, row['start date']) * row['Total Consumption (kWh)'], axis=1)
    
    # Annual Emission using a constant factor of 32gCO2eq/kWh
    df['Annual Emission'] = df['Total Consumption (kWh)'] * 32
    
    # Define filters for different CO2 ranges
    filters = {
        'below_0_5gCO2eq': df[df['Real time Emission (gCO2eq)'] < 0.5],
        'between_0_5_and_2gCO2eq': df[(df['Real time Emission (gCO2eq)'] >= 0.5) & (df['Real time Emission (gCO2eq)'] < 2)],
        'between_2_and_15gCO2eq': df[(df['Real time Emission (gCO2eq)'] >= 2) & (df['Real time Emission (gCO2eq)'] < 15)],
        'above_or_equal_15gCO2eq': df[df['Real time Emission (gCO2eq)'] >= 15]
    }
    
    # Define sorting methods
    sorting_methods = {
        'sorted_by_real_time': 'Real time Emission (gCO2eq)',
        'sorted_by_daily': 'Daily Emission',
        'sorted_by_annual': 'Annual Emission'
    }
    
    # Generate plots
    for filter_name, filtered_df in filters.items():
        for sort_name, sort_column in sorting_methods.items():
            sorted_df = filtered_df.sort_values(by=sort_column).reset_index(drop=True)
            title = f"{filter_name.replace('_', ' ').capitalize()} - {sort_name.replace('_', ' ').capitalize()}"
            output_image_path = os.path.join(output_dir, f"{filter_name}_{sort_name}.png")
            plot_and_save(sorted_df, title, output_image_path)

def load_co2_data(co2_data_path):
    """Open the co2.json file and return its content as a dictionary indexed by date and heure"""
    with open(co2_data_path, 'r') as json_file:
        data = json.load(json_file)
    
    # Convert list of dictionaries to a dictionary indexed by 'date' and 'heure'
    co2_dict = {(record['date'], record['heure']): record['taux_co2'] for record in data}
    
    return co2_dict

# Exemple d'utilisation :
output_dir = './pic'
log_user_path = "/mnt/beegfs/projects/coca4ai/metrics/summary_per_user/bousquet.csv"
co2_data_path = "/mnt/beegfs/projects/coca4ai/metrics/co2.json"

co2_data = load_co2_data(co2_data_path)
process_csv(log_user_path, co2_data, output_dir)
