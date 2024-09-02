import os
import pandas as pd

def process_csv_file(file_path):
    df = pd.read_csv(file_path)
    df_cleaned = df[df['Total Duration'] != '0s']
    df_cleaned.to_csv(file_path, index=False)

def process_csv_folder(folder_path):
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            process_csv_file(file_path)


folder_path = '/mnt/beegfs/home/saesvin/coca4ai/metrics/summary_per_user'  
process_csv_folder(folder_path)