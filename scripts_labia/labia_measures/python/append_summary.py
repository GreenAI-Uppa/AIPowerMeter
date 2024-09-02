import os
import json
from tqdm import tqdm
import datetime
import statistics
from deep_learning_power_measure.power_measure import experiment
from deep_learning_power_measure import labia
import glob
import csv
import subprocess
import sys
import argparse

# column of the resulting csv files
fields = ['node', 'job_id', 'date', 'total_duration', 'gpu_consumption', 'gpu_ram', 'gpu_sm_usage', 'cpu_consumption', 'cpu_ram_abs', 'relative_cpu_usage','gpu_emission_rte','gpu_emission_ademe','cpu_emission_rte','cpu_emission_ademe']

# where are stored the csv files
log_user_folder = "/mnt/beegfs/projects/coca4ai/metrics/summary_per_user"

co2_data_path = "/mnt/beegfs/projects/coca4ai/metrics/co2.json"

def get_list_job_csv(start=0, end=float('inf'), folder="/mnt/beegfs/projects/coca4ai/metrics/summary_per_user"):
    """Return the list of jobs already summarised in the CSV files within the specified time range"""
    jobs_listed = []
    csv_files = glob.glob(folder + '/*.csv')
    
    if not csv_files:
        print(f"Aucun fichier CSV trouvé dans le dossier: {folder}")
        return jobs_listed
    
    for csv_file in csv_files:
        with open(csv_file, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for fields in reader:
                if len(fields) > 1 and fields[1] != 'job_id':  # if the line is not empty and not the header
                    try:
                        job_start = datetime.datetime.fromisoformat(fields[2]).timestamp()
                        if start <= job_start <= end:
                            jobs_listed.append(fields[1])
                    except ValueError as e:
                        print(f"Erreur de conversion de la date dans le fichier {csv_file}: {e}")
    
    return jobs_listed

# Get the average GPU SM usage, and considering that a GPU is used if at least one of the recorded values is non-zero
def get_gpu_sm_usage(gpu_sm_data):
    total_usage = 0
    gpu_count = 0

    for gpu, data in gpu_sm_data.items():
        values = data['values']
        # Skip GPUs with no usage
        if all(v == 0 for v in values):
            continue
        # average for each GPU
        avg_usage = sum(values) / len(values)
        total_usage += avg_usage
        gpu_count += 1
    
    if gpu_count == 0:
        return 0  # If no GPU is used, return 0
    else:
        return total_usage / gpu_count
    
## Functions for CO2 data


def integrate_dictionnary(metrics):
    """integral of the metric values over time
    list : a list of dictionnaries with keys 'date' and 'value'
    same thing as integrate but returns a list of integrated values instead of total
    """
    newdict = {}
    for i in range(len(metrics) - 1):
        x1 = metrics[i]['date']
        x2 = metrics[i + 1]['date']
        x2 = metrics[i+1]['date']
        y1 = metrics[i]['value']
        y2 = metrics[i + 1]['value']
        
        # Integration with trapeze method
        if ((x2 - x1) < 0) and ((y2 + y1)<0):
            print("NEGATIVE TIME STEP")
            if ((y2 + y1)<0):
                print("y2+y1>0, y2 :",y2,"y1:",y1)
            elif ((x2 - x1) < 0):
                print("x2-x1>0, x2 :",x2,"x1:",x1)


        integrated_value = (x2 - x1) * (y2 + y1) / 2
        
        # add value to the dictionnary
        if x1 not in newdict:
            newdict[x1] = 0
        newdict[x1] += integrated_value
    
    return newdict

def convert_dates(joules_dict):
    readable_joules_dict = {}
    for timestamp, joules in joules_dict.items():
        # convert timestamp to datetime object
        dt = datetime.datetime.fromtimestamp(timestamp)
        # Format the date as a readable string
        readable_date = dt.strftime('%Y-%m-%d %H:%M:%S') # Add to the new dictionary
        readable_joules_dict[readable_date] = joules
    return readable_joules_dict


# Specific functions for CO2 data

ADEME_CO2_RATE = 52 # Average carbon intensity in France in gCO2eq/kWh https://base-empreinte.ademe.fr/

def load_co2_data():
    """Open the co2.json file and return its content as a dictionary indexed by date and heure"""
    file_path = co2_data_path
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    
    # Convert list of dictionaries to a dictionary indexed by 'date' and 'heure'
    co2_dict = {(record['date'], record['heure']): record['taux_co2'] for record in data}
    
    return co2_dict

def get_closest_date(date_str):
    """
    Takes a date string in the format 'YYYY-MM-DD HH:MM:SS' and returns the same date
    rounded down to the nearest 15-minute interval.
    """
    dt = datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
    
    # Find the closest previous quarter hour
    new_minute = (dt.minute // 15) * 15
    
    # Return the new datetime with the minutes rounded down to the closest quarter hour
    rounded_dt = dt.replace(minute=new_minute, second=0, microsecond=0)
    
    return rounded_dt.strftime('%Y-%m-%d %H:%M')

def fetch_new_co2_data(date):
    """Fetch new CO2 data for the given date by calling get_co2.py."""
    print("FETCH_NEW_DATA ENTERED")
    formatted_date = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S').strftime('%d/%m/%Y')
    subprocess.run(['python', 'get_co2.py', formatted_date, formatted_date])
    print("FETCH_NEW_DATA EXITED")

def get_co2_rate(co2_data, date_str):
    """
    Given co2_data as a dictionary with keys 'date' and 'heure',
    and date_str in the format 'YYYY-MM-DD HH:MM:SS',
    return the CO2 rate at the closest date and heure, rounded down to the nearest 15-minute interval.
    """
    closest_datetime = get_closest_date(date_str)
    date, time = closest_datetime.split(' ')

    # Direct lookup in the dictionary using the tuple (date, time)
    if (date, time) in co2_data:
        return co2_data[(date, time)]
    
    # If no record is found, return None
    return None

def get_average_co2_rate_of_the_day(co2_data, date_str):
    """
    Given co2_data as a dictionary with keys 'date' and 'heure',
    and date_str in the format 'YYYY-MM-DD HH:MM:SS',
    return the average CO2 rate for the whole day.
    """
    date = datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')
    
    # Filter the CO2 data for the given date
    daily_co2_data = {key: value for key, value in co2_data.items() if key[0] == date}
    
    # Calculate the average CO2 rate for the day
    average_co2_rate = round(statistics.mean(daily_co2_data.values()),2)

    return average_co2_rate

# Config for the script
parser = argparse.ArgumentParser(description='Program to add user data from jobs to csv files')
parser.add_argument('-d', '--daily', action='store_true', help='Get jobs from the last 24 hours')
parser.add_argument('-a', '--all', action='store_true', help='Get all jobs until yesterday (not today because we are not sure to have the CO2 data for today)')
parser.add_argument('-c', '--custom', nargs=2, metavar=('days_start', 'days_end'), type=int, help='Get jobs from custom time range, in days')

args = parser.parse_args()

if args.daily:
    start_time = (datetime.datetime.now() - datetime.timedelta(days=70)).timestamp()
    end_time = (datetime.datetime.now() - datetime.timedelta(days=1)).timestamp()
    all_jobs = labia.get_list_job(start=start_time, end=end_time)
    all_jobs_in_csv = get_list_job_csv(start=start_time, end=end_time)
elif args.all:
    end_time = (datetime.datetime.now() - datetime.timedelta(days=1)).timestamp()
    all_jobs = labia.get_list_job(end=end_time)
    all_jobs_in_csv = get_list_job_csv(end=end_time)
elif args.custom:
    days_start = args.custom[0]
    days_end = args.custom[1]
    start_time = (datetime.datetime.now() - datetime.timedelta(days=days_start)).timestamp()
    end_time = (datetime.datetime.now() - datetime.timedelta(days=days_end)).timestamp()
    all_jobs = labia.get_list_job(start=start_time, end=end_time)
    all_jobs_in_csv = get_list_job_csv(start=start_time, end=end_time)
else:
    print("Usage: python summary.py [-d | --daily] for -48h to -24h or [-a | --all] for all jobs until yesterday or [-c <days_start> <days_end>] for custom range")
    exit(0)

# get all the jobs not already summarised in the csv files
jobs_per_user_sorted = {}
for jobid, v in all_jobs.items():
    if jobid in all_jobs_in_csv:
        continue
    u = v['user']
    if u not in jobs_per_user_sorted:
        jobs_per_user_sorted[u] = []
    if 'start_date' not in v:
        start_date = datetime.datetime.now().timestamp()
    else:
        start_date = datetime.datetime.fromisoformat(v['start_date']).timestamp()
    jobs_per_user_sorted[u].append((start_date, jobid))

# For each job, summarise it and append the summary to its user csv file
print('appending', sum([len(v) for (u, v) in jobs_per_user_sorted.items()]), 'new jobs')
for user, job_ids in tqdm(jobs_per_user_sorted.items()):
    for start_date, job_id in job_ids:
        # read and summarise for this job
        print("adding job", job_id)
        v = all_jobs[job_id]
        user = v['user']
        job_folder = v['folder']
        metrics = labia.get_aiPowerMeterInfo_1job(job_folder)
        co2_data = load_co2_data()
        if metrics is None:
            continue

         # and append the results to the csv file
        output_file = os.path.join(log_user_folder, user) + '.csv'
        if not os.path.isfile(output_file):
            with open(output_file, 'a') as of:
                fields_csv = [labia.mapping[f] for f in fields]
                of.write(','.join(fields_csv) + '\n')

        with open(output_file, 'a') as of:
            l = []

            # Variable to skip the job, if true we dont record it
            skip_job = False

            gpu_list_joules = convert_dates(metrics['rapl_nvidia']['gpu']['joules_list'])

            # We calculate the total joules consumed by the GPU and the CO2 emissions here, for not having to go through the dictionary more than once
            cpu_list_joules = convert_dates(metrics['rapl_nvidia']['cpu']['joules_list'])
            tot_joulesCPU=0
            co2_cpu = 0
            for date, joules in cpu_list_joules.items():
                if (joules > 0):
                    tot_joulesCPU+=joules
                    co2_rate = get_co2_rate(co2_data, date)
                    if co2_rate == None:
                        skip_job = True
                        print("CO2 rate is None, date is :", date)
                    else :
                        co2_cpu += joules*co2_rate
            co2_cpu = round(co2_cpu/3_600_000, 2)
            cpu_cons_wh = round(experiment.joules_to_wh(tot_joulesCPU), 2)  # Conversion from joules to watt-hours


            # Same for GPU
            co2_gpu=0
            tot_joulesGPU=0
            for date,joules in gpu_list_joules.items():
                # it seems that we have to divide by 2 the joules for the gpu 
                # I don't know why, but when I compare the results to the old method I have to divide per two to get the same results
                joules=joules/2
                if joules > 0:
                    co2_rate = get_co2_rate(co2_data, date)
                    tot_joulesGPU+=joules
                    if co2_rate == None:
                        skip_job = True
                        print("CO2 rate is None, date is :", date)
                    else :
                        co2_gpu += joules*co2_rate
            co2_gpu = round(co2_gpu/3_600_000,2)
            gpu_cons_wh = round(experiment.joules_to_wh(tot_joulesGPU), 2)  # Conversion from joules to watt-hours         

            for f in fields:
                if f == 'date':
                    start_date = datetime.datetime.fromtimestamp(metrics['date']['start'])
                    new_f = start_date.replace(microsecond=0).isoformat()
                    
                elif f == 'total_duration':
                    duration = round(metrics['rapl_nvidia']['duration'])
                    if duration == 0:
                        skip_job = True
                        print("Skipping job", job_id, "because duration is 0")
                    hours = duration // 3600
                    minutes = (duration % 3600) // 60
                    seconds = duration % 60
                    if (hours > 0):
                        new_f = f"{hours}h{minutes}min{seconds}s"
                    elif (minutes > 0):
                        new_f = f"{minutes}min{seconds}s"
                    else:
                        if (seconds == 0):
                            skip_job = True
                            print("Skipping job", job_id, "because duration is 0")
                        new_f = f"{seconds}s"

                elif f == 'node':
                    new_f = metrics['node']

                elif f == 'job_id':
                    new_f = metrics['job_id']

                elif f == 'gpu_consumption':
                    # We already calculated the total GPU consumption in watt-hours
                    new_f = str(gpu_cons_wh)

                elif f == 'cpu_consumption':
                    # We already calculated the total CPU consumption in watt-hours
                    new_f = str(cpu_cons_wh)

                elif f == 'cpu_ram_abs':
                    cpu_ram = metrics['rapl_nvidia']['cpu']['mem_use_abs']
                    new_f = experiment.humanize_bytes(cpu_ram)

                elif f == 'relative_cpu_usage':
                    relative_cpu_usage = metrics['rapl_nvidia']['cpu']['relative_cpu_use']
                    new_f = str(round(relative_cpu_usage * 100, 2))

                elif f == 'gpu_ram':
                    gpu_ram = sum(metrics['rapl_nvidia']['gpu']['nvidia_mem_use_abs'].values())
                    new_f = experiment.humanize_bytes(gpu_ram)

                elif f == 'gpu_sm_usage':
                    gpu_sm_list = metrics['rapl_nvidia']['gpu']['nvidia_sm_list']
                    gpu_usage = get_gpu_sm_usage(gpu_sm_list['nvidia_sm_use'])
                    new_f = str(round(gpu_usage * 100, 2))

                elif f == 'cpu_emission_rte':
                    # We already calculated the total CPU consumption in watt-hours
                    new_f = str(co2_cpu)

                elif f == 'gpu_emission_rte':
                    # We already calculated the total GPU consumption in watt-hours
                    new_f = str(co2_gpu)

                elif f == 'cpu_emission_ademe':
                    # We already calculated the total CPU consumption in joules
                    cpu_cons_kwh = experiment.joules_to_kwh(tot_joulesCPU)
                    cpu_emission = round(cpu_cons_kwh * ADEME_CO2_RATE,2)
                    new_f = str(cpu_emission)

                elif f == 'gpu_emission_ademe':
                    # We already calculated the total GPU consumption in joules
                    gpu_cons_kwh = experiment.joules_to_kwh(tot_joulesGPU)
                    gpu_emission = round(gpu_cons_kwh * ADEME_CO2_RATE,2)
                    new_f = str(gpu_emission)
                l.append(str(new_f))
            if not (skip_job):

                of.write(','.join(l) + '\n')
            else:
                print("Skipping job", job_id)