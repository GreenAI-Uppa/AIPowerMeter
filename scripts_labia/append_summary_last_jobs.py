import os
from tqdm import tqdm
import datetime
import statistics
from deep_learning_power_measure.power_measure import experiment
from deep_learning_power_measure import labia
import glob, csv


# column of the resulting csv files
fields = ['node', 'job_id', 'date', 'total_duration', 'gpu_consumption', 'gpu_ram', 'gpu_sm_usage', 'cpu_consumption', 'cpu_ram_abs', 'relative_cpu_usage']

# where are stored the csv files
log_user_folder = "/mnt/beegfs/home/gay/coca4ai/recordings/summary_per_user/"

def get_list_job_csv(folder="/mnt/beegfs/home/gay/coca4ai/recordings/summary_per_user/"):
    """return the list of jobs already summarised in the csv files"""
    jobs_listed = []
    for csv_file in glob.glob(folder+'/*.csv'):
        for fields in csv.reader(open(csv_file), delimiter= ','):
          if fields[1] != 'job_id':
            jobs_listed.append(fields[1])
    return jobs_listed

all_jobs_in_csv = get_list_job_csv()

all_jobs = labia.get_list_job()

# get all the jobs not already summarised in the csv files
# and sort them with their end date
jobs_per_user_sorted = {}
for jobid, v in all_jobs.items():
    if jobid in all_jobs_in_csv:
        continue
    u = v['user']
    if u not in jobs_per_user_sorted:
        jobs_per_user_sorted[u] = []
    end_date = datetime.datetime.fromisoformat(end_date).timestamp()
    jobs_per_user_sorted[u].append((end_date,jobid))


# For each job, summarise it and append the summary to its user csv file
print('appending',sum([len(v) for (u,v) in jobs_per_user_sorted.items()]),'new jobs')
for user, job_ids in tqdm(jobs_per_user_sorted.items()):
  for e, job_id in sorted(job_ids):
    # read and summarise for this job
    v = all_jobs[job_id]
    user = v['user']
    job_folder = v['folder']
    metrics = labia.get_aiPowerMeterInfo_1job(job_folder)
    if metrics == None:
        continue

    # and append the results to the csv file
    output_file = os.path.join(log_user_folder,user)+'.csv'
    if not os.path.isfile(output_file):
        of = open(output_file,'a')
        fields_csv = [labia.mapping[f] for f in fields]
        of.write(','.join(fields_csv)+'\n')
    else:
        of = open(output_file,'a')
    l = []
    for f in fields:
        if f == 'date':
            start_date = datetime.datetime.fromtimestamp(metrics['date']['start'])
            new_f = start_date.isoformat()
        elif f == 'total_duration':
            duration = metrics['rapl_nvidia']['duration']
            new_f = str(duration)
        elif f == 'node':
            new_f = metrics['node']
        elif f == 'job_id':
            new_f = metrics['job_id']
        elif f == 'gpu_consumption':
            gpu_cons = metrics['rapl_nvidia']['gpu']['per_gpu_attributable_power']['all']
            new_f = experiment.joules_to_kwh(gpu_cons)
        elif f == 'cpu_consumption':
            cpu_cons = metrics['rapl_nvidia']['cpu']['rel_intel_power']
            new_f = experiment.joules_to_kwh(cpu_cons)
        elif f == 'cpu_ram_abs':
            cpu_ram = metrics['rapl_nvidia']['cpu']['mem_use_abs']
            new_f = experiment.humanize_bytes(cpu_ram)
        elif f == 'relative_cpu_usage':
            new_f = metrics['rapl_nvidia']['cpu']['relative_cpu_use']
        elif f == 'gpu_ram':
            gpu_ram = sum(metrics['rapl_nvidia']['gpu']['nvidia_mem_use_abs'].values())
            new_f = experiment.humanize_bytes(gpu_ram)
        elif f == 'gpu_sm_usage':
            new_f = statistics.mean(metrics['rapl_nvidia']['gpu']['nvidia_average_sm'].values())
        l.append(str(new_f))
    of.write(','.join(l)+'\n')
of.close()
