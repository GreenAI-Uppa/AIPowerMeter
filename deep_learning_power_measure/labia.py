import zipfile, os, json, sys
import subprocess
import pandas as pd
import traceback
import math
from tqdm import tqdm
import datetime
from deep_learning_power_measure.power_measure import experiment, parsers

from config import *

def get_user_job_folders(user, start=0, end=math.inf):
    """
    return the 
    """
    folders = []
    for dirpath, dirnames, filenames in os.walk(RECORDING_RAPL_NVIDI_DIR):
        if user in dirpath.replace(RECORDING_RAPL_NVIDI_DIR,''):
            for dirname in dirnames:
                year, month = dirpath.split('/')[-2].split('_')
                start_folder = datetime.datetime(int(year),int(month),1).timestamp()
                end_folder = (datetime.datetime(int(year),int(month)+1,1) - datetime.timedelta(days=1)).timestamp()
                if min(end_folder, end) - max(start_folder, start) > 0:
                    folders.append(os.path.join(dirpath, dirname))
    return folders

def get_list_user(start=0, end=math.inf):
    """return the list of users for which RAPL recording are present"""
    users = []
    for dirpath in os.listdir(RECORDING_RAPL_NVIDI_DIR):
        if os.path.isdir(os.path.join(RECORDING_RAPL_NVIDI_DIR,dirpath)):
            year, month = dirpath.split('_')
            start_folder = datetime.datetime(int(year),int(month),1).timestamp()
            end_folder = (datetime.datetime(int(year),int(month)+1,1) - datetime.timedelta(days=1)).timestamp()
            if min(end_folder, end) - max(start_folder, start) > 0:
                users += os.listdir(os.path.join(RECORDING_RAPL_NVIDI_DIR,dirpath))
    return users

def get_list_job(start=0, end=math.inf, users=[], nodes=[]):
    """get the jobs list given specified by the parameters
    Args: 
    start, end : float timestamps. Jobs will overlap with this temporal segment.
    users : if not empty, contain a list of username. The returned jobs only belong to these users
    nodes : if not empty, contain a list of compute nodes. The returned jobs have be run only on these nodes
    """
    jobs = {}
    for dirpath in os.listdir(RECORDING_RAPL_NVIDI_DIR):
        if os.path.isdir(os.path.join(RECORDING_RAPL_NVIDI_DIR,dirpath)):
            year, month = dirpath.split('_')
            start_folder = datetime.datetime(int(year),int(month),1).timestamp()
            end_folder = (datetime.datetime(int(year),int(month)+1,1) - datetime.timedelta(days=1)).timestamp()
            if min(end_folder, end) - max(start_folder, start) > 0:
                for user in os.listdir(os.path.join(RECORDING_RAPL_NVIDI_DIR,dirpath)):
                    if len(users)!=0 and user not in users:
                        continue
                    user_folder = os.path.join(RECORDING_RAPL_NVIDI_DIR,dirpath,user)
                    job_ids = os.listdir(user_folder)
                    for job_id in job_ids:
                        job_folder = os.path.join(user_folder,job_id)
                        meta_data = load_meta_data(job_folder)
                        node_name = meta_data['node_id']
                        start_job = datetime.datetime.fromisoformat(meta_data['start_date']).timestamp()
                        if 'end_date' not in meta_data:
                            end_job = datetime.datetime.today().timestamp()
                        else:
                            end_job = datetime.datetime.fromisoformat(meta_data['end_date']).timestamp()
                        #end = datetime.datetime.fromisoformat(meta_data['end_date']).timestamp()
                        if not experiment.is_iou(start, end, start_job, end_job):
                            continue
                        if len(nodes) > 0 and node_name not in nodes:
                            continue
                        jobs[job_id] = {'folder':job_folder, 'user':user}
    return jobs


def summarize(user_stats):
    summary = {}
    summary['number_of_jobs'] = len(user_stats)
    summary['gpu_consumption'] = sum([ v['rapl_nvidia']['gpu']['per_gpu_attributable_power']['all']  for (jobid, v) in user_stats.items()])
    summary['omegawatt_power_draw'] = sum([ v['omegawatt_power_draw']  for (jobid, v) in user_stats.items()])
    summary['cpu_consumption'] = sum([ v['rapl_nvidia']['cpu']['rel_intel_power']  for (jobid, v) in user_stats.items()])
    summary['total_duration'] = sum([ v['rapl_nvidia']['duration']  for (jobid, v) in user_stats.items()])
    return summary 
     
def get_summaries(all_users_stats):
    summaries = {}
    for user, user_stats in all_users_stats.items():
        summaries[user] = summarize(user_stats) 
    return summaries

def extract_value_for_each_job(user_stats, metrics):
    per_job_metrics = {}
    for (jobid, v) in user_stats.items():
        if 'gpu_consumption' in metrics:
            per_job_metrics['gpu_consumption'] = v['rapl_nvidia']['gpu']['per_gpu_attributable_power']['all']
        if 'omegawatt_power_draw' in metrics:
            per_job_metrics['omegawatt_power_draw'] = v['omegawatt_power_draw']
        if 'cpu_consumption' in metrics:
            per_job_metrics['cpu_consumption'] = v['rapl_nvidia']['cpu']['rel_intel_power']
        if 'duration' in metrics:
            per_job_metrics['duration'] = v['rapl_nvidia']['duration']
    return per_job_metrics

def print_summaries(summaries):
    global_stats = {}
    for k in ['number_of_jobs', 'gpu_consumption', 'cpu_consumption', 'total_duration', 'omegawatt_power_draw']:
        global_stats[k] =  sum([s[k] for s in summaries.values()])
    summaries['all'] = global_stats
    df = pd.DataFrame(summaries).transpose()
    df = df.rename(columns=mapping)
    df[mapping['omegawatt_power_draw']] = df[mapping['omegawatt_power_draw']].map(experiment.joules_to_kwh)
    df[mapping['cpu_consumption']] = df[mapping['cpu_consumption']].map(experiment.joules_to_kwh)
    df[mapping['gpu_consumption']] = df[mapping['gpu_consumption']].map(experiment.joules_to_kwh)
    print(df.round(2))


def get_stats_all_users(start=0, end=math.inf):
    all_user_stats = {}
    for user in tqdm(get_list_user(start, end)):
        all_user_stats[user] = get_user_stats(user, start=start, end=end)
    return all_user_stats

def get_slurm_exit_status(jobid):
    return subprocess.run(['sacct', '-X', '-n', '-o', 'State', '--j', jobid], stdout=subprocess.PIPE).stdout.decode('utf-8').replace('\n','').replace(' ','')

def get_stats_per_job_status(all_user_stats):
    per_status = {}
    for u, v in all_user_stats.items():
        for jobid, stats in v.items():
            status = get_slurm_exit_status(jobid)
            if status not in per_status:
                per_status[status] = {}
            per_status[status][jobid] = stats
    return per_status

def aiPowerMeterInfo(user, start=0, end=math.inf):
    job_folders = get_user_job_folders(user, start=start, end=end)
    metrics = {}
    for job_folder in tqdm(job_folders):
        driver = parsers.JsonParser(job_folder)
        try:
            exp_result = experiment.ExpResults(driver)
        except Exception as e:
            traceback.print_exc(limit=2, file=sys.stdout)
            continue
        _, se, ee = exp_result.get_exp_duration()
        if start  < ee and se < end: # there is overlap
            mtrcs = exp_result.get_summary(max(start,se), min(end,ee))
            meta_data = load_meta_data(job_folder) #json.load(open(os.path.join(job_folder,'meta_data.json')))
            node_name = meta_data["node_id"] 
            job_id = meta_data["jobid"]
            metrics[job_id] = {}
            metrics[job_id]['date'] = {'start':se,'end':ee}
            metrics[job_id]['node'] = node_name
            metrics[job_id]['rapl_nvidia'] = mtrcs
    return metrics

def rapl_nvidia_omegawatt_per_node(nodes, period=3600, meas_delta = 10, start=0, end=math.inf):
    """
    return the energy consumption measured by RAPL, Nvidia and powermeter for the given nodes on the given period
    Args: 
        - nodes : list of node names 
        - period : in seconds, intervall between two measurements
        - meas_delta : in seconds, value of the curve will be averaged on this delta
        - start, end : time stamps of the considered period
    Output: 
        per_node_measure : dictionnary where each key is a node name
                      each value is a dictionnary : 
                       where k is a timestamp
                       values are intel_power (rapl power), nvidia_draw_absolute and omegawatt_power_draw
    """
    # initialise the structure to store the results
    points = list(range(int(start),int(end),period))
    per_node_measure = {} 
    for n in nodes:
        per_node_measure[n] = {}
        for point in points:
            per_node_measure[n][point] = {}

    jobs = get_list_job()
    for job_id, v in tqdm(jobs.items()):
        job_folder = v['folder']
        meta_data = load_meta_data(job_folder)
        node_name = meta_data['node_id']
        start_ts = datetime.datetime.fromisoformat(meta_data['start_date']).timestamp()
        end_ts = datetime.datetime.fromisoformat(meta_data['end_date']).timestamp()
        points_on_this_job = [t for t in points if start_ts < t - meas_delta/2 and t + meas_delta/2 < end_ts ]
        if len(points_on_this_job) == 0:
            continue 
        if node_name not in per_node_measure:
            continue
        measures = per_node_measure[node_name]
        driver = parsers.JsonParser(job_folder)
        try:
          exp_result = experiment.ExpResults(driver)
        except Exception as e:
            traceback.print_exc(limit=2, file=sys.stdout)
            continue
        for t in points_on_this_job:
            if 'rapl' not in measures[t]:
                rapl_power_draw = exp_result.total_('intel_power', start=t-meas_delta/2,end=t+meas_delta/2) #total_('intel_power', start=t-meas_delta/2,end=t+meas_delta/2)
                nvidia_power_draw = exp_result.total_('nvidia_draw_absolute', start=t-meas_delta/2,end=t+meas_delta/2) #total_('intel_power', start=t-meas_delta/2,end=t+meas_delta/2)
                measures[t]['rapl'] = rapl_power_draw /meas_delta
                measures[t]['nvidia'] = nvidia_power_draw / meas_delta
    segments = []
    for node_name in nodes:
        segments += [(point - meas_delta/2, point + meas_delta/2, node_name, (point,node_name)) for point in points ]
        
    metrics = omegaWattInfo(segments)
    for node_name in nodes:
        for (point,node_name), v in metrics.items():
            per_node_measure[node_name][point]['omegawatt_power_draw'] = v['omegawatt_power_draw'] / meas_delta
    return per_node_measure


def load_meta_data(job_folder):
    meta_data = json.load(open(os.path.join(job_folder,'meta_data.json')))
    if 'end_date' not in meta_data:
        driver = parsers.JsonParser(job_folder)
        try:
            exp_result = experiment.ExpResults(driver)
        except Exception as e:
                print('end date missing for folder',job_folder)
                traceback.print_exc(limit=2, file=sys.stdout)
                return meta_data
        _, se, ee = exp_result.get_exp_duration()
        meta_data['end_date'] = datetime.datetime.fromtimestamp(ee).isoformat()
    return meta_data

def get_aiPowerMeterInfo_1job(job_folder):
    driver = parsers.JsonParser(job_folder)
    try:
        exp_result = experiment.ExpResults(driver)
    except Exception as e:
        traceback.print_exc(limit=2, file=sys.stdout)
        return None
    _, se, ee = exp_result.get_exp_duration()
    mtrcs = exp_result.get_summary()
    meta_data = load_meta_data(job_folder) #json.load(open(os.path.join(job_folder,'meta_data.json')))
    node_name = meta_data["node_id"]
    job_id = meta_data["jobid"]
    metrics = {}
    metrics['date'] = {'start':se,'end':ee}
    metrics['node'] = meta_data["node_id"] 
    metrics['rapl_nvidia'] = mtrcs
    metrics['job_id'] = meta_data["jobid"]
    if 'job_name' in meta_data:
        metrics['job_name'] = meta_data['job_name']
    return metrics

def omegaWattInfo(segments, start=0, end=math.inf):
    metrics = {}
    zip_files = get_zip_files(segments)
    for zip_file, segments in tqdm(zip_files.items()):
        curves = read_zip_file(zip_file, segments, start, end)
        for job_id, nodes in curves.items():
            if job_id not in metrics:
                metrics[job_id] = {}
            for node_name, curve in nodes.items():
                if 'omegawatt_power_draw' not in metrics[job_id]:
                    metrics[job_id]['omegawatt_power_draw'] = 0
                metrics[job_id]['omegawatt_power_draw'] += experiment.integrate(curve)[-1]
    return metrics

def get_user_stats(user, start=0, end=math.inf):
    """
    return a metric for each job for the given period
    """
    #recordings from AIPowerMeter
    metrics = aiPowerMeterInfo(user, start, end)
    # recordings from OmegaWatt
    # removing jobs which did not finish before the last omegawatt measures were recorded
    today = datetime.datetime.today()
    last_omegawatt_measures_date = datetime.datetime(today.year, today.month, today.day, 6, 59).timestamp()
    to_remove = []
    for job_id, data in metrics.items():
        end_job = data['date']['end']
        if last_omegawatt_measures_date < end_job:
            to_remove.append(job_id)
    for job_id in to_remove:
        del(metrics[job_id])

    segments = [(data['date']['start'], data['date']['end'],data['node'], job_id) for job_id, data in metrics.items()]
    metrics_omegawatt = omegaWattInfo(segments, start, end)
    # merge the two in one dict
    for job, v in metrics_omegawatt.items():
        metrics[job]['omegawatt_power_draw'] = metrics_omegawatt[job]['omegawatt_power_draw']
    return metrics

def get_zip_filename(year, month, day, usbport):
    filename = (str(month) if len(str(month))==2 else '0'+str(month))+'_'+(str(day) if len(str(day))==2 else '0'+str(day))+'_'+str(year)+'_USB'+str(usbport)+'.zip'
    return os.path.join(RECORDING_OMEGA_WATT_DIR, filename) 

def get_zip_files(segments):
    """
    this function is annoyingly complicated because each zip file is one day recording starting from 7am
    And you want to collect all files concerned by a set of temporal segments
    and the filename are called after the day of their last recording
    in practice, a database for temporal series would be simpler
    """
    zip_files = {}
    for start, end, node_name, job_id in tqdm(segments):
        ## 1 select through which usb port is recorded this node
        if node_name in node_to_column_usb0:
            usbport = 0
        elif node_name in node_to_column_usb1:
            usbport = 1
        else:
            ubsport = None

        ## 2 get the zip file corresponding to the start of the experiment
        start_date = datetime.datetime.fromtimestamp(start)   
        if start_date.hour < 7: 
            # this experiment started between 00:00 am and 7:00 am
            # so the beginning is stored in the file saved the same day at 7:00am
            end_of_first_zip_file = datetime.datetime(int(start_date.year), int(start_date.month), int(start_date.day), 7)
        else:
            # this experiments started between 7:00am and 23:59:am
            # so the beginning is stored at start day + 1 , 7:00am
            end_of_first_zip_file = datetime.datetime(int(start_date.year), int(start_date.month), int(start_date.day), 7) + datetime.timedelta(days=1)
        start_zip_file = get_zip_filename(end_of_first_zip_file.year, end_of_first_zip_file.month, end_of_first_zip_file.day, usbport)
        if start_zip_file not in zip_files:
            zip_files[start_zip_file] = []
        zip_files[start_zip_file].append((start_date.timestamp(), min(end, end_of_first_zip_file.timestamp()), node_name, job_id))

        ## 3 now searching for the end of the experiment and collecting the zip files in between
        end_date = datetime.datetime.fromtimestamp(end)
        if end_of_first_zip_file < end_date:# we need to find next zip file 
            ## 3.1 looping for the completed zipfiles of this experiment 
            # looping using the number of days between the end date and the end of the first zip file, which is already counted
            ndays = (end_date - datetime.timedelta(hours=7) - end_of_first_zip_file ).days + 1
            for day in range(ndays):
                file_date = end_of_first_zip_file + datetime.timedelta(days=day+1) #+1 because loop starts at 0
                zip_file = get_zip_filename(file_date.year, file_date.month, file_date.day, usbport)
                # one full day for this experiment
                if zip_file not in zip_files:
                    zip_files[zip_file] = []
                zip_files[zip_file].append(((file_date - datetime.timedelta(days=1)).timestamp(), min(end_date.timestamp(), file_date.timestamp()), node_name, job_id))
            ## 3.2 adding the last zip file if required
            if 7 < end_date.hour:
                last_file_date = end_date 
                zip_file = get_zip_filename(last_file_date.year, last_file_date.month, last_file_date.day, usbport)
                if zip_file not in zip_files:
                    zip_files[zip_file] = []
                    zip_files[zip_file].append(((last_file_date-datetime.timedelta(days=-1)).timestamp(), min(end_date.timestamp(), file_date.timestamp()), node_name, job_id))
    return zip_files

def head_zip_file(zip_file):
    myzip = zipfile.ZipFile(zip_file)
    f = myzip.open('log_omegawatt_of_the_day')
    for i in range(10):
        print(f.readline())

def extract_header(zip_file,usbport=0):
    myzip = zipfile.ZipFile(zip_file)
    f = myzip.open('log_omegawatt_of_the_day')
    f.readline()
    header = f.readline().decode('utf-8')[:-1].split(',')
    node_name_to_csv_column = {}
    if usbport == 0:
        node_to_omegawatt_column = node_to_column_usb0
    elif usbport == 1:
        node_to_omegawatt_column = node_to_column_usb1
    else:
        return None
    for n, v in  node_to_omegawatt_column.items():
        node_name_to_csv_column[n] = []
        for col in v['columns']:
            for i, h in enumerate(header):
                if h == '#activepow'+str(col):
                    node_name_to_csv_column[n].append(i)
                    break
    return node_name_to_csv_column

def read_zip_file(zip_file, segments, start=0, end=math.inf):
    """return power consumption curve during each job on all the machines used by this job 
    input:
      - segments : list [(s,e,node_name),... ]
      - start, end : eventually restrict the recording data to this period
      - zip_file : omegawatt log
    return 
    curves { job_id: { node_name, [(date, value),.... ]
    """
    curves = {}
    for (s, e, node_name, job_id) in  segments:
        if job_id not in curves:
            curves[job_id] = {}
        if node_name not in curves[job_id]:
            curves[job_id][node_name] = []
    if not os.path.isfile(zip_file):
        return curves
    myzip = zipfile.ZipFile(zip_file)
    f = myzip.open('log_omegawatt_of_the_day')
    # read the first two lines we don't need
    f.readline()
    header = f.readline().decode('utf-8')[:-1].split(',')
    for l in f:
        # now read the measures line by line
        recordings = l.decode('utf-8').replace('\n','').split(',')
        if len(recordings)<2 or recordings[1] != 'true':
            continue
        timestamp = float(recordings[0])
        # and if the line fall into one of the segments, we keep the value
        for (s,e,node_name, job_id) in  segments:
            if node_name not in node_name_to_csv_column:
                continue
            if (timestamp < start or end < timestamp) or (timestamp < s or e < timestamp ):
                continue
            power = sum([ float(recordings[c]) for c in node_name_to_csv_column[node_name]])
            curves[job_id][node_name].append({'date':timestamp,'value':power})
    return curves
