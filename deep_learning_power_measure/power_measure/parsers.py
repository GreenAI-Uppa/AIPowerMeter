"""Contains the parsers to read the recordings. Currently, only contain parsing for json"""
import json
import os
import datetime
import shutil
import pandas as pd
import subprocess
import logging

SEGMENT_END = "END OF RECORDING SEGMENT"
class JsonParser():
    """write and parse the measurement recording from and to json files"""
    def __init__(self, location : str):
        """
        JSonParser will save the recordings of the experiment as json files
        folder : the location where the json will be saved, it will be created if it does not exist
        cont : if set to False and the parameter folder is an existing directory, then previous recordings will be erased. If set to True, new recordings will be appended to existing ones
        """
        self.folder = location
        self.wattmeter_logfile = os.path.join(self.folder, 'omegawatt.csv')
        self.wattemeter_exec = "/home/ntirel/libs/wattmeter_rapid_omegawatt/wattmetre-read"
        self.power_metric_filename = self.folder + '/power_metrics.json'
        self.exp_metric_filename = self.folder + '/results_exp.json'
        self.model_card_file = os.path.join(self.folder,'model_summary.json')
        if not os.path.isdir(self.folder):
            os.makedirs(self.folder)


    def erase(self):
        """delete the recordings"""
        if  os.path.isdir(self.folder):
            shutil.rmtree(self.folder)
            os.makedirs(self.folder)

    def save_model_card(self, model_card):
        """save the model card as a json file"""
        json.dump(model_card, open(self.model_card_file, 'w'))

    def save_power_metrics(self, metrics):
        """"save the power and CPU/GPU usage related metrics"""
        if not os.path.isdir(self.folder):
            os.makedirs(self.folder)
        power_metric_fileout = open(self.power_metric_filename,'a')
        today_str = datetime.datetime.now().__str__()
        data = { 'date': today_str, 'metrics': metrics }
        json_str = json.dumps(data)
        power_metric_fileout.write(json_str+'\n')

    def close_driver(self):
        power_metric_fileout = open(self.power_metric_filename,'a')
        json_str = json.dumps(SEGMENT_END)
        power_metric_fileout.write(json_str+'\n')

    def save_wattmeter_metrics(self):
        """save the model card as a json file"""
        if not os.path.isdir(self.folder):
            os.makedirs(self.folder)
        print("OMMEGAWATT")
        print(self.wattemeter_exec + f" --tty=/dev/ttyUSB0 --nb=6 > {self.wattmeter_logfile} 2>&1 &")
        proc = subprocess.Popen(self.wattemeter_exec + f" --tty=/dev/ttyUSB0 --nb=6 > {self.wattmeter_logfile} 2>&1 &", shell=True, preexec_fn=os.setsid)
        return proc
        #os.system(f"libs/wattmeter_rapid_omegawatt/wattmetre-read --tty=/dev/ttyUSB0 --nb=6 > {self.wattmeter_logfile} 2>&1 & echo $! > /tmp/pid")

    def save_exp_metrics(self, metrics):
        """save experiment (accuracy, latency,...) related metrics"""
        if not os.path.isdir(self.folder):
            os.makedirs(self.folder)
        exp_metric_fileout = open(self.exp_metric_filename,'a')
        today_str = datetime.datetime.now().__str__()
        data = { 'date': today_str, 'metrics': metrics }
        json_str = json.dumps(data)
        exp_metric_fileout.write(json_str+'\n')

    def get_model_card(self):
        """read the json containing the model card"""
        if os.path.isfile(self.model_card_file):
            return json.load(open(self.model_card_file))

    def get_segments(self):
        """
        get time stamps where recordings have been stopped AND re-initiated later
        """
        segments = []
        if os.path.isfile(self.power_metric_filename):
            for line in open(self.power_metric_filename):
                result = json.loads(line)
                if 'date' in result:
                    last_date = datetime.datetime.fromisoformat(result['date'])
                elif result==SEGMENT_END:
                    segments.append(last_date)
            if len(segments) != 0: # there are actually segments in this file 
                segments.pop() # We don't need the last one, we know the file is over
        return segments

    def load_cpu_metrics(self):
        """load the metrics related to the cpu usage and energy consumption"""
        if os.path.isfile(self.power_metric_filename):
            metrics = {}
            for line in open(self.power_metric_filename):
                result = json.loads(line)
                if isinstance(result,dict) and 'cpu' in result['metrics']:
                    date = datetime.datetime.fromisoformat(result['date'])
                    for k, v in result['metrics']['cpu'].items():
                        if isinstance(v, dict):
                            v = sum(v.values())
                        if k not in metrics:
                            metrics[k] = {'dates':[], 'values':[]}
                        metrics[k]['dates'].append(date)
                        metrics[k]['values'].append(v)
            if len(metrics) == 0:
                return None
            return metrics
        return None

    def load_gpu_metrics(self):
        """load the metrics related to the GPU usage and energy consumption"""
        if os.path.isfile(self.power_metric_filename):
            metrics = {} #Streaming Processor / Shared Processor sm
            for line in open(self.power_metric_filename):
                result = json.loads(line)
                if  isinstance(result,dict) and 'gpu' in result['metrics']:
                    date = datetime.datetime.fromisoformat(result['date'])

                    v = result['metrics']['gpu']['nvidia_draw_absolute']
                    if 'nvidia_draw_absolute' not in metrics:
                        metrics['nvidia_draw_absolute'] = {'dates':[], 'values':[]}
                    metrics['nvidia_draw_absolute']['dates'].append(date)
                    metrics['nvidia_draw_absolute']['values'].append(v)
                    
                    v = result['metrics']['gpu']['per_gpu_attributable_power']['all']
                    if 'nvidia_attributable_power' not in metrics:
                        metrics['nvidia_attributable_power'] = {'dates':[], 'values':[]}
                    metrics['nvidia_attributable_power']['dates'].append(date)
                    metrics['nvidia_attributable_power']['values'].append(v)

                    per_gpu_mem_use = result['metrics']['gpu']['per_gpu_attributable_mem_use']
                    if 'nvidia_mem_use' not in metrics:
                        metrics['nvidia_mem_use'] = {}
                    for gpu, mem_uses in per_gpu_mem_use.items():
                        if gpu not in metrics['nvidia_mem_use']:
                            metrics['nvidia_mem_use'][gpu] = {'dates':[], 'values':[]}
                        v = [v for pid, v in mem_uses.items() if v is not None ]
                        if len(v) == 0:
                            v = 0
                        else:
                            v = sum(v)
                        metrics['nvidia_mem_use'][gpu]['dates'].append(date)
                        metrics['nvidia_mem_use'][gpu]['values'].append(v)

                    per_gpu_sm_use = result['metrics']['gpu']['per_gpu_absolute_percent_usage']
                    if 'nvidia_sm_use' not in metrics:
                        metrics['nvidia_sm_use'] = {} 
                    for gpu, sm_use in per_gpu_sm_use.items():
                        if gpu not in metrics['nvidia_sm_use']:
                            metrics['nvidia_sm_use'][gpu] = {'dates':[], 'values':[]}
                        metrics['nvidia_sm_use'][gpu]['dates'].append(date)
                        metrics['nvidia_sm_use'][gpu]['values'].append(sm_use)

                    per_gpu_power_draw = result['metrics']['gpu']['per_gpu_power_draw']
                    if 'per_gpu_power_draw' not in metrics:
                        metrics['per_gpu_power_draw'] = {} 
                    for gpu, sm_use in per_gpu_power_draw.items():
                        if gpu not in metrics['per_gpu_power_draw']:
                            metrics['per_gpu_power_draw'][gpu] = {'dates':[], 'values':[]}
                        metrics['per_gpu_power_draw'][gpu]['dates'].append(date)
                        metrics['per_gpu_power_draw'][gpu]['values'].append(sm_use)

                    per_gpu_attributable_power = result['metrics']['gpu']['per_gpu_attributable_power']
                    if 'per_gpu_attributable_power' not in metrics:
                        metrics['per_gpu_attributable_power'] = {} 
                    for gpu, sm_use in per_gpu_attributable_power.items():
                        if gpu not in metrics['per_gpu_attributable_power']:
                            metrics['per_gpu_attributable_power'][gpu] = {'dates':[], 'values':[]}
                        metrics['per_gpu_attributable_power'][gpu]['dates'].append(date)
                        metrics['per_gpu_attributable_power'][gpu]['values'].append(sm_use)

                    per_gpu_estimated_attributable_utilization = result['metrics']['gpu']['per_gpu_estimated_attributable_utilization']
                    if 'per_gpu_estimated_attributable_utilization' not in metrics:
                        metrics['per_gpu_estimated_attributable_utilization'] = {} 
                    for gpu, sm_use in per_gpu_estimated_attributable_utilization.items():
                        if gpu not in metrics['per_gpu_estimated_attributable_utilization']:
                            metrics['per_gpu_estimated_attributable_utilization'][gpu] = {'dates':[], 'values':[]}
                        metrics['per_gpu_estimated_attributable_utilization'][gpu]['dates'].append(date)
                        metrics['per_gpu_estimated_attributable_utilization'][gpu]['values'].append(sm_use)
            if len(metrics) == 0:
                return None
            return metrics
        return None

    def load_exp_metrics(self):
        """load the experiment (accuracy, latency,...) related metrics"""
        if os.path.isfile(self.exp_metric_filename):
            results = json.load(open(self.exp_metric_filename))
            metrics = {}
            for result in results:
                if isinstance(results, dict):
                    result = results[result]
                date = datetime.datetime.fromisoformat(result['end_training_epoch'])
                for k in result:
                    if k == 'end_training_epoch':
                        continue
                    if k not in metrics:
                        metrics[k] = {'dates':[], 'values':[]}
                    metrics[k]['dates'].append(date)
                    metrics[k]['values'].append(result[k])
            return metrics
        return None

    def load_wattmeter_metrics(self):
        """load the metrics related to the wattmeter"""
        if os.path.isfile(self.wattmeter_logfile):
            try:
                results = pd.read_csv(self.wattmeter_logfile, header=1)
            except pd.errors.ParserError as e:
                print(e)
                return None
            # drop last line to make sure to have a good csvyyy
            results.drop(results.tail(1).index,inplace=True)
            # convert to boolean
            if '#frame_is_ok' not in results:
                logging.error("ERROR check the omegawatt csv")
                return None
            if results['#frame_is_ok'].dtype != bool:
                logging.warning('omegawatt csv column #frame_is_ok is not loaded as boolean, trying to cast from "false" and "true"')  
                results['#frame_is_ok'] = results['#frame_is_ok'].map({'true': True, 'false': False})
            results = results[results["#frame_is_ok"]]
            metrics = {}
            for k in results.columns:
                if k not in metrics:
                    metrics[k] = {'dates':[], 'values':[]}
                metrics[k]['dates'] = results["#timestamp"].values
                metrics[k]['values'] = results[k].values
            if len(metrics) == 0:
                return None
            return metrics
        return None

    def load_metrics(self):
        """load all metrics. Returns None if the metric is not available"""
        cpu_metrics = self.load_cpu_metrics()
        gpu_metrics = self.load_gpu_metrics()
        exp_metrics = self.load_exp_metrics()
        wattmeter_metrics = self.load_wattmeter_metrics()
        return cpu_metrics, gpu_metrics, exp_metrics, wattmeter_metrics
