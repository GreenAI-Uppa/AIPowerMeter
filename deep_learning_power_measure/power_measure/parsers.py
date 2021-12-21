"""Contains the parsers to read the recordings. Currently, only contain parsing for json"""
import json
import os
import datetime
import shutil

class JsonParser():
    """write and parse the measurement recording from and to json files"""
    def __init__(self, location : str):
        """
        JSonParser will save the recordings of the experiment as json files
        folder : the location where the json will be saved, it will be created if it does not exist
        cont : if set to False and the parameter folder is an existing directory, then previous recordings will be erased. If set to True, new recordings will be appended to existing ones
        """
        self.folder = location
        self.power_metric_filename = self.folder + '/power_metrics.json'
        self.exp_metric_filename = self.folder + '/results_exp.json'
        self.model_card_file = os.path.join(self.folder,'model_summary.json')


    def erase(self):
        """delete the recordings"""
        if  os.path.isdir(self.folder):
            shutil.rmtree(self.folder)
            os.makedirs(self.folder)

    def save_model_card(self, model_card):
        """save the model card as a json file"""
        if not os.path.isdir(self.folder):
            os.makedirs(self.folder)
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

    def load_cpu_metrics(self):
        """load the metrics related to the cpu usage and energy consumption"""
        if os.path.isfile(self.power_metric_filename):
            metrics = {}
            for line in open(self.power_metric_filename):
                result = json.loads(line)
                date = datetime.datetime.fromisoformat(result['date'])
                if 'cpu' in result['metrics']:
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
                if 'gpu' in result['metrics']:
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
                    mem_use = sum([ v for gpu, mem_uses in per_gpu_mem_use.items() for pid, v in mem_uses.items() ])
                    if 'nvidia_mem_use' not in metrics:
                        metrics['nvidia_mem_use'] = {'dates':[], 'values':[]}
                    metrics['nvidia_mem_use']['dates'].append(date)
                    metrics['nvidia_mem_use']['values'].append(mem_use)
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


    def load_metrics(self):
        """load all metrics. Returns None if the metric is not available"""
        cpu_metrics = self.load_cpu_metrics()
        gpu_metrics = self.load_gpu_metrics()
        exp_metrics = self.load_exp_metrics()
        return cpu_metrics, gpu_metrics, exp_metrics
