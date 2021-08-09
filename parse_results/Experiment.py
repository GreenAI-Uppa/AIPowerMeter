import json, glob, os, numpy as np, datetime

class Experiment():
    def __init__(self, power_folder, acc_metric_file=None):
        self.power_folder = power_folder
        self.acc_metric_file = acc_metric_file
        self.load_metrics()
        self.get_model_card(self)

    def get_model_card(self, folder):
        model_card_file = os.path.join(self.power_folder, 'model_summary.json')
        if os.path.isfile(model_card_file):
            self.model_card = json.load(open(model_card_file))
        else:
            self.model_card = None

    def get_max_acc_and_time(self):
        max_acc = round(max(self.metrics['test_accuracy']['values'] )* 100)/100
        #import pdb; pdb.set_trace()
        num_epochs_to_get_max_acc = min( [ i for (i,v) in enumerate(self.metrics['test_accuracy']['values']) if round(v * 100)/100 == max_acc  ])
        training_time = sum(self.metrics['training_time']['values'][:num_epochs_to_get_max_acc])
        return training_time, max_acc

    def load_metrics(self):
        self.metrics = {}
        if os.path.isdir(self.power_folder):
            for line in open(os.path.join(self.power_folder,'power_metrics.json')):
                result = json.loads(line)
                date = datetime.datetime.fromisoformat(result['date'])
                for k, v in result['metrics'].items():
                    if k == 'date':
                        continue
                    if k == 'per_gpu_average_estimated_utilization_absolute': #
                        continue
                    if k == 'per_gpu_performance_state':
                        continue
                    if k not in self.metrics:
                        self.metrics[k] = {'dates':[], 'values':[]}
                    if isinstance(v, dict):
                        v = sum(v.values())
                    self.metrics[k]['dates'].append(date)
                    self.metrics[k]['values'].append(v)
        #assert len(self.metrics['nvidia_draw_absolute']['values']) == len(self.metrics['nvidia_draw_absolute']['dates'])
        if self.acc_metric_file is not None:
            results = json.load(open(self.acc_metric_file))
            for result in results:
                date = datetime.datetime.fromisoformat(result['end_training_epoch'])
                for k in result:
                    if k == 'end_training_epoch':
                        continue
                    if k not in self.metrics:
                        self.metrics[k] = {'dates':[], 'values':[]}
                    self.metrics[k]['dates'].append(date)
                    self.metrics[k]['values'].append(result[k])

    def set_files_and_folders(self):
        self.simulations = {}
        for power_folder in glob.glob(os.path.join(self.root_folder,'*power')):
            simulation_id = int(power_folder.split('_')[-2])
            self.simulations[simulation_id] = {'power_folder' : power_folder}
        for acc_metric_file in glob.glob(self.root_folder+'/*.json'):
            simulation_id = int(acc_metric_file.replace('.json','').split('_')[-1])
            if simulation_id not in self.simulations:
                self.simulations[simulation_id] = {'acc_metric_file' : acc_metric_files}
            else:
                self.simulations[simulation_id]['acc_metric_file'] = acc_metric_file

    #@staticmethod maybe better to be non static because it can changes in function of the Experiment instances
    def time_to_sec(self, t):
        return t.timestamp()

    def get_curve(self, name, x=None):
        """
        each name is a metric which will have an x
        if x is et to None, I take the x of the first metric, or the intersection?
        """
        assert name in self.metrics
        return [{'date':self.time_to_sec(x), 'value':v} for (x,v) in zip(self.metrics[name]['dates'], self.metrics[name]['values']) ]

    def cumsum(self, metric):
        return np.cumsum([ m['value'] for m in metric ])

    def integrate(self, metric):
        r = [0]
        for i in range(len(metric)-1):
            x1 = metric[i]['date']
            x2 = metric[i+1]['date']
            y1 = metric[i]['value']
            y2 = metric[i+1]['value']
            v = (x2-x1)*(y2+y1)/2
            v += r[-1]
            r.append(v)
        return r

    def wtowh(self, xs):
        return [ x/3600 for x in xs]

    def interpolate(self, metric1, metric2):
        x1 = [m['date'] for m in metric1]
        x2 = [m['date'] for m in metric2]
        x = sorted( x1 + x2)
        y1 = [m['value'] for m in metric1]
        y2 = [m['value'] for m in metric2]
        y1 = np.interp(x, x1, y1)
        y2 = np.interp(x, x2, y2)
        metric1 = [{'date':x, 'value':v} for (x,v) in zip(x, y1) ]
        metric2 = [{'date':x, 'value':v} for (x,v) in zip(x, y2) ]
        return metric1, metric2

    def total_energy_consumed(unit='Wh'):
        # integration
        delta_sec = driver.e.time_to_sec(driver.e.metrics['nvidia_draw_absolute']['dates'][-1]) - driver.e.time_to_sec(driver.e.metrics['nvidia_draw_absolute']['dates'][0])
        self.metrics['nvidia_draw_absolute']
