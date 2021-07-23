import json, glob, os, numpy as np, datetime

class Experiment():
    def __init__(self, power_folder, acc_metric_file):
        self.power_folder = power_folder
        self.acc_metric_file = acc_metric_file
        self.load_metrics()
        self.get_model_card(self)

    def get_model_card(self, folder):
        model_card_file = os.path.join(self.power_folder, 'model_summary.json')
        self.model_card = json.load(open(model_card_file))

    def load_metrics(self):
        self.metrics = { 'nvidia_draw_absolute' : {'dates':[], 'values':[]}}
        for line in open(os.path.join(self.power_folder,'power_metrics.json')):
            result = json.loads(line)
            self.metrics['nvidia_draw_absolute']['dates'].append(datetime.datetime.fromisoformat(result['date']))
            self.metrics['nvidia_draw_absolute']['values'].append(result['metrics']['nvidia_draw_absolute'])
        assert len(self.metrics['nvidia_draw_absolute']['values']) == len(self.metrics['nvidia_draw_absolute']['dates'])
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


    def set_metric_names(self):
        self.metric_names = set()
        for i, simulation in self.simulations.items():
            self.metric_names.update(simulation['metrics'].keys())

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
        return [{'date':self.time_to_sec(x), 'value':v} for (x,v) in zip(self.metrics[name]['dates'], self.metrics[name]['values']) ]

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


    def collect_metrics(self, metric_name, mean=True):
        """
        given the names, you collect the values
        to get the time serie for each metric


        you get all the x, so you remove the offsets
        you interpolate over the union of the x
        but the x might not be equivalent
        you take the average of what you can

        first case
        you return one y and one x for each metric

        second case
        you plot one metric against the other
        """
        xs = []
        for i, simulation in self.simulations.items():
            start = simulation['metrics'][metric_name]['dates'][0]
            for date in simulation['metrics'][metric_name]['dates']:
                xs.append(date - start)
        xs.sort()



        metrics = dict( [(metric_name,[ [] for i in range(len(self.simulations)) ]) for metric_name in metric_names ] )
        # getting the x



        for k, measures in metrics.items():
            for i, simulation in enumerate(self.simulations):
                for result in simulation:
                    if k in result:
                        metrics[k][i].append(result[k])

        metrics_power = dict( [(metric_name,[ [] for i in range(len(self.simulations_power)) ]) for metric_name in metric_names ] )
        for k, measures in metrics.items():
            for i, simulation in enumerate(self.simulations_power):
                for result in simulation:
                    if k in result:
                        metrics_power[k][i].append(result[k])

        """
        TO CHECK
        metric_means = {}
        metric_std = {}
        for k, measures in metrics.items():
            metric_means[k] = np.array(measures).mean(axis=0)
            metric_std[k] = np.array(measures).std(axis=0)
        """
        return metrics, metrics_power
