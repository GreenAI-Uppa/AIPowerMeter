import json, glob, os, numpy as np, datetime

class JsonParser():
    def __init__(self, folder):
        self.folder = folder
        self.power_metric_filename = self.folder + '/power_metrics.json'
        self.exp_metric_filename = self.folder + '/results_exp.json'
        if not os.path.isdir(self.folder):
            os.makedirs(self.folder)
        self.model_card_file = os.path.join(self.folder,'model_summary.json')

    def save_model_card(self, model_card):
        print("saving model card")
        json.dump(model_card, open(self.model_card_file, 'w'))

    def save_power_metrics(self, metrics):
        power_metric_fileout = open(self.power_metric_filename,'a')
        today_str = datetime.datetime.now().__str__()
        data = { 'date': today_str, 'metrics': metrics }
        json_str = json.dumps(data)
        power_metric_fileout.write(json_str+'\n')

    def save_exp_metrics(self, metrics):
        exp_metric_fileout = open(self.exp_metric_filename,'a')
        today_str = datetime.datetime.now().__str__()
        data = { 'date': today_str, 'metrics': metrics }
        json_str = json.dumps(data)
        exp_metric_fileout.write(json_str+'\n')

    def get_model_card(self, folder):
        assert os.path.isfile(self.model_card_file)
        return json.load(open(self.model_card_file))

    def load_metrics(self):
        metrics = {}
        if os.path.isfile(self.power_metric_filename):
            for line in open(self.power_metric_filename):
                result = json.loads(line)
                date = datetime.datetime.fromisoformat(result['date'])
                for k, v in result['metrics'].items():
                    if k == 'date':
                        continue
                    if k == 'per_gpu_average_estimated_utilization_absolute': #
                        continue
                    if k == 'per_gpu_performance_state':
                        continue
                    if k not in metrics:
                        metrics[k] = {'dates':[], 'values':[]}
                    if isinstance(v, dict):
                        v = sum(v.values())
                    metrics[k]['dates'].append(date)
                    metrics[k]['values'].append(v)
        #assert len(self.metrics['nvidia_draw_absolute']['values']) == len(self.metrics['nvidia_draw_absolute']['dates'])
        if os.path.isfile(self.exp_metric_filename):
            results = json.load(open(self.exp_metric_filename))
            for result in results:
                date = datetime.datetime.fromisoformat(result['end_training_epoch'])
                for k in result:
                    if k == 'end_training_epoch':
                        continue
                    if k not in metrics:
                        self.metrics[k] = {'dates':[], 'values':[]}
                    metrics[k]['dates'].append(date)
                    metrics[k]['values'].append(result[k])
        return metrics
