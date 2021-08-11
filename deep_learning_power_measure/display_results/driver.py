from parse_results import Experiment


class Driver():
    def __init__(self):
        self.e = Experiment.Experiment('/home/paul/data/uppa/edouard/grad_1_layer/grad_1_layer_0.001_0_power/','/home/paul/data/uppa/edouard/grad_1_layer/grad_1_layer_0.001_0.json')

    def get_curve(self, name):
        return self.e.get_curve(name)

    def get_models(self):
        return self.e.model_card

    def metrics(self):
        return list(self.e.metrics.keys())

    def interpolate(self, metric1, metric2):
        return self.e.interpolate(metric1, metric2)
