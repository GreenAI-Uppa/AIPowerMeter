from flask import Flask
from prometheus_client import Gauge, generate_latest
import threading
import logging

metric_metadata = {'power_draw_cpu':'cpu consumption for the specified process','intel_power':'total rapl power','mem_used_cpu':'total cpu memory used (uss)','mem_used_gpu':'average memory used by the gpu','power_draw_gpu':'Power consumed by gpu as given by nvidia-smi'}

class PrometheusClient():
    """
    this class serve the data so that prometheus can scrape it
      it serves the data with a flask app on the /metrics route
    Experiment will call this class
    """
    def __init__(self, port=None):
        #self.app = Flask(__name__)
        if port == None:
            self.port = 5001 
        else:
            self.port = port
        self.wattemeter_exec = None
        self.gauges = {}
        for metric, description in metric_metadata.items():
            self.gauges[metric] = Gauge(metric, description)
            
        self.app = Flask(__name__)
        @self.app.route('/metrics')
        def metrics():
            r = generate_latest()
            print(r)
            return r
        x = threading.Thread(target=self.run, daemon=True)
        x.start()

    def erase(self):
        """
        not implemented, this would mean to erase the prometheus database, not sure it is relevant in this case.
        """
        pass

    def save_power_metrics(self, metrics):
        """
        updates the self.gauges
        Note that prometheus scrape the data once in the while, and store only the latest metrics

        Prometheus is a pull based monitoring and not a push based one
        A hack with prometheus would be to use a pushgateway
        https://prometheus.io/docs/instrumenting/pushing/
        https://github.com/prometheus/pushgateway/blob/master/README.md
        """
        if 'cpu' in metrics:
            power_draw_cpu = float(metrics['cpu']['total_cpu_power'])
            self.gauges["power_draw_cpu"].set(power_draw_cpu)
            intel_power = float(metrics['cpu']['intel_power'])
            self.gauges["intel_power"].set(intel_power)
            mem_used_cpu = sum(metrics['cpu']['per_process_mem_use_abs'].values())
            self.gauges["mem_used_cpu"].set(mem_used_cpu)
        if 'gpu' in metrics:
            mem_used_gpu = sum([sum(mems.values()) for gpu_id, mems in metrics['gpu']['per_gpu_attributable_mem_use'].items()])
            self.gauges["mem_used_gpu"].set(mem_used_gpu)
            power_draw_gpu = metrics['gpu']['nvidia_draw_absolute']
            self.gauges["power_draw_gpu"].set(power_draw_gpu)

    def save_wattmeter_metrics(self):
        logging.warning('You are trying to log wattmeter metric to prometheus, but this has not been implemented. Skipping.')

    def run(self):
        self.app.run(host = 'localhost', port=self.port)