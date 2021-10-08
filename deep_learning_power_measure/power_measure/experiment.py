"""
this module contains mainly two classes
    Experiment is an entry point to start and end the recording of the power consumption of your Experiment
    ExpResult is used to process and format the recordings.

    Both classes uses a driver attribute to communicate with a database, or read and write in json files
"""
from functools import wraps
import os
import sys
import traceback
from multiprocessing import Process, Queue
from queue import Empty as EmptyQueueException
import time
import numpy as np
import psutil
from . import rapl_power
from . import gpu_power
from . import model_complexity

STOP_MESSAGE = "Stop"

def joules_to_wh(n):
    """conversion function"""
    if hasattr(n, '__iter__'):
        return [i*3600/1000 for i in n]
    return n*3600/1000

def integrate(metric):
    """integral of the metric values over time"""
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

def interpolate(metric1, metric2):
    """
    return two new metrics so that metric1 and metric2 have the same range of dates
    """
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

def humanize_bytes(num, suffix='B'):
    """
    convert a float number to a human readable string to display a number of bytes
    (copied from https://stackoverflow.com/questions/1094841/get-human-readable-version-of-file-size)
    """
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

def time_to_sec(t):
    """convert a date to timestamp in seconds"""
    return t.timestamp()

def cumsum(metric):
    """simple wrapper to cumsum"""
    return np.cumsum([ m['value'] for m in metric ])

def processify(func):
    """Decorator to run a function as a process.
    The created process is joined, so the code does not
    run in parallel.
    """
    def process_func(self, queue, *args, **kwargs):
        try:
            ret = func(self, queue, *args, **kwargs)
        except Exception as e:
            ex_type, ex_value, tb = sys.exc_info()
            error = ex_type, ex_value, "".join(traceback.format_tb(tb))
            ret = None
            queue.put((ret, error))
            raise e
        else:
            error = None
        queue.put((ret, error))

    # register original function with different name
    # in sys.modules so it is pickable
    process_func.__name__ = func.__name__ + "processify_func"
    setattr(sys.modules[__name__], process_func.__name__, process_func)

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        queue = Queue()  # not the same as a Queue.Queue()
        p = Process(target=process_func, args=[self, queue] + list(args), kwargs=kwargs)
        p.start()
        queue.put(p.pid)
        return p, queue
    return wrapper


class Experiment():
    """
    This class provides the method to start an experiment
    by launching a thread which will record the power draw.
    The recording can be ended by sending a stop message
    to this thread
    """
    def __init__(self, driver, model=None, input_size=None, cont=False):
        """
        wrapper class with the methods in charge of recording
        cpu and gpu uses

        driver: interface with a database to save the results
        model and input_size : to compute a model card which will summarize the model
            model is a pytorch model from nn.Module, and input_size is a tuple of int,
            for instance, you can provide  input_size = (64, 3, 32, 32) if you are processing
            3 channels images with a batch size of 64.
        cont : whether or not to erase existing results or append new results.
        """
        self.db_driver = driver
        if not cont:
            driver.erase()
        if model is not None:
            device = 'cpu'
            model.to(device)
            self.save_model_card(model, input_size, device=device)
        self.rapl_available, msg = rapl_power.is_rapl_compatible()
        self.nvidia_available = gpu_power.is_nvidia_compatible()
        if not self.rapl_available and not self.nvidia_available:
            raise Exception(
            "\n\n Neither rapl and nvidia are available, I can't measure anything. Regarding rapl:\n "
            + msg
            )
        if not self.rapl_available:
            print("rapl not available, " + msg)
        else:
            print("CPU power will be measured with rapl")
        if not self.nvidia_available:
            print("nvidia not available, the power of the gpu won't be measured")
        else:
            print("GPU power will be measured with nvidia")

    def save_model_card(self, model, input_size, device='cpu'):
        """
        get a model summary and save it

        model : pytorch model
        input_size : input_size for this model (batch_size, *input_data_size)
        """
        if model is None:
            raise Exception('You tried to compute the model card with the parameter model set to None')

        if model is not None and input_size is None:
            raise Exception('a model was given as parameter, but the input_size argument must also be supplied to estimate the model card')
        summary = model_complexity.get_summary(model, input_size, device=device)
        self.db_driver.save_model_card(summary)

    @processify
    def measure_from_pid_list(self, queue, pid_list, period=1):
        """record power use for the processes given in pid_list"""
        if model is not None:
            self.save_model_card(model)
        self.measure(queue, pid_list, period=period)

    @processify
    def measure_yourself(self, queue, period=1):
        """
        record power use for the process which calls this method
        """
        current_pid = queue.get()
        current_process = psutil.Process(os.getppid())
        pid_list = [current_process.pid] + [
            child.pid for child in current_process.children(recursive=True)
        ]
        pid_list.remove(current_pid)
        self.measure(queue, pid_list, period=period)

    def measure(self, queue, pid_list, period=1):
        """
        performs power use recording

        queue : queue used to communicate to the thread which ask
        for the recording
        pid_list : the recording will be done for these process
        period : waiting time between two RAPL samples.
        """
        print("we'll take the measure of the following pids", pid_list)
        while True:
            time.sleep(period)
            metrics = {}
            if self.rapl_available:
                metrics['cpu'] = rapl_power.get_metrics(pid_list)
            if self.nvidia_available:
                metrics['gpu'] = gpu_power.get_nvidia_gpu_power(pid_list)
            self.db_driver.save_power_metrics(metrics)
            try:
                message = queue.get(block=False)
                # so there are two types of expected messages.
                # The STOP message which is a string, and the metrics dictionnary that this function is sending
                if message == STOP_MESSAGE:
                    print("Done with measuring")
                    return
            except EmptyQueueException:
                pass

class ExpResults():
    """
    Process the power recording from an experiment.
    The actual reading of the recording is done by the driver.
    """
    def __init__(self, db_driver):
        self.db_driver = db_driver
        self.cpu_metrics, self.gpu_metrics, self.exp_metrics = self.db_driver.load_metrics()
        if self.cpu_metrics is None and self.gpu_metrics is None and self.exp_metrics is None:
            raise Exception('I could not load any recordings from folder: "' +
            self.db_driver.folder +
            '".\n Please check that the folder contains valid recordings')
        self.model_card = self.db_driver.get_model_card(self)


    def get_curve(self, name):
        """
        name : key to one of the metric dictionnaries

        return a series of data points [{ "date": date1, "value" : value1}, {"date": date2 ,... }]
        where the dates are in seconds and the value unit is taken from the dictionnaries without transformation.
        """
        if self.cpu_metrics is not None:
            if name in self.cpu_metrics:
                return [{'date':time_to_sec(x), 'value':v} for (x,v) in zip(self.cpu_metrics[name]['dates'], self.cpu_metrics[name]['values']) ]

        if self.gpu_metrics is not None:
            if name in self.gpu_metrics:
                return [{'date':time_to_sec(x), 'value':v} for (x,v) in zip(self.gpu_metrics[name]['dates'], self.gpu_metrics[name]['values']) ]

        if self.exp_metrics is not None:
            if name in self.exp_metrics:
                return [{'date':time_to_sec(x), 'value':v} for (x,v) in zip(self.exp_metrics[name]['dates'], self.exp_metrics[name]['values']) ]
        return None

    def total_(self, metric_name):
        """Total value for this metric. For instance if the metric is in watt and the time in seconds,
        the return value is the energy consumed in Joules"""
        metric = self.get_curve(metric_name)
        return integrate(metric)[-1]

    def duration(self):
        pass

    def average_(self, metric_name):
        """take the average of a metric"""
        metric = self.get_curve(metric_name)
        r = integrate(metric)[-1]
        if len(metric) == 1:
            return r
        return r /( metric[-1]['date'] - metric[0]['date'])

    def print(self):
        """
        simple print of the experiment summary
        """
        print("============================================ EXPERIMENT SUMMARY ============================================")
        if self.model_card is not None:
            print("MODEL SUMMARY: ", self.model_card['total_params'],"parameters and ",self.model_card['total_mult_adds'], "mac operations during the forward pass")
            print()
        if self.cpu_metrics is not None:
            print("ENERGY CONSUMPTION: ")
            print("on the cpu")
            print()
            total_psys_power = self.total_('psys_power')
            total_intel_power = self.total_('intel_power')
            total_dram_power = self.total_('total_dram_power')
            total_cpu_power = self.total_('total_cpu_power')
            rel_dram_power = self.total_('per_process_dram_power')
            rel_cpu_power = self.total_('per_process_cpu_power')
            mem_use_abs = self.average_('mem_use_abs')
            mem_use_uss = self.average_('mem_use_uss')

            print("Total RAM consumption:", total_dram_power, "joules, your experiment consumption: ", rel_dram_power, "joules, for an average of",humanize_bytes(mem_use_abs), 'with an overhead of',humanize_bytes(mem_use_uss))
            print("Total CPU consumption:", total_cpu_power, "joules, your experiment consumption: ", rel_cpu_power, "joules")
            print("total intel power: ", total_intel_power, "joules")
            print("total psys power: ",total_psys_power, "joules")
        if self.gpu_metrics is not None:
            print()
            print()
            print("on the gpu")
            rel_nvidia_power = self.total_('nvidia_draw_absolute')
            abs_nvidia_power = self.total_('nvidia_estimated_attributable_power_draw')
            nvidia_mem_use_abs = self.average_("nvidia_mem_use")
            print("nvidia total consumption:",abs_nvidia_power, "joules, your consumption: ",rel_nvidia_power, ', average memory used:',humanize_bytes(nvidia_mem_use_abs))
