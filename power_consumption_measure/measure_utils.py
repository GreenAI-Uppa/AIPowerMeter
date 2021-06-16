import json
#from thop import profile
import datetime
from functools import wraps
import psutil
import os
import sys
import traceback
from multiprocessing import Process, Queue
from queue import Empty as EmptyQueueException
import time
from . import rapl_power
from . import gpu_power

STOP_MESSAGE = "Stop"
STOPPED_MESSAGE = "Stopped"

def propagate_scores(process_tree, stat_name, leaves=None):
    if leaves is None:
        leaves = [n for n in process_tree.nodes if not list(process_tree.neighbors(n)) ]
    # for each leave
    for pid in leaves:
        # propagate the score up to the root
        stat = process_tree.nodes[pid][stat_name]
        parent = list(process_tree.predecessors(pid))
        while parent:
            parent=parent[0]
            if stat_name in process_tree.nodes[parent]:
                process_tree.nodes[parent][stat_name] += stat
            else:
                process_tree.nodes[parent][stat_name] = stat
            parent = list(process_tree.predecessors(parent))


def processify(func):
    """Decorator to run a function as a process.
    Be sure that every argument and the return value
    is *pickable*.
    The created process is joined, so the code does not
    run in parallel.
    """

    def process_func(q, *args, **kwargs):
        try:
            ret = func(q, *args, **kwargs)
        except Exception as e:
            ex_type, ex_value, tb = sys.exc_info()
            error = ex_type, ex_value, "".join(traceback.format_tb(tb))
            ret = None
            q.put((ret, error))
            raise e
        else:
            error = None
        q.put((ret, error))

    # register original function with different name
    # in sys.modules so it is pickable
    process_func.__name__ = func.__name__ + "processify_func"
    setattr(sys.modules[__name__], process_func.__name__, process_func)

    @wraps(func)
    def wrapper(*args, **kwargs):
        queue = Queue()  # not the same as a Queue.Queue()
        p = Process(target=process_func, args=[queue] + list(args), kwargs=kwargs)
        p.start()
        return p, queue

    return wrapper


@processify
def measure_from_pid_list(queue, pids, outdir=None, period=1):
    rapl_available, nvidia_available, outfile = init(outdir)
    measure(queue, pid_list, rapl_available, nvidia_available, outfile=outfile, period=period)

@processify
def measure_yourself(queue, outdir=None, period=1):
    """
    # {'cpu_uses': cpu_uses, 'mem_uses': mem_uses, 'intel_power' :intel_power, 'total_cpu_power':cpu_power, 'total_dram_power':dram_power, 'uncore_power':uncore_power, 'per_process_cpu_power':cpu_power_use, 'per_process_dram_power':dram_power_use, 'psys_power':psys_power} 
    """
    rapl_available, nvidia_available, outfile = init(outdir)
    current_process = psutil.Process(os.getppid())
    pid_list = [current_process.pid] + [
        child.pid for child in current_process.children(recursive=True)
    ]
    measure(queue, pid_list, rapl_available, nvidia_available, outfile=outfile, period=period)

def measure(queue, pid_list, rapl_available, nvidia_available, outfile=None, period=1):
    print("we'll take the measure of the following pids", pid_list)
    while True:
        time.sleep(period)
        metrics = {}
        if rapl_available:
            metrics = rapl_power.get_metrics(pid_list)
        if nvidia_available:
            metrics_gpu = gpu_power.get_nvidia_gpu_power(pid_list)
            metrics = {**metrics, **metrics_gpu}
        if outfile:
            today_str = datetime.datetime.now().__str__()
            data = { 'date': today_str, 'metrics': metrics }
            json_str = json.dumps(data)
            outfile.write(json_str+'\n')
        else:
            queue.put(metrics)
        try:
            message = queue.get(block=False)
            # so there are two types of expected messages. 
            # The STOP message which is a string, and the metrics dictionnary that this function is sending
            print('receiving message',message)
            if isinstance(message, str):
                print('receving stop')
                if message == STOP_MESSAGE:
                    queue.put(STOPPED_MESSAGE)
                    print("sending stopped")
                    return
            else:# Put back the message, it is a metric and the parent process should read it 
                queue.put(message)
        except EmptyQueueException:
            pass

def init(outdir=None):
    rapl, msg = rapl_power.is_rapl_compatible()
    nvidia = gpu_power.is_nvidia_compatible()
    if not rapl and not nvidia:
        raise Exception("\n\n Neither rapl and nvidia are available, I can't measure anything. Regarding rapl:\n "+ msg)
    if not rapl:
        print("rapl not available" + msg)
    else:
        print("CPU power will be measured with rapl")
    if not nvidia:
        print("nvidia not available, the power of the gpu won't be measured")
    else:
        print("GPU power will be measured with nvidia")
    if outdir:
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        outfile = open(outdir + '/power_metrics.json','w')
        return rapl, nvidia, outfile
    return rapl, nvidia, None

import networkx as nx
import psutil
def get_pids():
    """
    return a tree where each node corresponds to a running process
    the parent-child in the tree follows the parent child process relations, ie a child process had been launched by its father
    """
    process_tree = nx.DiGraph()
    for proc in psutil.process_iter():
        try:
            process_tree.add_node(proc.pid)
            process_tree.nodes[proc.pid]['name'] = proc.name()
            process_tree.nodes[proc.pid]['user'] = proc.username()
            for child in proc.children(recursive=False):
                process_tree.add_edge(proc.pid, child.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return process_tree
