import os, sys, json, re, statistics, pandas as pd, numpy as np, time, datetime
"""
Lecture simple d'un power_metrics.json avec possibilité de pondérer chaque metrique par une valeur. 
Traiter la latency séparemment si besoin (sinon trop de cas par cas)
"""



def read_power_metrics(path: str, n=1):
    """summarize a power_metrics.json file

    Args:
        path (str): power_metrics file path
        n (int, optional): N divider. Defaults to 1.

    Returns:
        dict: aggregated and usefull power_metrics
    """

    with open(path, 'r') as f:
        metrics = [json.loads(line) for line in f]


    #convert en joules 
    # intel_power
    # total_cpu_power
    # nvidia_estimated_attributable_power_draw
    intel_power_watt = [get_value(power_metrics=metric, metrics=['metrics', 'cpu', 'intel_power']) for metric in metrics]
    total_cpu_power = [get_value(power_metrics=metric, metrics=['metrics', 'cpu', 'total_cpu_power']) for metric in metrics]
    nvidia_draw_absolute = [get_value(power_metrics=metric, metrics=['metrics', 'gpu', 'nvidia_draw_absolute']) for metric in metrics]
    nvidia_estimated_attributable_power_draw = [get_value(power_metrics=metric, metrics=['metrics', 'gpu', 'nvidia_estimated_attributable_power_draw']) for metric in metrics]
    date = [get_value(power_metrics=metric, metrics=['date']) for metric in metrics]

    # concatenation des lists par la médiane
    # cube processing
    return {
        'intel_power': integrate(date=date, watt=intel_power_watt)/n,
        'total_cpu_power': integrate(date=date, watt=total_cpu_power)/n,
        'mem_use_abs': calc_median(power_metrics=metrics, metrics=['metrics', 'cpu', 'per_process_mem_use_abs', 'pid'])/n,
        'nvidia_draw_absolute': integrate(date=date, watt=nvidia_draw_absolute)/n,
        'nvidia_estimated_attributable_power_draw': integrate(date=date, watt=nvidia_estimated_attributable_power_draw)/n,
        'per_gpu_attributable_mem_use': calc_median(power_metrics=metrics, metrics=['metrics', 'gpu', 'per_gpu_attributable_mem_use', '0', 'pid'])/n,
        'sm': calc_median(power_metrics=metrics, metrics=['metrics', 'gpu', 'per_gpu_average_estimated_utilization_absolute', 'sm'])/n,
    }
        
def integrate(date, watt):
    """integrate x: date, y: watt

    Args:
        date (list): list of date
        watt (list): list of watt

    Raises:
        ValueError: [description]

    Returns:
        float: estimated joule metric
    """
    v = []
    if len(date) != len(watt):
        raise ValueError('not the same length')
    for i in range(len(watt)-1):
        x1 = datetime.datetime.fromisoformat(date[i]).timestamp()
        x2 = datetime.datetime.fromisoformat(date[i+1]).timestamp()
        y1 = watt[i]
        y2 = watt[i+1]
        v.append((x2-x1)*(y2+y1)/2)
    return statistics.median(v)*len(v)

def calc_median(power_metrics, metrics):
    """calc power consumption by iterate, based on median

    Args:
        power_metrics (list): [description]
        metrics (list): [description]

    Returns:
        float: consumption estimation
    """
    
    values = [get_value(
        power_metrics=power_metric,
        metrics=metrics
    ) for power_metric in power_metrics]
    
    n = len(values)
    med = statistics.median(values)
    return n * med


def get_value(power_metrics=None, metrics=None, debug=False):
    """travel across dictionary

    Args:
        power_metrics (dict): dictionnary. Defaults to None.
        metrics (list): keys way to get the target value. Defaults to None.
        debug (bool): print running step. Defaults to False.
        
    Returns:
        float: return the target value 
    """
    for metric in metrics:
        if debug:
            print(f'running on {metric}')
        if metric == 'pid':
            # if not exist then return error -> return 0
            try:
                power_metrics = list(power_metrics.values())[0]
            except IndexError:
                print("no metrics on GPU found")
                power_metrics = 0
        elif metric == 'sm':
            power_metrics = sum([s.get('sm') for s in power_metrics])
        elif metric in power_metrics.keys(): 
            power_metrics = power_metrics.get(metric)
        else:
            return 0

    return power_metrics
