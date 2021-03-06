import os, sys, json, re, statistics, pandas as pd, numpy as np, time, datetime

# main_folder = '/data/mfrancois/measure'
# n_iterations = {
#     'input_100': 9000,
#     'input_200': 8000,
#     'input_300': 7000,
#     'input_400': 6000,
#     'input_500': 5000,
# }
# file_to_write = '/data/mfrancois/measure/res.csv'

def main(output='csv', main_folder=None, n_iterations=None, file_to_write=None):
    
    """Merge several power_metrics.json located in several directories.

    Args:
        output (str): Can be "csv", "full" or "cube". Defaults to 'csv'.
        main_folder (str): Path where the previous required architecture is. Defaults to None.
        n_iterations (int): Number of iteration to create one power_metrics.json. If various number of iterations have been used, please use this format: {'folder_name': number_of_iteration} instead of int. Defaults to None.
        file_to_write (str): Output path/file where to write your data. Defaults to None.
    """
    
    if main_folder is None or n_iterations is None or file_to_write is None:
        raise ValueError('Missing argument')
    
    global cube, full_data
    folders = os.listdir(main_folder)
    # fitre sur les noms de dossier : doivent contenir 'input'
    r = re.compile('.*input.*')
    folders = list(filter(r.match, folders))

    full_data = {}# -> données brutes
    cube = {}# -> filtre sur les variables d'intéret + mediane

    for folder in folders:
        # lecture des dossiers d'input size
        full_data[folder] = {} 
        cube[folder] = {} 

        sub_folders = os.listdir(f'{main_folder}/{folder}')
        
        for sub_folder in sub_folders:
            full_data[folder][sub_folder] = {}
            # lecture des dossiers d'itérations
            
            # lecture de chaque json
            with open(f'{main_folder}/{folder}/{sub_folder}/power_metrics.json') as f:
                metrics = [json.loads(line) for line in f]
                full_data[folder][sub_folder]['metrics'] = metrics
            
            if 'latency.json' in os.listdir(f'{main_folder}/{folder}/{sub_folder}'):
                with open(f'{main_folder}/{folder}/{sub_folder}/latency.json') as f:
                    full_data[folder][sub_folder]['latency'] = json.load(f)
            else: 
                full_data[folder][sub_folder]['latency'] = np.loadtxt(f'{main_folder}/{folder}/{sub_folder}/latency.csv').tolist()
            
            n = n_iterations[folder] if type(n_iterations) is dict else n_iterations

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
            cube[folder][sub_folder] = {
                'intel_power': integrate(date=date, watt=intel_power_watt)/n,
                'total_cpu_power': integrate(date=date, watt=total_cpu_power)/n,
                'mem_use_abs': calc_median(power_metrics=metrics, metrics=['metrics', 'cpu', 'per_process_mem_use_abs', 'pid'])/n,
                'nvidia_draw_absolute': integrate(date=date, watt=nvidia_draw_absolute)/n,
                'nvidia_estimated_attributable_power_draw': integrate(date=date, watt=nvidia_estimated_attributable_power_draw)/n,
                'per_gpu_attributable_mem_use': calc_median(power_metrics=metrics, metrics=['metrics', 'gpu', 'per_gpu_attributable_mem_use', '0', 'pid'])/n,
                'sm': calc_median(power_metrics=metrics, metrics=['metrics', 'gpu', 'per_gpu_average_estimated_utilization_absolute', 'sm'])/n,
                'latency': statistics.median(full_data[folder][sub_folder]['latency']),
            }

    write_data(path=file_to_write, output=output)

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

def write_data(path, output):
    """write data

    Args:
        path (str): output path
        output (str): output format
    """
    if output == 'csv':
        get_csv(path=path)
    elif output == 'full':
        get_raw(path=path)
    else: 
        get_cube(path=path)

def get_cube(path):
    """write cube data

    Args:
        path (str): path to write data to.
    """
    with open(path, 'w') as f:
        json.dump(cube, f)

def to_pandas():
    """Transform cube to pandas dataframe

    Returns:
        pandas.DataFrame: dataframe with rows as input size and columne as measure consumption.
    """
    keys = [
        'intel_power',
        'total_cpu_power',
        'mem_use_abs',
        'nvidia_draw_absolute',
        'nvidia_estimated_attributable_power_draw',
        'per_gpu_attributable_mem_use',
        'sm',
        'latency'
    ]
    df = pd.DataFrame.from_dict(
        {k: [statistics.median([v.get(k) for _, v in cube.get(folder).items()]) for folder in cube.keys()] for k in keys}
    )
    ind = [folder.split('_')[1] for folder in cube.keys()]
    df.index = ind 
    
    print('-'*10)
    print('Dataframe transformed')
    print('-'*10)
    print(f'df shape: {df.shape}')
    print('-'*10)
    print(df)
    return df

def get_csv(path):
    """write csv data

    Args:
        path (str): path to write data to.
    """
    df = to_pandas()
    df.to_csv(path)
   

def get_raw(path):
    """write raw data

    Args:
        path (str): path to write data to.
    """
    with open(path, 'w') as f:
        json.dump(full_data, f)


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
                power_metrics = sum(list(power_metrics.values()))
            except IndexError:
                print("no metrics on GPU found")
                power_metrics = 0
        elif metric == 'sm':
            power_metrics = sum(s.get('sm') for s in power_metrics)
        else: 
            power_metrics = power_metrics.get(metric)

    return power_metrics



if __name__ == "__main__":  
    main(sys.argv[1])


