import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import datetime
import json
import argparse

parser = argparse.ArgumentParser("Example usage with default options : python scripts_labia/compare_raplNvidia_omegawatt.py")
parser.add_argument('--json_file', default='results_omegawatt_raplnvidia.json', help="json file containing the recordings. See script compare_raplNvidia_omegawatt.py for the file generation")
parser.add_argument('--save', default=False, action='store_true', help="set to True to save plots in png files")
args = parser.parse_args()

json_file = args.json_file
per_node_measure = json.load(open(json_file))

## dealing with outlier values and empty values
for node_name, r in per_node_measure.items():
    remove = []
    for m, v in r.items():
        if 'rapl' not in v:
            v['rapl'] = 0
            v['nvidia'] = 0
        if v['rapl'] > 10000:
            remove.append(m)
        if 'omegawatt_power_draw' not in v:
            print(m, node_name)
    for re in remove:
        del r[re]

    dates, rapl, nvidia, omega = zip(*[(m, v['rapl'], v['nvidia'], v['omegawatt_power_draw']) for (m, v) in r.items()])
    dates = [datetime.datetime.fromtimestamp(float(d)) for d in dates]

    # filter dates from 2024-07-01 to 2024-08-01

    # print("dates", dates)
    # dates = [d for d in dates if d > datetime.datetime(2023, 7, 1) and d < datetime.datetime(2023, 8, 1)]
    print("1st date", dates[0])
    print("last date", dates[-1])
    # rapl = rapl[len(rapl) - len(dates):]
    # nvidia = nvidia[len(nvidia) - len(dates):]
    # omega = omega[len(omega) - len(dates):]

    fig, ax = plt.subplots(constrained_layout=True)
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    total_nvidia_rapl = np.array(rapl) + np.array(nvidia)

    ax.plot(dates, total_nvidia_rapl, label='nvidia + rapl', color='red')  # Adjusted size
    ax.plot(dates, np.array(omega), label='omega', color='blue')  # Adjusted size
    ax.set_title('Node ' + node_name + ' Nvidia + RAPL and OmegaWatt consumptions')
    plt.ylabel('Watts')
    plt.legend()
    if args.save:
        plt.savefig(node_name + '.png')
    else:
        plt.show()
