import os, sys
import traceback
from tqdm import tqdm
import datetime
import argparse
import json
from deep_learning_power_measure.power_measure import experiment, parsers
from deep_learning_power_measure import labia

parser = argparse.ArgumentParser("Example usage with default options : python scripts_labia/compare_raplNvidia_omegawatt.py")
parser.add_argument('--node_names', default='n4,n102,n54,n55,n5')
parser.add_argument('--start',help='start date in isoformat',default='2023-10-23')
parser.add_argument('--end',help='end date in isoformat',default='2023-11-19')
parser.add_argument('--meas_delta',help='time delta on which to average a point recording value in secs.', type=int, default=10)
parser.add_argument('--period',help='interval between two point recording, in secs', type=int, default=3600)
parser.add_argument('--output_file',help='json file where to save the results',default='results_omegawatt_raplnvidia.json')
args = parser.parse_args()

nodes = args.node_names.split(',') 
print('Extracting recordings for nodes : ', nodes )
start = datetime.datetime.fromisoformat(args.start).timestamp()
end = datetime.datetime.fromisoformat(args.end).timestamp()
output_file = args.output_file


meas_delta = args.meas_delta #10 # in seconds, average the energy over this value
period = args.period #3600 # in seconds, take a measurement every this period

per_node_measure = labia.all_feature_per_t(nodes, start=start, end=end)
json.dump(per_node_measure, open(output_file,'w'))
