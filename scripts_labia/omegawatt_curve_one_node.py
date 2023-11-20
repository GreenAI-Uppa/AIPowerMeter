from deep_learning_power_measure import labia
import datetime 
import argparse
from tqdm import tqdm
import statistics

parser = argparse.ArgumentParser("example usage : python scripts_labia/omegafile_head.py --zip_file /mnt/beegfs/home/gay/coca4ai/recordings/omegawatt_subsample/11_16_2023_USB0.zip")
parser.add_argument('--node_name', default='n102')
parser.add_argument('--start',help='start date in isoformat',default='2023-11-16')
parser.add_argument('--end',help='end date in isoformat',default='2023-11-17')
args = parser.parse_args()

node_name = args.node_name
t1 = datetime.datetime.fromisoformat(args.start).timestamp()
t2 = datetime.datetime.fromisoformat(args.end).timestamp()

fake_job_id = '0'
zip_files = labia.get_zip_files([(t1, t2, node_name, fake_job_id)])
all_curves = {fake_job_id : {node_name : []}}
for zip_file, segments in tqdm(zip_files.items()):
  curves = labia.read_zip_file(zip_file, segments)
  all_curves[fake_job_id][node_name] += curves[fake_job_id][node_name]

values = [v['value'] for v in all_curves[fake_job_id][node_name]]
print('Read ',len(values), 'values recorded by omegawatt')
print('Median :', statistics.median(values))
print('Max :', max(values))
print('Min :', min(values))