from deep_learning_power_measure import labia
import argparse

parser = argparse.ArgumentParser("example usage : python scripts_labia/omegafile_head.py --zip_file /mnt/beegfs/home/gay/coca4ai/recordings/omegawatt_subsample/11_16_2023_USB0.zip")
parser.add_argument('--zip_file')
args = parser.parse_args()

labia.head_zip_file(args.zip_file)