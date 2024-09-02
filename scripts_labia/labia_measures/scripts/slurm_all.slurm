#!/bin/bash
#SBATCH --job-name=alljobs_tocsv
#SBATCH --output=/mnt/beegfs/projects/coca4ai/metrics/logs/out_summary_all_%j.log
#SBATCH --error=/mnt/beegfs/projects/coca4ai/metrics/logs/error_summary_all_%j.log
#SBATCH --time=10:00:00    

python /mnt/beegfs/projects/coca4ai/metrics/python/append_summary.py --all

python /mnt/beegfs/projects/coca4ai/metrics/python/sort_csv_per_date.py 
