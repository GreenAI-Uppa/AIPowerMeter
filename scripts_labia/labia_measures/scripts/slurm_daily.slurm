#!/bin/bash
#SBATCH --job-name=dailyjobs_tocsv
#SBATCH --output=/mnt/beegfs/projects/coca4ai/metrics/logs/out_summary_daily_%j.log
#SBATCH --error=/mnt/beegfs/projects/coca4ai/metrics/logs/error_summary_daily_%j.log
#SBATCH --time=10:00:00    

python /mnt/beegfs/projects/coca4ai/metrics/python/append_summary.py --daily

python /mnt/beegfs/projects/coca4ai/metrics/python/sort_csv_per_date.py 
