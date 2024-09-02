#!/bin/bash
#SBATCH --job-name=cmpNvOw
#SBATCH --output=/mnt/beegfs/home/saesvin/AIPowerMeter/scripts_labia/logs/out.log
#SBATCH --error=/mnt/beegfs/home/saesvin/AIPowerMeter/scripts_labia/logs/error.log
#SBATCH --time=10:00:00    

python /mnt/beegfs/home/saesvin/AIPowerMeter/scripts_labia/compare_raplNvidia_omegawatt.py
    