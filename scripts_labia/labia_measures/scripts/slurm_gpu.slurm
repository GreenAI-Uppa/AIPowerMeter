#!/bin/bash
#SBATCH --job-name=GPU
#SBATCH --output=/mnt/beegfs/projects/coca4ai/metrics/logs/gpu_out_%j.log
#SBATCH --error=/mnt/beegfs/projects/coca4ai/metrics/logs/gpu_error_%j.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --partition=all    

echo "Checking GPU availability..."
nvidia-smi

python /mnt/beegfs/projects/coca4ai/metrics/python/eat_gpu.py