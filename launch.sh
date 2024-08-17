#!/bin/bash
#
#SBATCH --partition=gpu_min24gb       # Reserved partition
#SBATCH --qos=gpu_min24gb_ext         # QoS level
#SBATCH --job-name=clip               # Job name
#SBATCH -o slurm_%x.%j.out           # File containing STDOUT output
#SBATCH -e slurm_%x.%j.err           # File containing STDERR output

echo "Running job in reserved partition"

# Executa o script Python
python3 train.py
echo "train completed successfully"


echo "Job completed successfully"
