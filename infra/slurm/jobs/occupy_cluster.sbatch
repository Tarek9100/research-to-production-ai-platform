#!/bin/bash

#SBATCH --job-name=occupy-cluster
#SBATCH --partition=cpu
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH --mem=500M
#SBATCH --time=00:03:00
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/occupy-%j.out

echo "Job $SLURM_JOB_ID started"
echo "Allocated nodes: $SLURM_JOB_NODELIST"

srun bash -c '
  echo "$(hostname): starting"
  sleep 90
  echo "$(hostname): finished"
'
