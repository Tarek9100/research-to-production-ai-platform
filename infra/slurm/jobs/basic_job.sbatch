#!/bin/bash

#SBATCH --job-name=basic-demo
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=256M
#SBATCH --time=00:02:00
#SBATCH --output=/tmp/slurm-basic-%j.out

echo "===== SLURM JOB ====="
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $(hostname)"
echo "CPUs assigned: $SLURM_CPUS_PER_TASK"
echo "Start:         $(date -Ins)"

sleep 30

echo "End:           $(date -Ins)"
