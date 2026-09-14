#!/bin/bash

#SBATCH --job-name=waiting-demo
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=256M
#SBATCH --time=00:02:00
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/waiting-%j.out

hostname
sleep 10
