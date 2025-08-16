#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -t 00:10:00
#SBATCH --mem=10GB
#SBATCH --job-name=collect
#SBATCH -p defq
#SBATCH --exclude=esplhpc-cp040


source /home/hodesse/miniconda3/etc/profile.d/conda.sh
conda activate tpot2env

python /common/hodesse/hpc_test/TPOT2_ensemble/collect/collect_csvs.py