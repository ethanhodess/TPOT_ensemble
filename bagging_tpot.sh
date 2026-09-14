#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH -t 3:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=tpot-ensemble
#SBATCH -p defq,moore,preemptable
#SBATCH --exclude=esplhpc-cp040
#SBATCH -o ./logs_missing/outputs/output.%j_%a.out # STDOUT
#SBATCH --array=6,63,65,68,323,326,328,329,333,480,521,523,526,527,528,530,531,534,539,540,544

RUN=${SLURM_ARRAY_TASK_ID:-1}
echo “Run: ${RUN}”

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ethan

echo RunStart
srun -u python bagging_pipeline_hpc.py \
--n_jobs 12 \
--savepath logs_bagging_missing \
--num_runs ${RUN} \
--data_dir /home/hernandezj45/Repos/TPOT_ensemble/Raw_OpenML_Suite_271_Classification