#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH -t 12:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=tpot_constrained
#SBATCH -p defq,moore,preemptable
#SBATCH -o ./logs_pipeline_constrained/outputs/output.%j_%a.out # STDOUT
#SBATCH --array=0-545

RUN=${SLURM_ARRAY_TASK_ID:-1}
echo “Run: ${RUN}”

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ethan

echo RunStart
srun -u python pipeline_spaces_hpc.py \
--n_jobs 12 \
--savepath tpot_constrained \
--num_runs ${RUN} \
--data_dir /home/hernandezj45/Repos/TPOT_ensemble/Raw_OpenML_Suite_271_Classification \
--gens 20 \
--pop_size 25 \
--pipeline_space "constrained"