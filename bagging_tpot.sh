#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH -t 1:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=tpot-ensemble
#SBATCH -p moore, defq
#SBATCH --exclude=esplhpc-cp040
#SBATCH -o ./logs/outputs/output.%j_%a.out # STDOUT
#SBATCH --array=0-545

RUN=${SLURM_ARRAY_TASK_ID:-1}
echo “Run: ${RUN}”
module load git/2.33.1

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ethan

echo RunStart
srun -u python bagging_pipeline_hpc.py \
--n_jobs 12 \
--savepath logs_bagging \
--num_runs ${RUN} \
--data_dir /common/hodesse/hpc_test/TPOTElites/openml_271