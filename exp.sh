#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH -t 110:00:00
#SBATCH --mem=0
#SBATCH --job-name=tpot-ensemble
#SBATCH -p defq
#SBATCH --exclusive
#SBATCH --exclude=esplhpc-cp040
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=Ethan.Hodess@cshs.org
#SBATCH -o ./logs/outputs/output.%j_%a.out # STDOUT
#SBATCH --array=0-74
RUN=${SLURM_ARRAY_TASK_ID:-1}
echo “Run: ${RUN}”
module load git/2.33.1

source /home/hodesse/miniconda3/etc/profile.d/conda.sh
#conda create --name tpot2env -c conda-forge python=3.10
conda activate tpot2env
#pip install -r requirements.txt

echo RunStart
srun -u /home/hodesse/miniconda3/envs/tpot2env/bin/python xgb_models_tests.py \
--n_jobs 48 \
--savepath logs \
--num_runs ${RUN} \