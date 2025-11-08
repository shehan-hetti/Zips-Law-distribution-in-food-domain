#!/bin/bash
#SBATCH --account=project_2015109
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --job-name=s24_year
#SBATCH --output=/projappl/project_2015109/work/logs/s24_year_%A_%a.out
#SBATCH --array=0-16

module load python-data/3.12
source /projappl/project_2015109/work/venv/bin/activate

YEAR=$((2001 + SLURM_ARRAY_TASK_ID))
echo "[info] SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID -> YEAR=$YEAR"

python /projappl/project_2015109/work/scripts/suomi24_process_year.py ${YEAR} \
  --log-every 20000 -v
