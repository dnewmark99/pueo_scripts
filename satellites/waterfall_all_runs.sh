#!/bin/sh
#SBATCH --cpus-per-task=1        #Cpu-Cores per task
#SBATCH --mem=10G
#SBATCH --time=12:00:00          #Max Time Needed (HH:MM:SS)
#SBATCH --account=pi-avieregg
#SBATCH --partition=avieregg
#SBATCH --output=/scratch/midway3/darcynewmark/slurm_scripts/res_%j.txt
#SBATCH --array=0-800

source ~/.bashrc
python /home/darcynewmark/pueo_scripts/satellites/waterfall_all_runs.py --array $SLURM_ARRAY_TASK_ID
