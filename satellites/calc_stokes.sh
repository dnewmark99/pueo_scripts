#!/bin/sh
#SBATCH --cpus-per-task=1        #Cpu-Cores per task
#SBATCH --mem=10G
#SBATCH --time=7:00:00          #Max Time Needed (HH:MM:SS)
#SBATCH --partition=avieregg
#SBATCH --account=pi-avieregg
#SBATCH --output=/scratch/midway3/darcynewmark/slurm_scripts/res_%j.txt
#SBATCH --array=0-800

source ~/.bashrc
python /home/darcynewmark/pueoBuilder/components/pueoAnalysisTools/pueoTALON/darcy_scripts/satellites/calc_stokes.py --array $SLURM_ARRAY_TASK_ID
