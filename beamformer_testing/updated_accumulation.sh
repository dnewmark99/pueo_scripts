#!/bin/sh
#SBATCH --cpus-per-task=1        #Cpu-Cores per task
#SBATCH --mem=6G                 #Total Memory
#SBATCH --time=2:00:00          #Max Time Needed (HH:MM:SS)
#SBATCH --partition=avieregg
#SBATCH --account=pi-avieregg
#SBATCH --output=/scratch/midway3/darcynewmark/slurm_scripts/res_%j.txt
#SBATCH --array=0-23

source ~/.bashrc
python /home/darcynewmark/pueoBuilder/components/pueoAnalysisTools/pueoTALON/darcy_scripts/beamformer_testing/updated_accumulation.py --array $SLURM_ARRAY_TASK_ID
