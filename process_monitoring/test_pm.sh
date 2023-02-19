#!/bin/bash
#SBATCH --partition=normal
#SBATCH --nodes=1
##SBATCH -n 20
##SBATCH --ntasks=20
#SBATCH --mem=10G
#SBATCH --time=00:10:00
#SBATCH --job-name=process_monitor_test
#SBATCH --mail-user=monique.shotande@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/momoshog/Tornado/tornado_jtti/process_monitoring
#SBATCH --output=/home/momoshog/Tornado/slurm_out/tornado_jtti/pm__%x_%j.out
#SBATCH --error=/home/momoshog/Tornado/slurm_out/tornado_jtti/pm__%x_%j.err
#SBATCH --array=0-1

##########################################################

# Source Lydia's python env
source /home/lydiaks2/.bashrc
bash 
conda activate tf_gpu

python --version
echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"

#run the script
python process_monitor.py

