#!/bin/bash
#SBATCH --partition=ai2es
#SBATCH --nodes=1
#SBATCH -n 1
#SBATCH --ntasks=1
#SBATCH --mem=10G
#SBATCH --time=01:05:00
#SBATCH --job-name=performance
#SBATCH --mail-user=lydiaks2@illinois.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/lydiaks2/tornado_project/output/R-%x.%j.out
#SBATCH --error=/home/lydiaks2/tornado_project/output/R-%x.%j.err\

#source my python env
source /home/lydiaks2/.bashrc
bash 
#conda activate tf_gpu

#pip install matplotlib

#run the script
python -u /home/lydiaks2/tornado_project/scripts_ml_model/ml_performance.py \
--path_to_predictions="/ourdisk/hpc/ai2es/tornado/unet/ZH_only/initialrun_model8/"