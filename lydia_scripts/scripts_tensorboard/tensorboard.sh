#!/bin/bash
#SBATCH --partition=ai2es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --job-name=board
#SBATCH --mail-user=lydiaks2@illinois.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/lydiaks2/tornado_project/output/R-%x.%j.out
#SBATCH --error=/home/lydiaks2/tornado_project/output/R-%x.%j.err

#source my python env
source /home/lydiaks2/.bashrc

bash 

conda activate tf_gpu
tensorboard \
--logdir="/scratch/lydiaks2/tensorboardlogs/jtti/logs/" \
--port=6006 \
--bind_all