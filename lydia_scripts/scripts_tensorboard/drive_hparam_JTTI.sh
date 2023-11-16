#!/bin/bash
#SBATCH --partition=ai2es_a100
#SBATCH -w c732
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=30G
#SBATCH --time=48:00:00
#SBATCH --job-name=hparam1_JTTI
#SBATCH --mail-user=lydiaks2@illinois.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/lydiaks2/tornado_project/output/R-%x.%j.out
#SBATCH --error=/home/lydiaks2/tornado_project/output/R-%x.%j.err

#source my python env
source /home/lydiaks2/.bashrc
bash 

conda activate tf_gpu

python -u /home/lydiaks2/tornado_project/scripts_tensorboard/hparams_JTTI.py \
--logdir="/scratch/lydiaks2/tensorboardlogs/jtti/logs/"
