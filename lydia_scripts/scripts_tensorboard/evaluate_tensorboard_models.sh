#!/bin/bash
#SBATCH --partition=ai2es
#SBATCH --nodes=1
#SBATCH -n 1
#SBATCH --ntasks=1
#SBATCH --mem=10G
#SBATCH --time=01:05:00
#SBATCH --job-name=tb_evaluate
#SBATCH --mail-user=lydiaks2@illinois.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/lydiaks2/tornado_project/output/R-%x.%j.out
#SBATCH --error=/home/lydiaks2/tornado_project/output/R-%x.%j.err

#source my python env
source /home/lydiaks2/.bashrc
bash 
conda activate tf_gpu

#pip install matplotlib

#run the script
python -u /home/lydiaks2/tornado_project/scripts_tensorboard/evaluate_tensorboard_models.py \
--path_to_models='/scratch/lydiaks2/models/jtti1/' \
--path_to_tf_datasets='/ourdisk/hpc/ai2es/tornado/learning_patches/tensorflow/3D_light/' \
--path_to_output='/scratch/lydiaks2/compare_tb_models/jtti1/' \
--dataset_patches_type='nt' \
--dataset_labels_type='int'