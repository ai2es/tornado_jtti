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
#SBATCH --error=/home/lydiaks2/tornado_project/output/R-%x.%j.err

#source my python env
source /home/lydiaks2/.bashrc
bash 
conda activate tf_gpu

#pip install matplotlib

#run the script
python -u /home/lydiaks2/tornado_project/scripts_ml_model/performance_training_validation_ZH_only.py \
--path_to_training_data="/ourdisk/hpc/ai2es/tornado/learning_patches/tensorflow/3D_light/training_onehot_tor/training_ZH_only.tf/" \
--path_to_validation_data="/ourdisk/hpc/ai2es/tornado/learning_patches/tensorflow/3D_light/validation_onehot_tor/validation1_ZH_only.tf/" \
--path_to_predictions="/ourdisk/hpc/ai2es/tornado/unet/ZH_only/initialrun_model8/" \
--model_checkpoint_path="/ourdisk/hpc/ai2es/tornado/unet/ZH_only/initialrun_model8/initialrun_model8.h5"