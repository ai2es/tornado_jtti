#!/bin/bash
#SBATCH --partition=ai2es #large_mem
##SBATCH --nodelist=c314
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --mem=200G
#SBATCH --time=12:00:00 #24:05:00
#SBATCH --job-name=2013_2019__create_tfdatasets
#SBATCH --mail-user=monique.shotande@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/momoshog/Tornado/tornado_jtti
#SBATCH --output=/home/momoshog/Tornado/slurm_out/tornado_jtti/%x_%j.out
#SBATCH --error=/home/momoshog/Tornado/slurm_out/tornado_jtti/%x_%j.err
###############################################################

#partition=large_mem, nodes=1, ntasks=20, mem=100G, time=24:05:00
# Source conda env
#source /home/lydiaks2/.bashrc
source ~/.bashrc
bash 
conda activate tf_experiments #tf_tornado #

#. /home/fagg/tf_setup.sh
#conda activate tf


# dataset_patches_type should be a list containing the patch types we want to include in training and 50/50 validation sets
# 'n' for nontor patches, 't' for tornadic patches, 's' for sigtor patches
# Ex: only tornadic patches: dataset_patches_type='t', nontor and tor patches: dataset_patches_type='nt' or 'tn'
# Ex: nontor, tor and sigtor patches: dataset_patches_type='stn','snt','tsn','tns','nts', or 'nst'
# dataset_labels_type should be a string, either 'int', for integer labels, or 'onehot', for onehot vector labels
#python -u /home/lydiaks2/tornado_project/scripts_data_pipeline/save_tensorflow_datasets_ZH_only.py \
python -u lydia_scripts/scripts_data_pipeline/create_tfdatasets.py \
--input_xarray_path="/ourdisk/hpc/ai2es/tornado/learning_patches_V2/xarray/3D_light/size_32/forecast_window_5/" \
--output_path="/ourdisk/hpc/ai2es/tornado/learning_patches_V2/tensorflow/3D_light/" \
--training_data_metadata_path='/ourdisk/hpc/ai2es/tornado/learning_patches_V2/tensorflow/3D_light/training_int_nontor_tor/training_metadata.nc' \
--batch_size=256 \
--dataset_patches_type='nt' \
--dataset_labels_type='int' \
--ZH_only \
-y 2013 2014 2015 2016 2017 2018 2019 
#--dry_run
