#!/bin/bash
#SBATCH --partition=large_mem
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --mem=100G
#SBATCH --time=24:05:00
#SBATCH --job-name=save_tdata_ZH_only
#SBATCH --mail-user=lydiaks2@illinois.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/lydiaks2/tornado_project/output/R-%x.%j.out
#SBATCH --error=/home/lydiaks2/tornado_project/output/R-%x.%j.err

#partition=large_mem, nodes=1, ntasks=20, mem=100G, time=24:05:00
#source my python env
source /home/lydiaks2/.bashrc
bash 
conda activate tf_gpu

# dataset_patches_type should be a list containing the patch types we want to include in training and 50/50 validation sets
# 'n' for nontor patches, 't' for tornadic patches, 's' for sigtor patches
# Ex: only tornadic patches: dataset_patches_type='t', nontor and tor patches: dataset_patches_type='nt' or 'tn'
# Ex: nontor, tor and sigtor patches: dataset_patches_type='stn','snt','tsn','tns','nts', or 'nst'
# dataset_labels_type should be a string, either 'int', for integer labels, or 'onehot', for onehot vector labels
python -u /home/lydiaks2/tornado_project/scripts_data_pipeline/save_tensorflow_datasets_ZH_only.py \
--input_xarray_path="/ourdisk/hpc/ai2es/tornado/learning_patches/xarray/3D_light/size_32/forecast_window_5/" \
--output_path="/ourdisk/hpc/ai2es/tornado/learning_patches/tensorflow/3D_light/" \
--training_data_metadata_path='/ourdisk/hpc/ai2es/tornado/learning_patches/tensorflow/3D_light/training_onehot_tor/training_metadata.nc' \
--batch_size=256 \
--dataset_patches_type='nt' \
--dataset_labels_type='int'