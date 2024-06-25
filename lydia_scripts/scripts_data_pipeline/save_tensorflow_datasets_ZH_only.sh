#!/bin/bash
#SBATCH --partition=ai2es #large_mem
#SBATCH --nodelist=c314
#SBATCH --nodes=1
#SBATCH --ntasks=1 #20
#SBATCH --mem=1G #100G
#SBATCH --time=24:00:00 #24:05:00
#SBATCH --job-name=debug__save_tdata_ZH_only
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
#conda activate tf_gpu
conda activate tf_tornado

# dataset_patches_type should be a list containing the patch types we want to include in training and 50/50 validation sets
# 'n' for nontor patches, 't' for tornadic patches, 's' for sigtor patches
# Ex: only tornadic patches: dataset_patches_type='t', nontor and tor patches: dataset_patches_type='nt' or 'tn'
# Ex: nontor, tor and sigtor patches: dataset_patches_type='stn','snt','tsn','tns','nts', or 'nst'
# dataset_labels_type should be a string, either 'int', for integer labels, or 'onehot', for onehot vector labels
#python -u /home/lydiaks2/tornado_project/scripts_data_pipeline/save_tensorflow_datasets_ZH_only.py \
python -u lydia_scripts/scripts_data_pipeline/save_tensorflow_datasets_ZH_only.py \
--input_xarray_path="/ourdisk/hpc/ai2es/tornado/learning_patches_V2/xarray/3D_light/size_32/forecast_window_5/" \
--output_path="/ourdisk/hpc/ai2es/tornado/learning_patches_V2/tensorflow/3D_light/" \
--training_data_metadata_path='/ourdisk/hpc/ai2es/tornado/learning_patches_V2/tensorflow/3D_light/training_int_nontor_tor/training_metadata_ZH_only.nc' \
--batch_size=256 \
--dataset_patches_type='nt' \
--dataset_labels_type='int' 
#--dry_run
