#!/bin/bash

#SBATCH --partition=ai2es #normal
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --mem=10G
#SBATCH --time=5:00:00
#SBATCH --job-name=run__patching
#SBATCH --mail-user=monique.shotande@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/momoshog/Tornado/tornado_jtti
#SBATCH --output=/home/momoshog/Tornado/slurm_out/tornado_jtti/%x_%j.out
#SBATCH --error=/home/momoshog/Tornado/slurm_out/tornado_jtti/%x_%j.err
#SBATCH --array=427-446%20
##SBATCH --array=1-214,262%20

###############################################################

# Source conda env
#source /home/lydiaks2/.bashrc
source ~/.bashrc
bash 

conda activate gewitter 
#this_spc_date_string='20130520'

echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
python --version

#python -u /home/lydiaks2/tornado_project/scripts_data_pipeline/patching.py \
python -u lydia_scripts/scripts_data_pipeline/patching.py \
--this_spc_date_string_array_val=$SLURM_ARRAY_TASK_ID \
--input_radar_dir_name="/ourdisk/hpc/ai2es/tornado/gridrad_gridded_V2/" \
--input_storm_mask_dir_name="/ourdisk/hpc/ai2es/tornado/storm_mask_unet_V2/" \
--output_patch_dir_name="/ourdisk/hpc/ai2es/tornado/learning_patches_V2/xarray/3D/" \
--patch_size=32 \
--n_patches=150
#--dry_run