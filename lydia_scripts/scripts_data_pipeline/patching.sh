#!/bin/bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH -n 20
#SBATCH --ntasks=20
#SBATCH --mem=10G
#SBATCH --time=24:00:00
#SBATCH --job-name=patching
#SBATCH --mail-user=lydiaks2@illinois.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/lydiaks2/tornado_project/output/R-%x.%j.out
#SBATCH --error=/home/lydiaks2/tornado_project/output/R-%x.%j.err
#SBATCH --array=1-262%20

###Total days in 2013 = 119 ### 262 total files

#source my python env
source /home/lydiaks2/.bashrc
bash 

conda activate gewitter 
#this_spc_date_string='20130520'

python -u /home/lydiaks2/tornado_project/scripts_data_pipeline/patching.py \
--this_spc_date_string_array_val=$SLURM_ARRAY_TASK_ID \
--input_radar_dir_name="/ourdisk/hpc/ai2es/tornado/gridrad_gridded/" \
--input_storm_mask_dir_name="/ourdisk/hpc/ai2es/tornado/storm_mask_unet/" \
--output_patch_dir_name="/ourdisk/hpc/ai2es/tornado/learning_patches/xarray/3D/" \
--patch_size=32 \
--n_patches=100