#!/bin/bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=20 ##-n 20
#SBATCH --mem=10G
#SBATCH --time=1:30:00
#SBATCH --job-name=test__save_light_data
#SBATCH --mail-user=monique.shotande@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/momoshog/Tornado/tornado_jtti
#SBATCH --output=/home/momoshog/Tornado/slurm_out/tornado_jtti/%x_%j.out
#SBATCH --error=/home/momoshog/Tornado/slurm_out/tornado_jtti/%x_%j.err
###############################################################


# Source conda env
#source /home/lydiaks2/.bashrc
source ~/.bashrc
bash 
# conda activate hagelslag

python -u lydia_scripts/scripts_data_pipeline/save_light_model_patches.py \
--input_xarray_path="/ourdisk/hpc/ai2es/tornado/learning_patches_V2/xarray/3D/size_32/forecast_window_5/" \
--output_path="/ourdisk/hpc/ai2es/tornado/learning_patches_V2/xarray/3D_light/size_32/forecast_window_5/" \
--dry_run

#python -u /home/lydiaks2/tornado_project/scripts_data_pipeline/save_light_model_patches.py \
#--input_xarray_path="/ourdisk/hpc/ai2es/tornado/learning_patches_V2/xarray/3D/size_32/forecast_window_5/" \
#--output_path="/ourdisk/hpc/ai2es/tornado/learning_patches_V2/xarray/3D_light/size_32/forecast_window_5/"