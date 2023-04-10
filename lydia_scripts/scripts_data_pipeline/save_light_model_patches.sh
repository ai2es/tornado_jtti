#!/bin/bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH -n 20
#SBATCH --ntasks=20
#SBATCH --mem=10G
#SBATCH --time=10:30:00
#SBATCH --job-name=save_light_data
#SBATCH --mail-user=lydiaks2@illinois.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/lydiaks2/tornado_project/output/R-%x.%j.out
#SBATCH --error=/home/lydiaks2/tornado_project/output/R-%x.%j.err

#source my python env
source /home/lydiaks2/.bashrc
bash 

#conda activate hagelslag
python -u /home/lydiaks2/tornado_project/scripts_data_pipeline/save_light_model_patches.py \
--input_xarray_path="/ourdisk/hpc/ai2es/tornado/learning_patches_V2/xarray/3D/size_32/forecast_window_5/" \
--output_path="/ourdisk/hpc/ai2es/tornado/learning_patches_V2/xarray/3D_light/size_32/forecast_window_5/"