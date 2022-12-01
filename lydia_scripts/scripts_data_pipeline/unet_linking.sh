#!/bin/bash
#SBATCH --partition=ai2es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=1024
#SBATCH --time=00:10:00
#SBATCH --job-name=unet_link
#SBATCH --mail-user=lydiaks2@illinois.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/lydiaks2/tornado_project/output/R-%x.%j.out
#SBATCH --error=/home/lydiaks2/tornado_project/output/R-%x.%j.err
#SBATCH --array=2-2%1

#source my python env
source /home/lydiaks2/.bashrc
bash 

conda activate gewitter 
#this_spc_date_string='20130520'

python -u /home/lydiaks2/tornado_project/scripts_data_pipeline/unet_linking.py \
--this_spc_date_string_array_val=$SLURM_ARRAY_TASK_ID \
--input_linked_tornado_dir_name="/ourdisk/hpc/ai2es/tornado/linked/" \
--input_tracking_dir_name="/ourdisk/hpc/ai2es/tornado/final_tracking/" \
--input_radar_dir_name="/ourdisk/hpc/ai2es/tornado/gridrad_gridded/" \
--output_labeled_storm_dir_name="/ourdisk/hpc/ai2es/tornado/labels_unet/" \
--output_storm_mask_dir_name="/ourdisk/hpc/ai2es/tornado/storm_mask_unet/"



