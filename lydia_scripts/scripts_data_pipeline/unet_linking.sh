#!/bin/bash
#SBATCH --partition=normal
##SBATCH --nodelist=c733
##SBATCH --gres=gpu:1
##SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=1024
#SBATCH --time=12:00:00
#SBATCH --job-name=2013_2016__unet_link
#SBATCH --mail-user=monique.shotande@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/momoshog/Tornado/tornado_jtti
#SBATCH --output=/home/momoshog/Tornado/slurm_out/tornado_jtti/%x_%j.out
#SBATCH --error=/home/momoshog/Tornado/slurm_out/tornado_jtti/%x_%j.err
#SBATCH --array=0-230%20
#########################################################################

# Source conda
#source /home/lydiaks2/.bashrc
source ~/.bashrc
bash 

conda activate gewitter 
#this_spc_date_string='20130520'

python --version

#python -u /home/lydiaks2/tornado_project/scripts_data_pipeline/unet_linking.py \
python -u lydia_scripts/scripts_data_pipeline/unet_linking.py \
--this_spc_date_string_array_val=$SLURM_ARRAY_TASK_ID \
--input_linked_tornado_dir_name="/ourdisk/hpc/ai2es/tornado/linked_V2/" \
--input_tracking_dir_name="/ourdisk/hpc/ai2es/tornado/final_tracking_V2/" \
--input_radar_dir_name="/ourdisk/hpc/ai2es/tornado/gridrad_gridded_V2/" \
--output_labeled_storm_dir_name="/ourdisk/hpc/ai2es/tornado/labels_unet_V2/" \
--output_storm_mask_dir_name="/ourdisk/hpc/ai2es/tornado/storm_mask_unet_V2/" \
#--dry_run

