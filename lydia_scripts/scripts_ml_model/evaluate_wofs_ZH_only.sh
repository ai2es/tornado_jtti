#!/bin/bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH -n 20
#SBATCH --ntasks=20
#SBATCH --mem=10G
#SBATCH --time=08:45:00
#SBATCH --exclusive
#SBATCH --job-name=evaluate_wofs
#SBATCH --mail-user=lydiaks2@illinois.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/lydiaks2/GewitterGefahr/gewittergefahr/scripts/
#SBATCH --output=/home/lydiaks2/tornado_project/output/R-%x.%j.out
#SBATCH --error=/home/lydiaks2/tornado_project/output/R-%x.%j.err
#SBATCH --array=0-1000%20

#array=0-1000%20

#source my python env
source /home/lydiaks2/.bashrc
bash 
conda activate tf_gpu

#run the script
python -u /home/lydiaks2/tornado_project/scripts_ml_model/evaluate_wofs_ZH_only.py \
--wofs_day_idx=$SLURM_ARRAY_TASK_ID \
--input_patch_dir_name="/ourdisk/hpc/ai2es/tornado/wofs_patched/size_32/" \
--output_model_checkpoint_path='/ourdisk/hpc/ai2es/tornado/unet/ZH_only/initialrun_model8/initialrun_model8.h5' \
--output_predictions_checkpoint_path='/ourdisk/hpc/ai2es/tornado/unet/ZH_only/initialrun_model8/' \
--training_data_metadata_path='/ourdisk/hpc/ai2es/tornado/learning_patches/tensorflow/3D_light/training_onehot_tor/training_metadata_ZH_only.nc' \
--path_to_wofs='/ourdisk/hpc/ai2es/wofs/' \
--patch_size=32