#!/bin/bash
#SBATCH --partition=large_mem
#SBATCH --nodes=1
#SBATCH -n 20
#SBATCH --ntasks=20
#SBATCH --mem=50G
#SBATCH --time=20:30:00
#SBATCH --job-name=evaluate
#SBATCH --mail-user=lydiaks2@illinois.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/lydiaks2/tornado_project/output/R-%x.%j.out
#SBATCH --error=/home/lydiaks2/tornado_project/output/R-%x.%j.err

#source my python env
source /home/lydiaks2/.bashrc
bash 
conda activate tf_gpu

#run the script
python -u /home/lydiaks2/tornado_project/scripts_ml_model/evaluate_ZH_only.py \
--patch_day_idx=$SLURM_ARRAY_TASK_ID \
--input_patch_dir_name="/ourdisk/hpc/ai2es/tornado/learning_patches/xarray/3D/size_32/forecast_window_5" \
--model_checkpoint_path='/ourdisk/hpc/ai2es/tornado/unet/ZH_only/initialrun_model2/initialrun_model2.h5' \
--output_predictions_path='/ourdisk/hpc/ai2es/tornado/unet/ZH_only/initialrun_model2/' \
--training_data_metadata_path='/ourdisk/hpc/ai2es/tornado/learning_patches/tensorflow/3D_light/training_onehot_tor/training_metadata_ZH_only.nc'


