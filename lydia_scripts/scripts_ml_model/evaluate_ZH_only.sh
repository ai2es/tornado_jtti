#!/bin/bash
#SBATCH --partition=large_mem
#SBATCH --nodes=1
#SBATCH --ntasks=10  #-n 20
#SBATCH --mem=50G
#SBATCH --time=12:00:00  #20:30:00
#SBATCH --job-name=eval_model2
#SBATCH --mail-user=monique.shotande@ou.edu
#SBATCH --mail-type=ALL
##SBATCH --chdir=/home/momoshog/Tornado/tornado_jtti
#SBATCH --output=/home/momoshog/Tornado/slurm_out/tornado_jtti/initialrun_test01__%x_%j.out
#SBATCH --error=/home/momoshog/Tornado/slurm_out/tornado_jtti/initialrun_test01__%x_%j.err

##########################################################

user=$(whoami)

# source Lydia's python env
source /home/lydiaks2/.bashrc
bash
conda activate tf_gpu

#source ~/.tornado_bashrc
#bash 
#conda activate tornado_mosho
#conda activate tf_gpu

python --version

#python -m pip install -U pip
#python -m pip install .
#python -m pip install "~/Tornado/tornado_jtti/setup.py"

echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"


#run the script
#--patch_day_idx=$SLURM_ARRAY_TASK_ID \  # index of the day we are evaluating (ith file of storm masks from which to make the patches)
python -u /home/momoshog/Tornado/tornado_jtti/lydia_scripts/scripts_ml_model/evaluate_ZH_only.py \
--patch_day_idx=$SLURM_ARRAY_TASK_ID \
--input_patch_dir_name="/ourdisk/hpc/ai2es/tornado/learning_patches/xarray/3D/size_32/forecast_window_5" \
--model_checkpoint_path='/ourdisk/hpc/ai2es/tornado/unet/ZH_only/initialrun_model2/initialrun_model2.h5' \
--output_predictions_path='/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/unet/ZH_only/initialrun_model2/' \
--training_data_metadata_path='/ourdisk/hpc/ai2es/tornado/learning_patches/tensorflow/3D_light/training_onehot_tor/training_metadata_ZH_only.nc'


