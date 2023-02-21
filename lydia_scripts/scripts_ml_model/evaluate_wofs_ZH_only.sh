#!/bin/bash
##SBATCH --partition=normal
##SBATCH --time=04:45:00
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=20  #-n 20
#SBATCH --mem=10G
##SBATCH --exclusive
#SBATCH --job-name=evaluate_wofs
#SBATCH --mail-user=monique.shotande@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/momoshog/Tornado/tornado_jtti/
#SBATCH --output=/home/momoshog/Tornado/slurm_out/tornado_jtti/env__%x_%j.out
#SBATCH --error=/home/momoshog/Tornado/slurm_out/tornado_jtti/env__%x_%j.err
#SBATCH --array=37
##SBATCH --array=0-1000%20

##########################################################

# Source Lydia's python env
#source /home/lydiaks2/.bashrc
#bash 
#conda activate tf_gpu

# Source conda
source ~/.bashrc
bash 
# TODO: if conda env tornado does not exist
#conda env create --name tornado --file environment.yml
conda activate tf_tornado

python --version
echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"

#run the script
#--output_predictions_checkpoint_path='/ourdisk/hpc/ai2es/tornado/unet/ZH_only/initialrun_model8/' \
#python -u /home/momoshog/Tornado/tornado_jtti/lydia_scripts/scripts_ml_model/evaluate_wofs_ZH_only.py \
python -u lydia_scripts/scripts_ml_model/evaluate_wofs_ZH_only.py \
--wofs_day_idx=$SLURM_ARRAY_TASK_ID \
--input_patch_dir_name="/ourdisk/hpc/ai2es/tornado/wofs_patched/size_32/" \
--output_model_checkpoint_path='/ourdisk/hpc/ai2es/tornado/unet/ZH_only/initialrun_model8/initialrun_model8.h5' \
--output_predictions_checkpoint_path='/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/unet/ZH_only/initialrun_model8/' \
--training_data_metadata_path='/ourdisk/hpc/ai2es/tornado/learning_patches/tensorflow/3D_light/training_onehot_tor/training_metadata_ZH_only.nc' \
--path_to_wofs='/ourdisk/hpc/ai2es/wofs/' \
--patch_size=32  \
--dry_run
