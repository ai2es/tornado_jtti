#!/bin/bash

#SBATCH --partition=debug_gpu
#SBATCH --time=00:30:00
##SBATCH --partition=gpu_kepler
##SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=20  #-n 20
#SBATCH --mem=10G
#SBATCH --job-name=tuning_JTTI
#SBATCH --chdir=/home/momoshog/Tornado/
#SBATCH --output=slurm_out/tornado_jtti/debug__%x_%j.out
#SBATCH --error=slurm_out/tornado_jtti/debug__%x_%j.err
#SBATCH --array=0-1
##SBATCH --array=0-1000%20
#SBATCH --mail-user=$USREMAIL
#SBATCH --mail-type=ALL

##########################################################

# Source conda
source ~/.bashrc
bash 
# TODO: if conda env tornado does not exist
#conda env create --name tornado --file environment.yml
conda activate tf_tornado

python --version
echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"

# Run hyperparameter search
#tornado_jtti/unet/ZH_only/initialrun_model8/' \
python -u tornado_jtti/lydia_scripts/scripts_tensorboard/tuning_JTTI.py \
--input_dir="/ourdisk/hpc/ai2es/tornado/wofs_patched/size_32/" \
--output_dir='/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/unet/' \
--training_data_metadata_path='/ourdisk/hpc/ai2es/tornado/learning_patches/tensorflow/3D_light/training_onehot_tor/training_metadata_ZH_only.nc' \
--path_to_wofs='/ourdisk/hpc/ai2es/wofs/' \
--wofs_day_idx=$SLURM_ARRAY_TASK_ID \
--patch_size=32  \
--dry_run
