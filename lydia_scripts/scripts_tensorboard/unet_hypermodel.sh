#!/bin/bash

##SBATCH --partition=debug_gpu
##SBATCH --time=00:30:00
##SBATCH --partition=gpu_a100
##SBATCH --time=1:00:00
#SBATCH -p ai2es
#SBATCH --time=24:00:00
#SBATCH --nodelist=c731
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=20  #-n 20
#SBATCH --mem=10G
#SBATCH --job-name=tuning_tests
#SBATCH --chdir=/home/momoshog/Tornado/tornado_jtti
#SBATCH --output=/home/momoshog/Tornado/slurm_out/tornado_jtti/debug__%x_%j.out
#SBATCH --error=/home/momoshog/Tornado/slurm_out/tornado_jtti/debug__%x_%j.err
#SBATCH --mail-user=$USEREMAIL
#SBATCH --mail-type=ALL
##SBATCH --array=0-1000%20
##SBATCH --profile=Task   #data collection on  (I/O, Memory, ...) data is collected. stored in an HDF5 file for the job

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
python -u lydia_scripts/scripts_tensorboard/unet_hypermodel.py \
--in_dir="/ourdisk/hpc/ai2es/tornado/learning_patches/tensorflow/3D_light/training_int_tor/training_ZH_only.tf" \
--in_dir_val="/ourdisk/hpc/ai2es/tornado/learning_patches/tensorflow/3D_light/validation_int_tor/validation1_ZH_only.tf" \
--out_dir="/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/unet/ZH_only/tuning" \
--out_dir_tuning="/scratch/momoshog/Tornado/tornado_jtti/tuning" \
--epochs=1000 \
--batch_size=300 \
--lrate=2e-4 \
--overwrite \
--number_of_summary_trials=3 \
--gpu \
--save=2 \
hyper \
--max_epochs=1000 \
--factor=3 
#--tuner_id="debug" \
#--dry_run \
#--nogo \


#--out_dir="/home/momoshog/Tornado/test_data/tmp" \
#tornado_jtti/unet/ZH_only/initialrun_model8/' \
#python -u lydia_scripts/scripts_tensorboard/tuning_JTTI.py \
#--input_dir="/ourdisk/hpc/ai2es/tornado/wofs_patched/size_32/" \
#--output_dir='/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/unet/' \
#--training_data_metadata_path='/ourdisk/hpc/ai2es/tornado/learning_patches/tensorflow/3D_light/training_onehot_tor/training_metadata_ZH_only.nc' \
#--path_to_wofs='/ourdisk/hpc/ai2es/wofs/' \
#--wofs_day_idx=$SLURM_ARRAY_TASK_ID \
#--patch_size=32  \
#--dry_run
