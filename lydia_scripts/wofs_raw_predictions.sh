#!/bin/bash
##SBATCH -p normal
##SBATCH --time=5:00:00
#SBATCH -p debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
##SBATCH -n 20
#SBATCH --mem-per-cpu=1000
#SBATCH --job-name=wofs_raw_predictions_master
#SBATCH --mail-user=monique.shotande@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/momoshog/Tornado/tornado_jtti_master/tornado_jtti/
#SBATCH --output=/home/momoshog/Tornado/slurm_out/tornado_jtti/%x_%j.out
#SBATCH --error=/home/momoshog/Tornado/slurm_out/tornado_jtti/%x_%j.err
#SBATCH --array=0

##########################################################

# Source conda env
source ~/.bashrc
bash
conda activate tf_tornado

python --version
echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"


#WOFS_REL_PATH="2019/20190430/1930/ENS_MEM_2"
#WOFS_FILE="wrfwof_d01_2019-04-30_19:50:00"

WOFS_REL_PATH="2019/20190430/1930/ENS_MEM_2"
WOFS_FILE="wrfwof_d01_2019-04-30_20:20:00"
#/ourdisk/hpc/ai2es/wofs/2019/20190430/1930/ENS_MEM_2
#YYYY-MM-DD_hh:mm:ss

python lydia_scripts/wofs_raw_predictions.py \
--loc_wofs="/ourdisk/hpc/ai2es/wofs/${WOFS_REL_PATH}/${WOFS_FILE}"  \
--datetime_format="%Y-%m-%d_%H:%M:%S"  \
--dir_preds="/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_preds/${WOFS_REL_PATH}"  \
--dir_patches="/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_patches/${WOFS_REL_PATH}" \
--dir_figs="/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_figs/${WOFS_REL_PATH}/${WOFS_FILE}" \
--with_nans  \
--loc_model="/ourdisk/hpc/ai2es/tornado/unet/ZH_only/initialrun_model8/initialrun_model8.h5"  \
--file_trainset_stats="/ourdisk/hpc/ai2es/tornado/learning_patches/tensorflow/3D_light/training_onehot_tor/training_metadata_ZH_only.nc" \
--write=2  \
--dry_run
