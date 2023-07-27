#!/bin/bash
#SBATCH -p debug #normal #
#SBATCH --time=00:30:00
#SBATCH --nodes=1
##SBATCH -n 20
#SBATCH --mem-per-cpu=1000
#SBATCH --job-name=wofs_preds_20230602_1900_ENS3
#SBATCH --mail-user=monique.shotande@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/momoshog/Tornado/tornado_jtti/
#SBATCH --output=/home/momoshog/Tornado/slurm_out/tornado_jtti/%x_%j.out
#SBATCH --error=/home/momoshog/Tornado/slurm_out/tornado_jtti/%x_%j.err
#SBATCH --array=0-37%20

##########################################################

# Source conda env
source ~/.bashrc
bash
conda activate tf_experiments #tf_tornado #

python --version


# Select WoFS file
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"

WOFS_ROOT="/ourdisk/hpc/ai2es/tornado/wofs-preds-2023"  
WOFS_REL_PATH="20230602/1900/ENS_MEM_3"  

wofs_files=($(ls $WOFS_ROOT/$WOFS_REL_PATH))
echo "WOFS DIR ${WOFS_ROOT}/${WOFS_REL_PATH}"
echo "WOFS FILES:: ${wofs_files[@]}"

WOFS_FILE=${wofs_files[$SLURM_ARRAY_TASK_ID]}
echo "Predicting for $WOFS_FILE"


# Execute predictions
python -u lydia_scripts/wofs_raw_predictions.py \
--loc_wofs="${WOFS_ROOT}/${WOFS_REL_PATH}/${WOFS_FILE}"  \
--datetime_format="%Y-%m-%d_%H:%M:%S"  \
--dir_preds="/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_preds/2023/${WOFS_REL_PATH}"  \
--dir_patches="/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_patches/2023/${WOFS_REL_PATH}" \
--dir_figs="/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_figs/2023/${WOFS_REL_PATH}/${WOFS_FILE}" \
-f U WSPD10MAX W_UP_MAX P PB PH PHB HGT \
--loc_model="/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/unet/ZH_only/tuning/tornado_unet_hyper/2023_06_04_09_55_00_hp_model00.h5" \
--file_trainset_stats="/ourdisk/hpc/ai2es/tornado/learning_patches/tensorflow/3D_light/training_int_nontor_tor/training_metadata_ZH_only.nc" \
--with_nans  \
-Z  \
--write=1 
#--dry_run


# WoFS data prior to 2023
#--loc_wofs="/ourdisk/hpc/ai2es/wofs/${WOFS_REL_PATH}/${WOFS_FILE}"  \


#fields UP_HELI_MAX U U10 V V10

