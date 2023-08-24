#!/bin/bash
#SBATCH -p debug #normal #
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH -n 15
#SBATCH --mem-per-cpu=1000
#SBATCH --job-name=wp10_1900
#SBATCH --mail-user=monique.shotande@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/momoshog/Tornado/tornado_jtti/
#SBATCH --output=/home/momoshog/Tornado/slurm_out/tornado_jtti/%x_%j_out.txt
#SBATCH --error=/home/momoshog/Tornado/slurm_out/tornado_jtti/%x_%j_err.txt
#SBATCH --array=4
##SBATCH --array=0-36%6
##SBATCH --array=7-9 #at 35, 40, 35 min #37files

##########################################################

# Source conda env
source ~/.bashrc
bash
conda activate tf_experiments #tf_tornado #

python --version


# Select WoFS file
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"

WOFS_ROOT="/ourdisk/hpc/ai2es/tornado/wofs-preds-2023"  
WOFS_REL_PATH="20230602/1900/ENS_MEM_10"  

wofs_files=($(ls $WOFS_ROOT/$WOFS_REL_PATH))
echo "WOFS DIR ${WOFS_ROOT}/${WOFS_REL_PATH}"
echo "WOFS FILES:: [${SLURM_ARRAY_TASK_ID}] ${wofs_files[@]}"

WOFS_FILE=${wofs_files[$SLURM_ARRAY_TASK_ID]}
echo "Predicting for $WOFS_FILE"


# Execute predictions
python lydia_scripts/wofs_raw_predictions.py --loc_wofs="${WOFS_ROOT}/${WOFS_REL_PATH}/${WOFS_FILE}"  --datetime_format="%Y-%m-%d_%H:%M:%S"  --dir_preds="/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_preds/2023/${WOFS_REL_PATH}"  --dir_patches="/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_patches/2023/${WOFS_REL_PATH}" --dir_figs="/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_figs/2023/${WOFS_REL_PATH}/${WOFS_FILE}" --with_nans  -Z  -f U WSPD10MAX W_UP_MAX P PB PH PHB HGT --loc_model="/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/unet/ZH_only/tuning/tor_unet_sample50_50_classweightsNone_hyper/2023_07_20_20_55_39_hp_model00.h5"  --loc_model_calib="/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/gridrad/preds/calibraion_model_iso.pkl"  --file_trainset_stats="/ourdisk/hpc/ai2es/tornado/learning_patches/tensorflow/3D_light/training_int_nontor_tor/training_metadata_ZH_only.nc" --write=3

#--dry_run

#--loc_model="/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/unet/ZH_only/tuning/tornado_unet_hyper/2023_06_04_09_55_00_hp_model00.h5"
# WoFS data prior to 2023
#--loc_wofs="/ourdisk/hpc/ai2es/wofs/${WOFS_REL_PATH}/${WOFS_FILE}"  \

#fields UP_HELI_MAX U U10 V V10