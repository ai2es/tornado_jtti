#!/bin/bash
# SBATCH -p debug #normal #debug 
# SBATCH --time=00:10:00
# SBATCH --nodes=1
# SBATCH -n 18
# SBATCH --mem-per-cpu=1000
# SBATCH --job-name=pred_0512
# SBATCH --mail-user=ggantos@ucar.edu
# SBATCH --mail-type=ALL
# SBATCH --chdir=/home/ggantos/tornado_jtti/
# SBATCH --output=/home/ggantos/slurmouts/R-%x.%j.%N.out
# SBATCH --error=/home/ggantos/slurmouts/R-%x.%j.%N.err
# SBATCH --open-mode=truncate
# SBATCH --array=37  #0-36%5
# #SBATCH --array=7-9 #at 35, 40, 35 min #37files

# #########################################################

# Select WoFS file
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "SLURM_ARRAY_JOB_ID: ${SLURM_ARRAY_JOB_ID}"
echo "SLURM_JOBID,SLURM_JOB_ID: ${SLURM_JOBID}, ${SLURM_JOB_ID}"

WOFS_ROOT="/ourdisk/hpc/ai2es/tornado/wofs-preds-2023-hgt"
WOFS_REL_PATH="20230512/1800/ENS_MEM_14"

wofs_files=($(ls $WOFS_ROOT/$WOFS_REL_PATH))
echo "WOFS DIR ${WOFS_ROOT}/${WOFS_REL_PATH}"
echo "WOFS FILES:: [${SLURM_ARRAY_TASK_ID}] ${wofs_files[@]}"

WOFS_FILE=${wofs_files[$SLURM_ARRAY_TASK_ID]}
echo "Predicting for $WOFS_FILE"

TUNER="tor_unet_sample50_50_classweightsNone_hyper"
MODEL="2023_07_20_20_55_39_hp_model00"

# Execute predictions
python lydia_scripts/wofs_raw_predictions_oscer.py --loc_wofs="${WOFS_ROOT}/${WOFS_REL_PATH}/${WOFS_FILE}"  --datetime_format="%Y-%m-%d_%H:%M:%S"  --dir_preds="/ourdisk/hpc/ai2es/tornado/wofs-preds-2023-update/${WOFS_REL_PATH}"  --dir_patches="/ourdisk/hpc/ai2es/tornado/wofs_patches/${WOFS_REL_PATH}" --dir_figs="/ourdisk/hpc/ai2es/tornado/wofs-figs/${WOFS_REL_PATH}/${WOFS_FILE}" --with_nans  -Z  -f U V W WSPD10MAX W_UP_MAX P PB PH PHB HGT --loc_model="/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/unet/ZH_only/tuning/${TUNER}/${MODEL}.h5"  --loc_model_calib="/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/gridrad/preds/${TUNER}/${MODEL}_calibraion_model_iso.pkl"  --file_trainset_stats="/ourdisk/hpc/ai2es/tornado/learning_patches/tensorflow/3D_light/training_int_nontor_tor/training_metadata_ZH_only.nc" --write=1
