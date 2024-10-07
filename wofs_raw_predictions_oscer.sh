#!/usr/bin/bash -l
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=18
#SBATCH --mem-per-cpu=8G
#SBATCH --time=04:00:00
#SBATCH --chdir=/home/ggantos/tornado_jtti
#SBATCH --job-name=pred_0512
#SBATCH --mail-user=ggantos@ucar.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/ggantos/slurmouts/R-%A_%a.out
#SBATCH --error=/home/ggantos/slurmouts/R-%A_%a.err
#SBATCH --open-mode=truncate
#SBATCH --array=21

# #########################################################

# Select WoFS file
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "SLURM_ARRAY_JOB_ID: ${SLURM_ARRAY_JOB_ID}"
echo "SLURM_JOBID,SLURM_JOB_ID: ${SLURM_JOBID}, ${SLURM_JOB_ID}"

WOFS_ROOT="/ourdisk/hpc/ai2es/tornado/wofs-preds-2023-hgt"
DATE=""$date""

INIT_TIMES=($(ls $WOFS_ROOT/$DATE))

echo "WOFS DIR:: ${WOFS_ROOT}/${DATE}"
echo "INIT TIMES:: ${INIT_TIMES[@]}"

INIT_TIME=${INIT_TIMES[$SLURM_ARRAY_TASK_ID]}
echo "Predicting for $INIT_TIME"

TUNER="tor_unet_sample50_50_classweightsNone_hyper"
MODEL="2023_07_20_20_55_39_hp_model00"

echo "CPUs $SLURM_CPUS_PER_TASK"

# Execute predictions
python -u process_wofs_oscer.py --loc_wofs="${WOFS_ROOT}/${DATE}/${INIT_TIME}"  --datetime_format="%Y-%m-%d_%H:%M:%S"  --dir_preds="/ourdisk/hpc/ai2es/tornado/wofs-preds-2023-update/${DATE}"  --dir_patches="/ourdisk/hpc/ai2es/tornado/wofs_patches/${DATE}"  --with_nans  -Z  -f U V W WSPD10MAX W_UP_MAX P PB PH PHB HGT --loc_model="/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/unet/ZH_only/tuning/${TUNER}/${MODEL}.h5"  --loc_model_calib="/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/gridrad/preds/${TUNER}/${MODEL}_calibraion_model_iso.pkl"  --file_trainset_stats="/ourdisk/hpc/ai2es/tornado/learning_patches/tensorflow/3D_light/training_int_nontor_tor/training_metadata_ZH_only.nc" --variables ML_PREDICTED_TOR COMPOSITE_REFL_10CM UP_HELI_MAX --thresholds 0.08 20 25 --write=1
