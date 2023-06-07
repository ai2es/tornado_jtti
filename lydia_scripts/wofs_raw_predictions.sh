#!/bin/bash
#SBATCH -p normal #debug #
#SBATCH --time=00:35:00
#SBATCH --nodes=1
##SBATCH -n 20
#SBATCH --mem-per-cpu=1000
#SBATCH --job-name=newgridrad__wofs_raw_predictions
#SBATCH --mail-user=monique.shotande@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/momoshog/Tornado/tornado_jtti/
#SBATCH --output=/home/momoshog/Tornado/slurm_out/tornado_jtti/%x_%j.out
#SBATCH --error=/home/momoshog/Tornado/slurm_out/tornado_jtti/%x_%j.err

##########################################################

# Source conda env
source ~/.bashrc
bash
conda activate tf_experiments #tf_tornado #
#conda uninstall protobuf
#>conda install -c conda-forge tensorflow
#conda install -c conda-forge tensorflow-gpu
#conda install tensorflow-gpu
#conda install -c conda-forge tensorboard
#>conda install -c conda-forge keras-tuner
#conda install -c conda-forge wandb
#conda clean --all -v
#conda uninstall tensorflow_datasets
#conda remove cftime

python --version

WOFS_ROOT="/ourdisk/hpc/ai2es/wofs" #"/ourdisk/hpc/ai2es/tornado/wofs-preds-2023"  #
WOFS_REL_PATH="2019/20190430/1930/ENS_MEM_2" #"20230511/1800/ENS_MEM_17"  #
WOFS_FILE="wrfwof_d01_2019-04-30_20:35:00" #"wrfwof_d01_2023-05-11_19_05_00_predictions.nc"  #

#WOFS_REL_PATH="2019/20190517/0300/ENS_MEM_12"
#WOFS_FILE="wrfwof_d01_2019-05-18_03:15:00"

python -u lydia_scripts/wofs_raw_predictions.py \
--loc_wofs="${WOFS_ROOT}/${WOFS_REL_PATH}/${WOFS_FILE}"  \
--datetime_format="%Y-%m-%d_%H:%M:%S"  \
--dir_preds="/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_preds/${WOFS_REL_PATH}"  \
--dir_patches="/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_patches/${WOFS_REL_PATH}" \
--dir_figs="/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_figs/${WOFS_REL_PATH}/${WOFS_FILE}" \
--with_nans  \
-Z  \
-f U WSPD10MAX W_UP_MAX \
--loc_model="/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/unet/ZH_only/tuning/tornado_unet_hyper/2023_06_04_09_55_00_hp_model00.h5" \
--file_trainset_stats="/ourdisk/hpc/ai2es/tornado/learning_patches/tensorflow/3D_light/training_int_nontor_tor/training_metadata_ZH_only.nc" \
--write=4 
#--dry_run 

#--loc_wofs="/ourdisk/hpc/ai2es/wofs/${WOFS_REL_PATH}/${WOFS_FILE}"  \

#--loc_model="/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/unet/ZH_only/tuning/tornado_unet_hyper/2023_05_02_17_29_29_hp_model00.h5" \ ## trained on old gridrad

#--loc_model="/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/unet/ZH_only/tuning/tornado_unet_hyper/2023_04_06_18_23_50_hp_model01.h5" \
#--file_trainset_stats="/ourdisk/hpc/ai2es/tornado/learning_patches/tensorflow/3D_light/training_int_tor/training_metadata_ZH_only.nc" \
#load_weights_hps \
#--hp_path="/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/unet/ZH_only/tuning/tornado_unet_hyper/2023_04_06_18_23_50_hps.csv" \
#--hp_idx=1 

#--loc_model="/ourdisk/hpc/ai2es/tornado/unet/ZH_only/initialrun_model2/initialrun_model2.h5" \
#--file_trainset_stats="/ourdisk/hpc/ai2es/tornado/learning_patches/tensorflow/3D_light/training_onehot_nontor_tor/training_metadata_ZH_only.nc" \

#--loc_model="/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/unet/ZH_only/tuning/tornado_unet_hyper/model00_2023_03_21_05_03_06.h5" \
#--file_trainset_stats="/ourdisk/hpc/ai2es/tornado/learning_patches/tensorflow/3D_light/training_int_tor/training_metadata_ZH_only.nc" \
#load_weights_hps \
#--hp_path="/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/unet/ZH_only/tuning/tornado_unet_hyper/hps_2023_03_21_05_03_06.csv" \


#fields UP_HELI_MAX U U10 V V10
#YYYY-MM-DD_hh:mm:ss
#--file_trainset_stats="/ourdisk/hpc/ai2es/tornado/learning_patches/tensorflow/3D_light/training_onehot_tor/training_metadata_ZH_only.nc"

