WOFS_REL_PATH="2019/20190520/0030/ENS_MEM_1"
WOFS_FILE="wrfwof_d01_2019-05-21_00:30:00 "

python wofs_raw_predictions_azure.py \
--loc_wofs="/datadrive/wofs/${WOFS_REL_PATH}/${WOFS_FILE}"  \
--datetime_format="%Y-%m-%d_%H:%M:%S"  \
--dir_preds="/datadrive/wofs_preds/${WOFS_REL_PATH}"  \
--dir_patches="/datadrive/wofs_patches/${WOFS_REL_PATH}" \
--dir_figs="/datadrive/wofs_figs/${WOFS_REL_PATH}/${WOFS_FILE}" \
--with_nans  \
--fields U WSPD10MAX W_UP_MAX \
--loc_model="/home/ggantos/tornado_jtti/lydia_scripts/models/2023_04_06_18_23_50_hp_model01.h5"  \
--file_trainset_stats="/home/ggantos/tornado_jtti/lydia_scripts/training_metadata/3D_light/training_onehot_tor/training_metadata_ZH_only.nc" \
--write=4 \
--dry_run
