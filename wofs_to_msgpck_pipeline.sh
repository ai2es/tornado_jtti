#!/usr/bin/bash -l
azcopy login --identity
python process_wofs.py  \
--account_url_wofs="https://storwofstest003.queue.core.windows.net/?sv=2019-02-02&st=2023-03-09T20%3A39%3A15Z&se=2024-01-01T06%3A00%3A00Z&sp=rp&sig=biy6JZg2n4Wmg%2BLHF4QnVLQvt%2F4W8oYJhXMiaTkyj4U%3D"  \
--queue_name_wofs="wofs-ucar"  \
--blob_url_ncar="https://wofsdltornado.blob.core.windows.net"  \
--vm_datadrive="/datadrive2"  \
--dir_wofs="wrf-wofs"  \
--dir_preds="wofs-preds"  \
--dir_patches="wofs-patches"  \
--datetime_format="%Y-%m-%d_%H:%M:%S"  \
--with_nans  \
--ZH_only  \
--fields UP_HELI_MAX U V W WSPD10MAX W_UP_MAX  \
--dir_preds_msgpk="wofs-preds-msgpk" \
--variables ML_PREDICTED_TOR COMPOSITE_REFL_10CM UP_HELI_MAX \
--thresholds 0.08 20 25 \
--loc_model="lydia_scripts/models/2023_04_06_18_23_50/2023_04_06_18_23_50_hp_model01.h5"  \
--file_trainset_stats="lydia_scripts/models/2023_04_06_18_23_50/training_metadata_ZH_only.nc"  \
--write=1  \
load_weights_hps  \
--hp_path="lydia_scripts/models/2023_04_06_18_23_50/2023_04_06_18_23_50_hps.csv"  \
--hp_idx=1  \
|& tee ./logs/$(date +"%Y%m%d_%H%M%S")_output.txt
