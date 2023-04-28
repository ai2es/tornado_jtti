azcopy login --identity

BLOB_URL_NCAR="https://wofsdltornado.blob.core.windows.net"
ACCOUNT_URL_NCAR="https://wofsdltornado.queue.core.windows.net/?sv=2021-12-02&ss=bfqt&srt=sco&sp=rwdlacupiytfx&se=2023-06-15T15:00:00Z&st=2023-04-26T15:00:00Z&spr=https&sig=H2JOkeMn0UuhOqyuifMc%2BCfoSXoN5ZRL7mCe9iGEjBM%3D"
QUEUE_NAME_NCAR_WOFS_TO_PREDS="queue-wofs-to-preds"
QUEUE_NAME_NCAR_PREDS_TO_MSGPK="queue-preds-to-msgpk"

python real_time_scripts/download_wofs.py  \
--account_url_wofs='https://storwofstest003.queue.core.windows.net/?sv=2019-02-02&st=2023-03-09T20%3A39%3A15Z&se=2024-01-01T06%3A00%3A00Z&sp=rp&sig=biy6JZg2n4Wmg%2BLHF4QnVLQvt%2F4W8oYJhXMiaTkyj4U%3D'  \
--queue_name_wofs='wofs-ucar'  \
--blob_url_ncar=$BLOB_URL_NCAR  \
--account_url_ncar=$ACCOUNT_URL_NCAR  \
--queue_name_ncar_wofs_to_preds=$QUEUE_NAME_NCAR_WOFS_TO_PREDS

python real_time_scripts/wofs_to_preds.py  \
--account_url_ncar=$ACCOUNT_URL_NCAR  \
--queue_name_ncar_wofs_to_preds=$QUEUE_NAME_NCAR_WOFS_TO_PREDS  \
--queue_name_ncar_preds_to_msgpk=$QUEUE_NAME_NCAR_PREDS_TO_MSGPK  \
--blob_url_ncar=$BLOB_URL_NCAR  \
--vm_datadrive='/datadrive2'
--dir_wofs='wrf-wofs'  \
--dir_preds="wofs-preds"  \
--dir_patches="wofs-patches" \
--datetime_format="%Y-%m-%d_%H:%M:%S"  \
--with_nans  \
--fields U WSPD10MAX W_UP_MAX \
--loc_model="lydia_scripts/models/initialrun_model8/initialrun_model8.h5"  \
--file_trainset_stats="lydia_scripts/training_metadata/3D_light/training_onehot_tor/training_metadata_ZH_only.nc" \
--write=4 \
--debug_on

python real_time_scripts/preds_to_msgpk.py \
--account_url_ncar=$ACCOUNT_URL_NCAR  \
--queue_name_ncar_preds_to_msgpk=$QUEUE_NAME_NCAR_PREDS_TO_MSGPK  \
--variable="ML_PREDICTED_TOR" \
--threshold=0.08
