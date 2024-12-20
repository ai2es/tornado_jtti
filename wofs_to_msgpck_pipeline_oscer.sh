#!/usr/bin/bash -l
#SBATCH -p debug #normal
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=16G
#SBATCH --chdir=/home/ggantos/tornado_jtti/
#SBATCH --job-name="mpk_0512"
#SBATCH --mail-user=ggantos@ucar.edu
#SBATCH --mail-type=ALL
#SBATCH --mail-type=END
#SBATCH --output=/home/ggantos/slurmouts/R-%x.%j.out
#SBATCH --error=/home/ggantos/slurmouts/R-%x.%j.err

export AZCOPY_AUTO_LOGIN_TYPE="MSI"
python -u process_wofs_oscer.py  \
--account_url_wofs="https://storwofstest003.queue.core.windows.net/?sv=2019-02-02&st=2023-03-09T20%3A39%3A15Z&se=2024-01-01T06%3A00%3A00Z&sp=rp&sig=biy6JZg2n4Wmg%2BLHF4QnVLQvt%2F4W8oYJhXMiaTkyj4U%3D"  \
--queue_name_wofs="wofs-ucar"  \
--blob_url_ncar="https://wofsdltornado.blob.core.windows.net"  \
--hours_to_analyze=3  \
--vm_datadrive="/ourdisk/hpc/ai2es/tornado"  \
--date="20230512"  \
--dir_wofs="wofs-preds-2023-hgt"  \
--dir_preds="wofs-preds-2023-update"  \
--dir_patches="wofs-patches"  \
--datetime_format="%Y-%m-%d_%H:%M:%S"  \
--with_nans  \
--ZH_only  \
--fields UP_HELI_MAX U V W WSPD10MAX W_UP_MAX P PB PH PHB HGT  \
--dir_preds_msgpk="wofs-dl-preds" \
--variables ML_PREDICTED_TOR COMPOSITE_REFL_10CM UP_HELI_MAX  \
--thresholds 0.08 20 25 \
--loc_model="lydia_scripts/models/2023_07_20_20_55_39/2023_07_20_20_55_39_hp_model00.h5"  \
--file_trainset_stats="lydia_scripts/models/2023_04_06_18_23_50/training_metadata_ZH_only.nc"  \
--write=1  \
|& tee ./logs/$(date +"%Y%m%d_%H%M%S")_output.txt
