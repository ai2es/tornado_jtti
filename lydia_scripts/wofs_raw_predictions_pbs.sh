#!/bin/bash -l 

#PBS -l select=1:ncpus=2:ngpus=0:mem=32GB
#PBS -l walltime=00:00:10
#PBS -A NAML0001
#PBS -q casper
#PBS -N wofs_to_gridrad
#PBS -j eo
#PBS -k eod
#PBS -o /glade/work/ggantos/tornado_jtti/lydia_scripts/output/results.out
#PBS -M ggantos@ucar.edu

source ~/.bashrc
conda activate tf_tornado

WOFS_REL_PATH="2019/20190520/1930/ENS_MEM_2"
WOFS_FILE="wrfwof_d01_2019-05-20_20:20:00"

!python /glade/work/ggantos/tornado_jtti/lydia_scripts/wofs_raw_predictions_casper.py \
--loc_wofs="/glade/p/cisl/aiml/jtti_tornado/wofs/{WOFS_REL_PATH}/{WOFS_FILE}"  \
--datetime_format="%Y-%m-%d_%H:%M:%S"  \
--dir_preds="/glade/p/cisl/aiml/jtti_tornado/wofs_preds/{WOFS_REL_PATH}"  \
--dir_patches="/glade/p/cisl/aiml/jtti_tornado/wofs_patches/{WOFS_REL_PATH}"  \
--dir_figs="/glade/p/cisl/aiml/jtti_tornado/wofs_figs/{WOFS_REL_PATH}/{WOFS_FILE}"  \
--with_nans  \
--loc_model="/glade/work/ggantos/tornado_jtti/lydia_scripts/models/initialrun_model8/initialrun_model8.h5"  \
--file_trainset_stats="/glade/work/ggantos/tornado_jtti/lydia_scripts/training_metadata/3D_light/training_onehot_tor/training_metadata_ZH_only.nc" \
--write=2

