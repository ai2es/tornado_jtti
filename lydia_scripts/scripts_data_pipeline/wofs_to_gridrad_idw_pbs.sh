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

#602 total days

source ~/.bashrc
conda activate tf_tornado

python /glade/work/ggantos/tornado_jtti/lydia_scripts/scripts_data_pipeline/wofs_to_gridrad_idw.py \
--path_to_raw_wofs="/glade/p/cisl/aiml/jtti_tornado/wofs/2019/20190520/0030/*" \
--model_init_date="20190520" \
--model_init_time="0030" \
--output_patches_path="/glade/p/cisl/aiml/jtti_tornado/wofs_patched_4/size_32/" \
--patch_size=32 \
--with_nans=1