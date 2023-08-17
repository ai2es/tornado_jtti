#!/bin/bash
#SBATCH -p normal
#SBATCH --nodes=1
#SBATCH -n 2
#SBATCH --mem=28G
#SBATCH --time=8:00:00
#SBATCH --chdir=/ourdisk/hpc/ai2es/alexnozka/GewitterGefahr/gewittergefahr/scripts
#SBATCH --job-name="S3C1_2013"
#SBATCH --mail-user=alexander.j.nozka-1@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --mail-type=END
#SBATCH --output=/ourdisk/hpc/ai2es/alexnozka/debug/R-%x.%j.out
#SBATCH --error=/ourdisk/hpc/ai2es/alexnozka/debug/R-%x.%j.err
#SBATCH --array=15,25,34,35,38,43,44,45,49,53,54,55,57,58,59,60,64,80%2

#need all yyyymmdd strings  
mapfile -t SPC_DATE_STRINGS < /ourdisk/hpc/ai2es/tornado/tornado_jtti/scripts/process_gridrad_scripts/orderofprocessing_files/orderofprocessing_spc_dates_2013.txt

#grab the current one that is running from the array param above
this_spc_date_string=${SPC_DATE_STRINGS[$SLURM_ARRAY_TASK_ID]}

#print some things to the .out for debuging if it fails.
echo "running prelim tracking for:"
echo "Array index = ${SLURM_ARRAY_TASK_ID} ... SPC date = ${this_spc_date_string}"

#source  my python envs
source ~/.bashrc
bash 

conda activate gewitter 

python -u ./run_echo_top_tracking.py \
--input_radar_dir_name="/ourdisk/hpc/ai2es/tornado/gridrad_myrorss_V2" \
--input_echo_classifn_dir_name="/ourdisk/hpc/ai2es/tornado/echo_class_V2" \
--first_spc_date_string=${this_spc_date_string} \
--last_spc_date_string=${this_spc_date_string} \
--echo_top_field_name="echo_top_40dbz_km" \
--min_echo_top_km=4 \
--min_size_pixels=5 \
--max_velocity_diff_m_s01=30 \
--max_link_distance_m_s01=30 \
--output_tracking_dir_name="/ourdisk/hpc/ai2es/tornado/prelim_tracking_V2"
