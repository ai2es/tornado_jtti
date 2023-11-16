#!/bin/bash
#SBATCH -p normal
#SBATCH --nodes=1
#SBATCH -n 2
#SBATCH --mem=28G
#SBATCH --time=05:00:00
#SBATCH --chdir=/ourdisk/hpc/ai2es/randychase/GewitterGefahr/gewittergefahr/scripts/
#SBATCH --job-name="Step4_2013"
#SBATCH --mail-user=randychase@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --mail-type=END
#SBATCH --output=/home/randychase/slurmouts/R-%x.%j.out
#SBATCH --error=/home/randychase/slurmouts/R-%x.%j.err
#SBATCH --array=32%1

#need all yyyymmdd strings  
mapfile -t SPC_DATE_STRINGS < /ourdisk/hpc/ai2es/tornado/tornado_jtti/scripts/process_gridrad_scripts/orderofprocessing_files/orderofprocessing_spc_dates_2013.txt

#grab the current one that is running from the array param above
this_spc_date_string=${SPC_DATE_STRINGS[$SLURM_ARRAY_TASK_ID]}

#print some things to the .out for debuging if it fails.
echo "running final tracking for:"
echo "Array index = ${SLURM_ARRAY_TASK_ID} ... SPC date = ${this_spc_date_string}"

#source  my python envs
source ~/.bashrc
bash 

conda activate gewitter 

python -u ./reanalyze_storm_tracks.py \
--input_tracking_dir_name="/ourdisk/hpc/ai2es/tornado/prelim_tracking_V2" \
--first_spc_date_string=${this_spc_date_string} \
--last_spc_date_string=${this_spc_date_string} \
--max_velocity_diff_m_s01=30 \
--max_link_distance_m_s01=30 \
--max_join_time_seconds=720 \
--max_join_error_m_s01=30 \
--min_duration_seconds=1 \
--output_tracking_dir_name="/ourdisk/hpc/ai2es/tornado/final_tracking_V2/"