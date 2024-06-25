#!/bin/bash
#SBATCH -p normal
#SBATCH --nodes=1
#SBATCH -n 2
#SBATCH --mem=4G
#SBATCH --time=08:00:00
#SBATCH --chdir=/ourdisk/hpc/ai2es/randychase/GewitterGefahr/gewittergefahr/scripts/
#SBATCH --job-name="Step11_2016"
#SBATCH --mail-user=randychase@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --mail-type=END
#SBATCH --output=/home/randychase/slurmouts/R-%x.%j.out
#SBATCH --error=/home/randychase/slurmouts/R-%x.%j.err
#SBATCH --array=74%2

#need all yyyymmdd strings  
mapfile -t SPC_DATE_STRINGS < /home/randychase/orderofprocessing_spc_dates_2016.txt

#grab the current one that is running from the array param above
this_spc_date_string=${SPC_DATE_STRINGS[$SLURM_ARRAY_TASK_ID]}

#source my python env
source /home/randychase/.bashrc
bash 

conda activate gewitter

#print some things to the .out for debuging if it fails.
echo "interpolating soundings to storms for"
echo "Array index = ${SLURM_ARRAY_TASK_ID} ... SPC date = ${this_spc_date_string}"

python -u  /ourdisk/hpc/ai2es/randychase/GewitterGefahr/gewittergefahr/scripts/interp_soundings_to_storm_objects.py \
--spc_date_string=${this_spc_date_string} \
--lead_times_seconds 0 450 900 1350 1800 2250 2700 3150 3600 \
--lag_time_for_convective_contamination_sec=1800 \
--input_ruc_directory_name="/condo/swatwork/ralager/ruc_data" \
--input_rap_directory_name="/ourdisk/hpc/ai2es/tornado/rap_data" \
--input_tracking_dir_name="/ourdisk/hpc/ai2es/tornado/conus_only" \
--input_elevation_dir_name="/ourdisk/hpc/ai2es/tornado/elevation" \
--output_sounding_dir_name="/ourdisk/hpc/ai2es/tornado/interped_soundings/"