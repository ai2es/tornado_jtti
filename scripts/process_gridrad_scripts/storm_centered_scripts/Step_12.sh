#!/bin/bash
#SBATCH -p ai2es_a100
#SBATCH --nodes=1
#SBATCH -n 4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --chdir=/ourdisk/hpc/ai2es/randychase/GewitterGefahr/gewittergefahr/scripts/
#SBATCH --job-name="Step12_2016"
#SBATCH --mail-user=randychase@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --mail-type=END
#SBATCH --output=/home/randychase/slurmouts/R-%x.%j.out
#SBATCH --error=/home/randychase/slurmouts/R-%x.%j.err
#SBATCH --array=[19,20]%2

#need all yyyymmdd strings  
mapfile -t SPC_DATE_STRINGS < /home/randychase/orderofprocessing_spc_dates_2016.txt

#grab the current one that is running from the array param above
this_spc_date_string=${SPC_DATE_STRINGS[$SLURM_ARRAY_TASK_ID]}

#source my python env
source /home/randychase/.bashrc
bash 

conda activate gewitter

#print some things to the .out for debuging if it fails.
echo "creatign learning examples for:"
echo "Array index = ${SLURM_ARRAY_TASK_ID} ... SPC date = ${this_spc_date_string}"

python -u /ourdisk/hpc/ai2es/randychase/GewitterGefahr/gewittergefahr/scripts/create_input_examples.py \
--input_storm_image_dir_name="/ourdisk/hpc/ai2es/tornado/images" \
--radar_source="gridrad" \
--num_radar_dimensions=3 \
--radar_field_names "reflectivity_dbz" "spectrum_width_m_s01" "vorticity_s01" "divergence_s01" "differential_reflectivity_db" "specific_differential_phase_deg_km01" "correlation_coefficient" \
--radar_heights_m_agl 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 \
--first_spc_date_string=${this_spc_date_string} \
--last_spc_date_string=${this_spc_date_string} \
--input_target_dir_name="/ourdisk/hpc/ai2es/tornado/labels" \
--target_names "tornado_lead-time=0000-3600sec_distance=00000-30000m_min-fujita=0" "tornado_lead-time=0000-3600sec_distance=00000-20000m_min-fujita=0" "tornado_lead-time=0000-3600sec_distance=00000-10000m_min-fujita=0" "tornado_lead-time=0000-3600sec_distance=00000-30000m_min-fujita=2" "tornado_lead-time=0000-3600sec_distance=00000-20000m_min-fujita=2" "tornado_lead-time=0000-3600sec_distance=00000-10000m_min-fujita=2" \
--input_sounding_dir_name="/ourdisk/hpc/ai2es/tornado/interped_soundings/" \
--sounding_lag_time_sec=1800 \
--num_examples_per_in_file=1000000 \
--output_dir_name="/ourdisk/hpc/ai2es/tornado/learning_examples"
