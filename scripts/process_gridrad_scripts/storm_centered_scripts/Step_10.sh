#!/bin/bash
#SBATCH -p normal
#SBATCH --nodes=1
#SBATCH -n 2
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --chdir=/ourdisk/hpc/ai2es/randychase/GewitterGefahr/gewittergefahr/scripts/
#SBATCH --job-name="Step10_2013"
#SBATCH --mail-user=slurmslave.randychase@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --mail-type=END
#SBATCH --output=/home/randychase/slurmouts/R-%x.%j.out
#SBATCH --error=/home/randychase/slurmouts/R-%x.%j.err
#SBATCH --array=0-100%2

#need all yyyymmdd strings  
mapfile -t SPC_DATE_STRINGS < /home/randychase/orderofprocessing_spc_dates_2016.txt

#grab the current one that is running from the array param above
this_spc_date_string=${SPC_DATE_STRINGS[$SLURM_ARRAY_TASK_ID]}

#source my python env
source /home/randychase/.bashrc
bash 

conda activate gewitter

#print some things to the .out for debuging if it fails.
echo "running aggregate for"
echo "Array index = ${SLURM_ARRAY_TASK_ID} ... SPC date = ${this_spc_date_string}"

python -u /ourdisk/hpc/ai2es/randychase/GewitterGefahr/gewittergefahr/scripts/agglom_storm_images_by_date.py \
--storm_image_dir_name="/ourdisk/hpc/ai2es/tornado/images/" \
--radar_source="gridrad" \
--spc_date_string=${this_spc_date_string} \
--radar_field_names "reflectivity_dbz" "spectrum_width_m_s01" "vorticity_s01" "divergence_s01" "differential_reflectivity_db" "specific_differential_phase_deg_km01" "correlation_coefficient" \
--radar_heights_m_agl 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 \
