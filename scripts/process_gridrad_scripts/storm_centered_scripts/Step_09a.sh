#!/bin/bash
#SBATCH -p normal 
#SBATCH --nodes=1
#SBATCH -n 2
#SBATCH --mem=8G
#SBATCH --time=17:00:00
#SBATCH --chdir=/ourdisk/hpc/ai2es/randychase/GewitterGefahr/gewittergefahr/scripts/
#SBATCH --job-name="Step9a_2013_redoday"
#SBATCH --mail-user=randychase@ou.edu 
#SBATCH --mail-type=ALL
#SBATCH --mail-type=END
#SBATCH --output=/home/randychase/slurmouts/R-%x.%j.out
#SBATCH --error=/home/randychase/slurmouts/R-%x.%j.err
#SBATCH --array=27%1

#need all yyyymmdd strings  
mapfile -t SPC_DATE_STRINGS < /home/randychase/orderofprocessing_spc_dates.txt

#grab the current one that is running from the array param above
this_spc_date_string=${SPC_DATE_STRINGS[$SLURM_ARRAY_TASK_ID]}

#print some things to the .out for debuging if it fails.
echo "running extract for "
echo "Split index = 0, array index $SLURM_ARRAY_TASK_ID ... SPC date = ${this_spc_date_string}"

#source my python env
source /home/randychase/.bashrc
bash 

conda activate gewitter 

python -u /ourdisk/hpc/ai2es/randychase/GewitterGefahr/gewittergefahr/scripts/extract_storm_images_from_gridrad.py \
--num_rows_per_image=48 \
--num_columns_per_image=48 \
--rotate_grids=1 \
--rotated_grid_spacing_metres=1500 \
--radar_field_names "reflectivity_dbz" "spectrum_width_m_s01" "vorticity_s01" "divergence_s01" "differential_reflectivity_db" "specific_differential_phase_deg_km01" "correlation_coefficient" \
--radar_heights_m_agl 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 \
--spc_date_string=${this_spc_date_string} \
--input_radar_dir_name="/ourdisk/hpc/ai2es/tornado/gridrad_gridded" \
--input_tracking_dir_name="/ourdisk/hpc/ai2es/tornado/conus_only" \
--input_elevation_dir_name="/ourdisk/hpc/ai2es/tornado/elevation/" \
--new_gridrad_version=True \
--output_dir_name="/ourdisk/hpc/ai2es/tornado/images/" \
--n_splits=4 \
--current_split=0 \
--random_split=True \