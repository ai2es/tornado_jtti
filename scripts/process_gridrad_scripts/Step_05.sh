#!/bin/bash
#SBATCH -p debug
#SBATCH --nodes=1
#SBATCH -n 2
#SBATCH --mem=28G
#SBATCH --time=00:30:00
#SBATCH --chdir=/ourdisk/hpc/ai2es/alexnozka/GewitterGefahr/gewittergefahr/scripts/
#SBATCH --job-name="Step5_2013"
#SBATCH --mail-user=alexander.j.nozka-1@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --mail-type=END
#SBATCH --output=/home/anozka/slurmouts/R-%x.%j.out
#SBATCH --error=/home/anozka/slurmouts/R-%x.%j.err
#SBATCH --array=32%1

#need all yyyymmdd strings  
mapfile -t SPC_DATE_STRINGS < /ourdisk/hpc/ai2es/tornado/tornado_jtti/scripts/process_gridrad_scripts/orderofprocessing_files/orderofprocessing_spc_dates_2013.txt


#grab the current one that is running from the array param above
this_spc_date_string=${SPC_DATE_STRINGS[$SLURM_ARRAY_TASK_ID]}

#print some things to the .out for debuging if it fails.
echo "cliping storms to inside CONUS:"
echo "Array index = ${SLURM_ARRAY_TASK_ID} ... SPC date = ${this_spc_date_string}"

#source  my python envs
source ~/.bashrc
bash 

conda activate gewitter 

python -u ./remove_storms_outside_conus.py \
--input_tracking_dir_name="/ourdisk/hpc/ai2es/tornado/final_tracking_V2/" \
--first_spc_date_string=${this_spc_date_string} \
--last_spc_date_string=${this_spc_date_string} \
--max_link_distance_metres=30000 \
--max_lead_time_sec=3600 \
--output_tracking_dir_name="/ourdisk/hpc/ai2es/tornado/conus_only_V2/"
