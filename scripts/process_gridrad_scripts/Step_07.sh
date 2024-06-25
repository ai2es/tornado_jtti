#!/bin/bash
#SBATCH -p normal
#SBATCH --nodes=1
#SBATCH -n 2
#SBATCH --mem=28G
#SBATCH --time=05:00:00
#SBATCH --chdir=/ourdisk/hpc/ai2es/randychase/GewitterGefahr/gewittergefahr/scripts/
#SBATCH --job-name="Step7_May202019"
#SBATCH --mail-user=randychase@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --mail-type=END
#SBATCH --output=/home/randychase/slurmouts/R-%x.%j.out
#SBATCH --error=/home/randychase/slurmouts/R-%x.%j.err
#SBATCH --array=32%1

#need all yyyymmdd strings  
mapfile -t SPC_DATE_STRINGS < /home/randychase/orderofprocessing_spc_dates_2019.txt

#grab the current one that is running from the array param above
this_spc_date_string=${SPC_DATE_STRINGS[$SLURM_ARRAY_TASK_ID]}

#print some things to the .out for debuging if it fails.
echo "running target values:"
echo "Array index = ${SLURM_ARRAY_TASK_ID} ... SPC date = ${this_spc_date_string}"

#source  my python envs
source ~/.bashrc
bash 

conda activate gewitter 

python -u ./compute_target_values.py \
--input_linkage_dir_name="/ourdisk/hpc/ai2es/tornado/linked_V2/" \
--first_spc_date_string=${this_spc_date_string} \
--last_spc_date_string=${this_spc_date_string} \
--min_lead_times_sec 0 900 1800 2700 0 \
--max_lead_times_sec 900 1800 2700 3600 3600 \
--min_link_distances_metres 0 0 0 \
--max_link_distances_metres 30000 20000 10000 \
--event_type_string="tornado" \
--output_dir_name="/ourdisk/hpc/ai2es/tornado/labels_V2/"