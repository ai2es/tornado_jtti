#!/bin/bash
#SBATCH -p normal
#SBATCH --nodes=1
#SBATCH -n 2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --chdir=/ourdisk/hpc/ai2es/alexnozka/GewitterGefahr/gewittergefahr/scripts
#SBATCH --job-name="S2aC1_2015"
#SBATCH --mail-user=alexander.j.nozka-1@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --mail-type=END
#SBATCH --output=/ourdisk/hpc/ai2es/alexnozka/debug/R-%x.%j.out
#SBATCH --error=/ourdisk/hpc/ai2es/alexnozka/debug/R-%x.%j.err
#SBATCH --array=27,34,38,43,44,45,49,53,54,55,57,58,59,60,64,80%2


#need all yyyymmdd strings  
mapfile -t SPC_DATE_STRINGS < /ourdisk/hpc/ai2es/tornado/tornado_jtti/scripts/process_gridrad_scripts/orderofprocessing_files/orderofprocessing_spc_dates_2013.txt

#grab the current one that is running from the array param above
this_spc_date_string=${SPC_DATE_STRINGS[$SLURM_ARRAY_TASK_ID]}

#source  my python envs
source ~/.bashrc
bash 

conda activate gewitter 

echo "converting...column_max_dbz"
python -u gridrad_to_myrorss_format.py \
--input_gridrad_dir_name="/ourdisk/hpc/ai2es/tornado/gridrad_gridded_V2" \
--output_myrorss_dir_name="/ourdisk/hpc/ai2es/tornado/gridrad_myrorss_V2/" \
--spc_date_string=${this_spc_date_string} \
--output_field_name "reflectivity_column_max_dbz"
