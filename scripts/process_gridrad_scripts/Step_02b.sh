#!/bin/bash
#SBATCH -p normal
#SBATCH --nodes=1
#SBATCH -n 2
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --chdir=/ourdisk/hpc/ai2es/randychase/GewitterGefahr/gewittergefahr/scripts/
#SBATCH --job-name="Step2b_2013"
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

#source  my python envs
source ~/.bashrc
bash 

conda activate gewitter 

echo "converting...40dBZ_echotop"
python -u gridrad_to_myrorss_format.py \
--input_gridrad_dir_name="/ourdisk/hpc/ai2es/tornado/gridrad_gridded_V2" \
--output_myrorss_dir_name="/ourdisk/hpc/ai2es/tornado/gridrad_myrorss_V2/" \
--spc_date_string=${this_spc_date_string} \
--output_field_name "echo_top_40dbz_km" 