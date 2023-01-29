#!/bin/bash
#SBATCH -p normal
#SBATCH --nodes=1
#SBATCH -n 2
#SBATCH --mem=28G
#SBATCH --time=08:00:00
#SBATCH --chdir=/ourdisk/hpc/ai2es/randychase/GewitterGefahr/gewittergefahr/scripts/
#SBATCH --job-name="Step2c_2013"
#SBATCH --mail-user=randychase@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --mail-type=END
#SBATCH --output=/home/randychase/slurmouts/R-%x.%j.out
#SBATCH --error=/home/randychase/slurmouts/R-%x.%j.err
#SBATCH --array=32%1

#need all yyyymmdd strings  
mapfile -t SPC_DATE_STRINGS < /home/randychase/orderofprocessing_spc_dates_2013.txt

#grab the current one that is running from the array param above
this_spc_date_string=${SPC_DATE_STRINGS[$SLURM_ARRAY_TASK_ID]}

#print some things to the .out for debuging if it fails.
echo "running echo classification for:"
echo "Array index = ${SLURM_ARRAY_TASK_ID} ... SPC date = ${this_spc_date_string}"


#source  my python envs
source ~/.bashrc
bash 

conda activate gewitter 

python /ourdisk/hpc/ai2es/randychase/GewitterGefahr/gewittergefahr/scripts/run_echo_classification.py \
--radar_source_name="gridrad" \
--spc_date_string=${this_spc_date_string} \
--input_radar_dir_name="/ourdisk/hpc/ai2es/tornado/gridrad_gridded_V2" \
--output_dir_name="/ourdisk/hpc/ai2es/tornado/echo_class_V2/"