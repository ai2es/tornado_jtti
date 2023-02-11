#!/bin/bash
#SBATCH -p ai2es_v100
#SBATCH --nodes=1
#SBATCH -n 2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --chdir=/ourdisk/hpc/ai2es/randychase/GewitterGefahr/gewittergefahr/scripts/
#SBATCH --job-name="Step8a_2016"
#SBATCH --mail-user=randychase@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --mail-type=END
#SBATCH --output=/home/randychase/slurmouts/R-%x.%j.out
#SBATCH --error=/home/randychase/slurmouts/R-%x.%j.err
#SBATCH --array=0-100%2

#need all yyyymmdd strings  
mapfile -t SPC_DATE_STRINGS < /home/randychase/orderofprocessing_spc_dates_2016.txt

#grab the current one that is running from the array param above
this_spc_date_string=${SPC_DATE_STRINGS[$SLURM_ARRAY_TASK_ID]}

#print some things to the .out for debuging if it fails.
echo "determining worker splits for:"
echo "Array index = ${SLURM_ARRAY_TASK_ID} ... SPC date = ${this_spc_date_string}"

#source my python env
source /home/randychase/.bashrc
bash 

conda activate gewitter 

#need a fancier way to split up the work. right now, the peak convective time of day overwelms one worker. 
python /ourdisk/hpc/ai2es/randychase/GewitterGefahr/gewittergefahr/scripts/determine_splits.py \
--spc_date_string=${this_spc_date_string} \
--input_directory_name='/ourdisk/hpc/ai2es/tornado/conus_only/' \
--n_splits=75