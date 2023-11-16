#!/bin/bash
#SBATCH -p ai2es_v100
#SBATCH --nodes=1
#SBATCH -n 2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --chdir=/ourdisk/hpc/ai2es/randychase/GewitterGefahr/gewittergefahr/scripts/
#SBATCH --job-name="Step8b_2016"
#SBATCH --mail-user=randychase@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --mail-type=END
#SBATCH --output=/home/randychase/slurmouts/R-%x.%j.out
#SBATCH --error=/home/randychase/slurmouts/R-%x.%j.err

#print some things to the .out for debuging if it fails.
echo "creating drive table for step 9"

#source my python env
source /home/randychase/.bashrc
bash 

conda activate gewitter 

#need a fancier way to split up the work. right now, the peak convective time of day overwelms one worker. 
python /ourdisk/hpc/ai2es/randychase/GewitterGefahr/gewittergefahr/scripts/determine_drive_table.py \
--spc_date_list_path='/home/randychase/orderofprocessing_spc_dates_2016.txt' \
--split_path='/ourdisk/hpc/ai2es/tornado/splits/' \
--drive_table_path='/ourdisk/hpc/ai2es/tornado/'