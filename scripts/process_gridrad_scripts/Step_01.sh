#!/bin/bash
#SBATCH -p normal
#SBATCH --nodes=1
#SBATCH -n 4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --chdir=/ourdisk/hpc/ai2es/randychase/GewitterGefahr/gewittergefahr/scripts/
#SBATCH --job-name="Step1_2013"
#SBATCH --mail-user=randychase@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --mail-type=END
#SBATCH --output=/home/randychase/slurmouts/R-%x.%j.out
#SBATCH --error=/home/randychase/slurmouts/R-%x.%j.err
#SBATCH --array=2%4

#read filedirs into array, check this file to see how many jobs need to be in the array SLURM  
mapfile -t SPC_DATE_STRINGS < /ourdisk/hpc/ai2es/tornado/tornado_jtti/scripts/process_gridrad_scripts/orderofprocessing_files/orderofprocessing_paths_2013.txt

#grab the current one that is running from the array param above
this_spc_date_string=${SPC_DATE_STRINGS[$SLURM_ARRAY_TASK_ID]}

#print to the .out file for transparency 
echo "Array index = ${SLURM_ARRAY_TASK_ID} ... SPC date = ${this_spc_date_string}"

#source  my python envs
source ~/.bashrc
bash 

#activate gewitter 
conda activate gewitter 
 
#QC and convert the sparse gridrad data to grids 
python -u gridrad_from_sparse_to_grid.py \
--input_directory_name=${this_spc_date_string} \
--output_directory_name="/ourdisk/hpc/ai2es/tornado/gridrad_gridded_V2" 