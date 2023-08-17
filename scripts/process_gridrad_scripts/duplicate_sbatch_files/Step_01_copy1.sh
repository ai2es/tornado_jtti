#!/bin/bash
#SBATCH -p debug
#SBATCH --nodes=1
#SBATCH -n 4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --chdir=/ourdisk/hpc/ai2es/alexnozka/GewitterGefahr/gewittergefahr/scripts
#SBATCH --job-name="S1C1_2019"
#SBATCH --mail-user=alexander.j.nozka-1@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --mail-type=END
#SBATCH --output=/ourdisk/hpc/ai2es/alexnozka/debug/R-%x.%j.out
#SBATCH --error=/ourdisk/hpc/ai2es/alexnozka/debug/R-%x.%j.err
#SBATCH --array=113%2

#read filedirs into array, check this file to see how many jobs need to be in the array SLURM  
mapfile -t SPC_DATE_STRINGS < /ourdisk/hpc/ai2es/tornado/tornado_jtti/scripts/process_gridrad_scripts/orderofprocessing_files/orderofprocessing_paths_2019.txt

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
