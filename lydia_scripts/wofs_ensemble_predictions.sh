#!/bin/bash
#SBATCH -p ai2es 
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH -n 15
#SBATCH --mem-per-cpu=10000
#SBATCH --job-name=EN_UH80_i1900-2000_d20190430
#SBATCH --mail-user=monique.shotande@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/momoshog/Tornado/tornado_jtti/
#SBATCH --output=/home/momoshog/Tornado/slurm_out/tornado_jtti/%x_%j_out.txt
#SBATCH --error=/home/momoshog/Tornado/slurm_out/tornado_jtti/%x_%j_err.txt
##SBATCH --array=0-17%6

##########################################################

# Source conda env
source ~/.bashrc
bash
conda activate tf_experiments #tf_tornado #

python --version


# Select WoFS file
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"

#>YEAR="2023"
#>DATE="${YEAR}0602"  #20190430 #20230602
#>ROOT="/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_preds/${YEAR}"  

#>echo "ENS ${ROOT}/${DATE}"


# Execute predictions
python lydia_scripts/wofs_ensemble_predictions.py #\
#--loc_date="${ROOT}/${DATE}"  \
#--loc_files_list="/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_preds/2023/20230602/wofs_preds_files.csv"
#--inits="1900"  \
#--for_ens \ 
#--out_dir="${ROOT}/${DATE}" \ #/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_figs/2023/${DATE}
#--write=0

