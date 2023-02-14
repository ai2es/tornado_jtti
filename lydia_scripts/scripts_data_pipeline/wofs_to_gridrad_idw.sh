#!/bin/bash
#SBATCH -p normal
#SBATCH --time=5:00:00
##SBATCH -p debug
##SBATCH --time=00:30:00
#SBATCH --nodes=1
##SBATCH -n 20
#SBATCH --mem-per-cpu=1000
#SBATCH --job-name=wofs_to_gridrad
#SBATCH --mail-user=monique.shotande@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/momoshog/Tornado/tornado_jtti/
#SBATCH --output=/home/momoshog/Tornado/slurm_out/tornado_jtti/res__%x_%j.out
#SBATCH --error=/home/momoshog/Tornado/slurm_out/tornado_jtti/res__%x_%j.err
#SBATCH --array=0-36%6
##SBATCH --array=0-500%15

##########################################################

# Source Lydia's base conda env
#source /home/lydiaks2/.bashrc
source ~/.bashrc
bash
#conda activate tornado

python --version
echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"

#602 total days
#python /home/momoshog/Tornado/tornado_jtti/lydia_scripts/scripts_data_pipeline/wofs_to_gridrad_idw.py \
python lydia_scripts/scripts_data_pipeline/wofs_to_gridrad_idw.py --path_to_raw_wofs="/ourdisk/hpc/ai2es/wofs/2019/*"  --output_patches_path="/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_patched_4/size_32/"  --array_index=$SLURM_ARRAY_TASK_ID  --patch_size=32  --with_nans=1  --dry_run