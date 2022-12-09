#!/bin/bash
#SBATCH -p normal
#SBATCH --nodes=1
#SBATCH -n 20
#SBATCH --mem-per-cpu=1000
#SBATCH --time=24:00:00
#SBATCH --job-name="wofs_to_gridrad"
#SBATCH --mail-user=lydiaks2@illinois.edu
#SBATCH --mail-type=ALL
#SBATCH --mail-type=END
#SBATCH --output=/home/lydiaks2/tornado_project/output/R-%x.%j.out
#SBATCH --error=/home/lydiaks2/tornado_project/output/R-%x.%j.err
#SBATCH --array=0-500%15

#602 total days

python /home/lydiaks2/tornado_project/scripts_data_pipeline/wofs_to_gridrad_idw.py \
--path_to_raw_wofs="/ourdisk/hpc/ai2es/wofs/2019/*" \
--output_patches_path="/ourdisk/hpc/ai2es/tornado/wofs_patched_4/size_32/" \
--array_index=$SLURM_ARRAY_TASK_ID \ 
--patch_size=32 \
--with_nans=1