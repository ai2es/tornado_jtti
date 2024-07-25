#!/bin/bash
#SBATCH -p debug
#SBATCH --nodes=1
#SBATCH -n 1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --chdir=/home/ggantos/tornado_jtti/
#SBATCH --job-name="hgt_debug"
#SBATCH --mail-user=ggantos@ucar.edu
#SBATCH --mail-type=ALL
#SBATCH --mail-type=END
#SBATCH --output=/home/ggantos/slurmouts/R-%x.%j.out
#SBATCH --error=/home/ggantos/slurmouts/R-%x.%j.err

module load Mamba
bash 
conda activate tf_tornado 
 
#QC and convert the sparse gridrad data to grids 
python -u gpheight.py