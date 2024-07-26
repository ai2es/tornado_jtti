#!/bin/bash
#SBATCH -p normal
#SBATCH --nodes=1
#SBATCH -n 1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --chdir=/home/ggantos/tornado_jtti/
#SBATCH --job-name="hgt_debug"
#SBATCH --mail-user=ggantos@ucar.edu
#SBATCH --mail-type=ALL
#SBATCH --mail-type=END
#SBATCH --output=/home/ggantos/slurmouts/R-%x.%j.out
#SBATCH --error=/home/ggantos/slurmouts/R-%x.%j.err

module load Mamba
mamba init
source ~/.bashrc
bash
mamba activate tf_tornado 
 
python -u gpheight.py
