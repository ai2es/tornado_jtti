#!/usr/bin/bash -l
#SBATCH -p normal
#SBATCH --nodes=1
#SBATCH -n 1
#SBATCH --mem=64G
#SBATCH --time=23:00:00
#SBATCH --chdir=/home/ggantos/tornado_jtti/
#SBATCH --job-name="hgt_0510"
#SBATCH --mail-user=ggantos@ucar.edu
#SBATCH --mail-type=ALL
#SBATCH --mail-type=END
#SBATCH --output=/home/ggantos/slurmouts/R-%x.%j.out
#SBATCH --error=/home/ggantos/slurmouts/R-%x.%j.err

python -u gpheight.py
