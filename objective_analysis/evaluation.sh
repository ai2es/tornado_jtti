#!/bin/bash
#SBATCH -p debug
#SBATCH --nodes=1
#SBATCH -n 20
#SBATCH --mem=28G 
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --chdir=/ourdisk/hpc/ai2es/alexnozka/
#SBATCH --job-name="DEval"
#SBATCH --mail-user=alexander.nozka-1@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --mail-type=END
#SBATCH --output=/ourdisk/hpc/ai2es/alexnozka/debug/R-%x.%j.out
#SBATCH --error=/ourdisk/hpc/ai2es/alexnozka/debug/R-%x.%j.err


#source my python env
source /home/anozka/.bashrc
bash 

conda activate wofs

python -u evaluation.py 
