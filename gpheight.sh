#!/usr/bin/bash -l
#SBATCH -p normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=64G
#SBATCH --time=23:00:00
#SBATCH --chdir=/home/ggantos/tornado_jtti/
#SBATCH --job-name="hgt_0510"
#SBATCH --mail-user=ggantos@ucar.edu
#SBATCH --mail-type=ALL
#SBATCH --mail-type=END
#SBATCH --output=/home/ggantos/slurmouts/R-%x.%j.out
#SBATCH --error=/home/ggantos/slurmouts/R-%x.%j.err

python -u gpheight.py  \
--preds_dir="/ourdisk/hpc/ai2es/tornado/wofs-preds-2023/"  \
--init_dir="/ourdisk/hpc/ai2es/tornado/wofs-preds-2023-init/2023/"  \
--date="20230510"