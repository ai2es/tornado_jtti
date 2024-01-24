#!/bin/bash
#SBATCH -p normal  
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH -n 1
#SBATCH --mem-per-cpu=10GB
#SBATCH --job-name=5555_mean_alldates 
#SBATCH --mail-user=monique.shotande@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/momoshog/Tornado/tornado_jtti/
#SBATCH --output=/home/momoshog/Tornado/slurm_out/tornado_jtti/%x_%j_out.txt
#SBATCH --error=/home/momoshog/Tornado/slurm_out/tornado_jtti/%x_%j_err.txt


##########################################################
# Source conda env
source ~/.bashrc
bash
conda activate tf_experiments #tf_tornado #

python --version

TUNER="tor_unet_sample50_50_classweights50_50_hyper"
DIR_PREDS="/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_preds1/$TUNER"

YEAR='2019'

python wofs_evaluations/wofs_preds_evaluations.py  --dir_wofs_preds="$DIR_PREDS"  --loc_storm_report="/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/tornado_reports/tornado_reports_${YEAR}_spring.csv"  --out_dir="/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_figs/$YEAR/summary/$TUNER"  --year "$YEAR" --date0 '2019-04-28'  --date1 '2019-06-03'  --thres_dist 50  --thres_time 20  --stat 'mean'  --uh_compute  --nthresholds 51  --model_name="$TUNER"  --write 1  #--dry_run

# --forecast_duration  --ml_probabs_dilation  33
#/ourdisk/hpc/ai2es/tornado/stormreports/processed/tornado_reports_2019.csv
#--dir_wofs_preds='/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_preds'
#--dir_wofs_preds='/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_preds1/tor_unet_sample50_50_classweights20_80_hyper' \
#--dir_wofs_preds='/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_preds1/tor_unet_sample50_50_classweights50_50_hyper' \
#--dir_wofs_preds='/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_preds1/tor_unet_sample90_10_classweights20_80_hyper' \ (TODO)
