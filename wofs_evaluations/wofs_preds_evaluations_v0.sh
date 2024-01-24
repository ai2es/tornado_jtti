# OLD DEPREICATED
# Works but used for a single day

#!/bin/bash
#SBATCH -p normal  
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH -n 1
#SBATCH --mem-per-cpu=10GB
#SBATCH --job-name=9128_d_mean
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

TUNER="tor_unet_sample90_10_classweights20_80_hyper"
DIR_PREDS="/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_preds1/$TUNER"


python wofs_evaluations/wofs_preds_evaluations.py  --dir_wofs_preds="$DIR_PREDS"  --loc_storm_report0='/ourdisk/hpc/ai2es/alexnozka/tornado_reports/190430_rpts_filtered_torn.csv'  --loc_storm_report1='/ourdisk/hpc/ai2es/alexnozka/tornado_reports/190501_rpts_filtered_torn.csv'  --out_dir="/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_figs/2019/20190430/summary/$TUNER"  --date0 2019 4 30  --date1 2019 5 1  --thres_dist 50  --thres_time 20  --stat 'mean'  --ml_probabs_dilation  33  --nthresholds 51  --model_name="$TUNER"  --write 1

#  --uh_compute
#--dir_wofs_preds='/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_preds'
#--dir_wofs_preds='/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_preds1/tor_unet_sample50_50_classweights20_80_hyper' \
#--dir_wofs_preds='/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_preds1/tor_unet_sample50_50_classweights50_50_hyper' \
#--dir_wofs_preds='/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_preds1/tor_unet_sample90_10_classweights20_80_hyper' \
