#!/bin/bash
#SBATCH -p normal 
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH -n 24
#SBATCH --mem-per-cpu=1000 #MB
#SBATCH --job-name=wpE_d19052018
#SBATCH --mail-user=monique.shotande@ou.edu
#SBATCH --mail-type=FAIL,INVALID_DEPEND,REQUEUE,STAGE_OUT  #ALL
#SBATCH --chdir=/home/momoshog/Tornado/tornado_jtti/
#SBATCH --output=/home/momoshog/Tornado/slurm_out/tornado_jtti/%x_J%j__I%a_N%A_out.txt
#SBATCH --error=/home/momoshog/Tornado/slurm_out/tornado_jtti/%x_J%j__I%a_N%A_err.txt
#SBATCH --open-mode=append
#SBATCH --array=0-76%23 #20-76%6  #0-36%5
##SBATCH --array=7-9 #at 35, 40, 35 min #37files

##########################################################

# Source conda env
source ~/.bashrc
bash
conda activate tf_experiments #tf_tornado #

python --version


echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "SLURM_ARRAY_JOB_ID: ${SLURM_ARRAY_JOB_ID}"
echo "SLURM_JOBID,SLURM_JOB_ID: ${SLURM_JOBID}, ${SLURM_JOB_ID}"


# Select WoFS file
YEAR="2018" 
DATE="${YEAR}0519"
#20180413  20180502  20180509  20180514  20180518  20180524      20180528  20180601  20180621  20180625  20180629  20180711  20180717  20180913
#20180429  20180503  20180510  20180515  20180519  20180525      20180529  20180618  20180622  20180626  20180630  20180712  20180718  20180914
#20180430  20180504  20180511  20180516  20180521  20180527      20180530  20180619  20180623  20180627  20180709  20180713  20180719  20181010
#20180501  20180507  20180512  20180517  20180523  20180527_RLT  20180531  20180620  20180624  20180628  20180710  20180716  20180720

#20190430  20190502  20190503  20190506  20190507  
#20190508  20190509  20190510  20190513  20190514  
#20190515  20190516  20190517  20190518  20190520  
#20190521  20190522  20190523  20190524  20190525  
#20190526  20190528  20190529  20190530

#WOFS_ROOT="/ourdisk/hpc/ai2es/tornado/wofs-preds-2023" 
#WOFS_REL_PATH="20230602/${INIT_TIME}/ENS_MEM_14" #1830  #1900   
WOFS_ROOT="/ourdisk/hpc/ai2es/wofs/${YEAR}"
#WOFS_REL_PATH="20190430/${INIT_TIME}/ENS_MEM_1"

INIT_TIMES=($(ls $WOFS_ROOT/$DATE))
echo "itimes:: ${INIT_TIMES[@]}"

#INIT_TIMES=("0000" "0030" "0100" "0130" "0200" "0230"  "0300"  "1900"  "1930"  "2000"  "2030"  "2100"  "2130"  "2200"  "2230"  "2300"  "2330") #"1700"
for INIT_TIME in ${INIT_TIMES[@]}; do

    DIR_PREDS_INIT_TIME="/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_preds/${YEAR}/${DATE}/${INIT_TIME}"
    if [ ! -d  $DIR_PREDS_INIT_TIME ]; then
        #echo "echo mkdir ${DIR_PREDS_INIT_TIME}"
        set -x
        mkdir $DIR_PREDS_INIT_TIME
        ##exit;
        ##echo "SHOULD NOT PRINT"
    fi

    for e in {1..18}; do
        WOFS_REL_PATH="${DATE}/${INIT_TIME}/ENS_MEM_${e}"
        echo "ENS${e} path:: ${WOFS_REL_PATH}"
        
        wofs_files=($(ls $WOFS_ROOT/$WOFS_REL_PATH))
        echo "WOFS DIR ${WOFS_ROOT}/${WOFS_REL_PATH}"
        #echo "WOFS FILES:: ${wofs_files[@]}"
        echo "n files = ${#wofs_files[@]}"

        WOFS_FILE=${wofs_files[$SLURM_ARRAY_TASK_ID]}
        echo "Predicting for [${SLURM_ARRAY_TASK_ID}] ${WOFS_FILE}"
        echo "  ${WOFS_ROOT}/${WOFS_REL_PATH}/${WOFS_FILE}"


        # Execute predictions
        python lydia_scripts/wofs_raw_predictions.py --loc_wofs="${WOFS_ROOT}/${WOFS_REL_PATH}/${WOFS_FILE}"  --datetime_format="%Y-%m-%d_%H:%M:%S"  --dir_preds="/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_preds/${YEAR}/${WOFS_REL_PATH}"  --dir_patches="/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_patches/${YEAR}/${WOFS_REL_PATH}" --dir_figs="/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_figs/${YEAR}/${WOFS_REL_PATH}/${WOFS_FILE}" --with_nans  -Z  -f U V W WSPD10MAX W_UP_MAX P PB PH PHB HGT --loc_model="/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/unet/ZH_only/tuning/tor_unet_sample50_50_classweightsNone_hyper/2023_07_20_20_55_39_hp_model00.h5"  --loc_model_calib="/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/gridrad/preds/calibraion_model_iso_v00.pkl"  --file_trainset_stats="/ourdisk/hpc/ai2es/tornado/learning_patches/tensorflow/3D_light/training_int_nontor_tor/training_metadata_ZH_only.nc" --write=1  #--dry_run

        EC=$?
        echo "exit code: $EC"
        if [ $EC -eq 0 ]; then
            echo "${SLURM_JOB_NAME}_${SLURM_JOBID}_out.txt"
        fi

        #fields UP_HELI_MAX U V W W_UP_MAX WSPD10MAX REFL_10CM COMPOSITE_REFL_10CM Times XTIME P PH PHB HGT
    done # for ens_member
done # for init_time
