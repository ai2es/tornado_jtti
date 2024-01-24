#!/bin/bash
#SBATCH -p normal 
#SBATCH --time=13:00:00
#SBATCH --nodes=1
#SBATCH -n 1
#SBATCH --mem-per-cpu=2000 #MB
#SBATCH --job-name=9128_190429 #
#SBATCH --mail-user=monique.shotande@ou.edu
#SBATCH --mail-type=FAIL,INVALID_DEPEND,REQUEUE,STAGE_OUT  #ALL
#SBATCH --chdir=/home/momoshog/Tornado/tornado_jtti/
#SBATCH --output=/home/momoshog/Tornado/slurm_out/tornado_jtti/%x_J%j__I%a_N%A_out.txt
#SBATCH --error=/home/momoshog/Tornado/slurm_out/tornado_jtti/%x_J%j__I%a_N%A_err.txt
#SBATCH --open-mode=append
#SBATCH --array=1-76%20 #20-76%6  #0-36%5
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
YEAR="2019" 
DATE="${YEAR}0429"
#20230602 20230615 20230629

#20180413  20180502  20180509  20180514  20180518  20180524      
#20180429  20180503  20180510  20180515  20180519  20180525      
#20180430  20180504  20180511  20180516  20180521  20180527      
#20180501  20180507  20180512  20180517  20180523 

#20190430  20190502  20190503  20190506  20190507  
#20190508  20190509  20190510  20190513  20190514  
#20190515  20190516  20190517  20190518  20190520  
#20190521  20190522  20190523  20190524  20190525  
#20190526  20190528  20190529  20190530
 


#WOFS_ROOT="/ourdisk/hpc/ai2es/tornado/wofs-preds-2023" 
#WOFS_REL_PATH="20230602/${INIT_TIME}/ENS_MEM_14"  #20190430/1830  #20190430/1900   
WOFS_ROOT="/ourdisk/hpc/ai2es/wofs/${YEAR}"

INIT_TIMES=($(ls $WOFS_ROOT/$DATE))
echo "itimes:: ${INIT_TIMES[@]}"


#TUNER="tor_unet_sample50_50_classweights20_80_hyper"
#MODEL="2023_07_21_08_33_57_hp_model00"
#TUNER="tor_unet_sample50_50_classweights50_50_hyper"
#MODEL="2023_07_21_08_33_16_hp_model00"
TUNER="tor_unet_sample90_10_classweights20_80_hyper"
MODEL="2023_07_20_09_32_38_hp_model00"
DIR_PREDS="/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_preds1/$TUNER"

#TUNER="tor_unet_sample50_50_classweightsNone_hyper"
#MODEL="2023_07_20_20_55_39_hp_model00"
#DIR_PREDS="/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_preds"

for INIT_TIME in ${INIT_TIMES[@]}; do

    #DIR_PREDS_INIT_TIME="/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_preds/${YEAR}/${DATE}/${INIT_TIME}"
    DIR_PREDS_INIT_TIME="${DIR_PREDS}/${YEAR}/${DATE}/${INIT_TIME}"
    if [ ! -d  $DIR_PREDS_INIT_TIME ]; then
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
        python lydia_scripts/wofs_raw_predictions.py --loc_wofs="${WOFS_ROOT}/${WOFS_REL_PATH}/${WOFS_FILE}"  --datetime_format="%Y-%m-%d_%H:%M:%S"  --dir_preds="${DIR_PREDS}/${YEAR}/${WOFS_REL_PATH}"  --dir_patches="/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_patches/${YEAR}/${WOFS_REL_PATH}" --dir_figs="/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_figs/${YEAR}/${WOFS_REL_PATH}/${WOFS_FILE}" --with_nans  -Z --loc_model="/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/unet/ZH_only/tuning/${TUNER}/${MODEL}.h5"  --loc_model_calib="/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/gridrad/preds/${TUNER}/${MODEL}_calibraion_model_iso.pkl"  --file_trainset_stats="/ourdisk/hpc/ai2es/tornado/learning_patches/tensorflow/3D_light/training_int_nontor_tor/training_metadata_ZH_only.nc" --write=1  #--dry_run
        #tor_unet_sample50_50_classweightsNone_hyper/2023_07_20_20_55_39_hp_model00.h5
        #calibraion_model_iso_v00.pkl

        EC=$?
        echo "exit code: $EC"
        if [ $EC -eq 0 ]; then
            echo "${SLURM_JOB_NAME}_${SLURM_JOBID}_out.txt"
        fi

        #fields UP_HELI_MAX U V W W_UP_MAX WSPD10MAX REFL_10CM COMPOSITE_REFL_10CM Times XTIME P PH PHB HGT
        #-f U V W WSPD10MAX W_UP_MAX P PB PH PHB HGT
    done # for ens_member
done # for init_time
