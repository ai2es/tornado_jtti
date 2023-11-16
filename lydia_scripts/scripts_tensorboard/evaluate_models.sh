#!/bin/bash

#SBATCH -p ai2es
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20 #-c 24
##SBATCH --ntasks=24  #-n 20
##SBATCH --mem=40G
#SBATCH --job-name=calib9010_2080
#SBATCH --chdir=/home/momoshog/Tornado/tornado_jtti
#SBATCH --output=/home/momoshog/Tornado/slurm_out/tornado_jtti/%x_%j_out.txt
#SBATCH --error=/home/momoshog/Tornado/slurm_out/tornado_jtti/%x_%j_err.txt
#SBATCH --mail-user=monique.shotande@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --verbose


##########################################################
# Source Andy's env
#. /home/fagg/tf_setup.sh
#conda activate tf #_2023_01
#conda env export --from-history > fagg_env.yml

# Source conda
source ~/.bashrc
bash 
conda activate tf_experiments
#conda install -c conda-forge netcdf4 -y


python --version
echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK" # # threads job has been allocated
echo "SLURM_NTASK=$SLURM_NTASK" 
echo "SLURM_GPUS=$SLURM_GPUS"
echo "SBATCH_MEM_PER_CPU=$SBATCH_MEM_PER_CPU"
echo "SBATCH_MEM_PER_GPU=$SBATCH_MEM_PER_GPU"
echo "SBATCH_MEM_PER_NODE=$SBATCH_MEM_PER_NODE" # Same as --mem

TUNER="tor_unet_sample90_10_classweights20_80_hyper"
MODEL="2023_07_20_09_32_38_hp_model00"

#@lydia_scripts/scripts_tensorboard/evaluate_models.txt \
#python lydia_scripts/scripts_tensorboard/evaluate_models.py \
python lydia_scripts/scripts_tensorboard/train_calibration_model.py \
--loc_model=/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/unet/ZH_only/tuning/$TUNER/$MODEL.h5 \
--in_dir=/ourdisk/hpc/ai2es/tornado/learning_patches_V2/tensorflow/3D_light/train_int_nontor_tor/train_ZH_only.tf \
--in_dir_val=/ourdisk/hpc/ai2es/tornado/learning_patches_V2/tensorflow/3D_light/val_int_nontor_tor/val_ZH_only.tf \
--in_dir_test=/ourdisk/hpc/ai2es/tornado/learning_patches_V2/tensorflow/3D_light/test_int_nontor_tor/test_ZH_only.tf \
--in_dir_labels=/ourdisk/hpc/ai2es/tornado/storm_mask_unet_V2 \
--dir_preds=/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/gridrad/preds/$TUNER \
--lscratch $LSCRATCH \
--batch_size=1024 \
--class_weight .2 .8 \
--epochs=100 \
--fname_prefix="${MODEL}_" \
--method_reg=iso \
--hp_path=/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/unet/ZH_only/tuning/$TUNER/$MODEL.csv \
--hp_idx=0 \
-w 2 \
-d


#--loc_model=/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/unet/ZH_only/tuning/tor_unet_sample50_50_classweightsNone_hyper/2023_07_20_20_55_39_hp_model00.h5 \
