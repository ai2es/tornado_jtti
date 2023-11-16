#!/bin/bash

##SBATCH --partition=gpu
#SBATCH -p ai2es #gpu #
##SBATCH --nodelist=c731
#SBATCH --time=30:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=24  #-n 20
#SBATCH --mem=40G
#SBATCH --job-name=sNone_c10_90
#SBATCH --chdir=/home/momoshog/Tornado/tornado_jtti
#SBATCH --output=/home/momoshog/Tornado/slurm_out/tornado_jtti/%x_%j.out
#SBATCH --error=/home/momoshog/Tornado/slurm_out/tornado_jtti/%x_%j.err
#SBATCH --mail-user=monique.shotande@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --verbose
##SBATCH --profile=Task   #data collection on  (I/O, Memory, ...) data is collected. stored in an HDF5 file for the job  (next:sNone_c )

##########################################################

# Source Andy's env
. /home/fagg/tf_setup.sh
conda activate tf #_2023_01
#conda env export --from-history > fagg_env.yml

# Source conda
#source ~/.bashrc
#bash 
# if conda env tornado does not exist
#conda env create --name tornado --file environment.yml
#conda clean --all -v


python --version
echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK" # # threads job has been allocated
echo "SLURM_NTASK=$SLURM_NTASK" 
echo "SLURM_JOB_NUM_NODES=$SLURM_JOB_NUM_NODES" # count of nodes actually allocated
echo "SLURM_CPUS_ON_NODE=$SLURM_CPUS_ON_NODE" # # CPUs allocated
echo "SLURM_JOB_CPUS_PER_NODE=$SLURM_JOB_CPUS_PER_NODE" # # avail CPUs to job on allocated nodes. format: CPU_count[(xnumber_of_nodes)][,CPU_count [(xnumber_of_nodes)] ...].
echo "SLURM_GPUS=$SLURM_GPUS"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "SBATCH_MEM_PER_CPU=$SBATCH_MEM_PER_CPU"
echo "SBATCH_MEM_PER_GPU=$SBATCH_MEM_PER_GPU"
echo "SBATCH_MEM_PER_NODE=$SBATCH_MEM_PER_NODE" # Same as --mem
#set -x; cat /proc/cpuinfo | grep processor | wc


# Run hyperparameter search
DATA_DIR="/ourdisk/hpc/ai2es/tornado/learning_patches_V2/tensorflow/3D_light"
#--in_dir="/ourdisk/hpc/ai2es/tornado/learning_patches/tensorflow/3D_light/validation_int_nontor_tor/validation1_ZH_only.tf" \
#--in_dir_val="/ourdisk/hpc/ai2es/tornado/learning_patches/tensorflow/3D_light/training_int_nontor_tor/training_ZH_only.tf" \
python -u lydia_scripts/scripts_tensorboard/unet_hypermodel.py \
--in_dir="${DATA_DIR}/train_int_nontor_tor/train_ZH_only.tf" \
--in_dir_val="${DATA_DIR}/val_int_nontor_tor/val_ZH_only.tf" \
--in_dir_test="${DATA_DIR}/test_int_nontor_tor/test_ZH_only.tf" \
--out_dir="/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/unet/ZH_only/tuning" \
--out_dir_tuning="/scratch/momoshog/Tornado/tornado_jtti/tuning" \
--lscratch $LSCRATCH \
--project_name_prefix="tor_unet_sampleNone_classweights10_90" \
--overwrite \
--epochs=100 \
--batch_size=1024 \
--class_weight .1 .9 \
--lrate=1e-3 \
--patience=10 \
--wandb_tags dslarge saveall \
--number_of_summary_trials=3 \
--gpu \
--dry_run \
--save=4 \
hyper \
--max_epochs=400 \
--factor=10 
#--hyperband_iterations=2
#--hps_index=1 \



#--tuner_id="debug" \
#--epochs=10 \

#--wandb_tags test resample repeat \


#--resample .9 .1 \
#--class_weight .1 .9 \

#--max_epochs=320 \
#--factor=10

#--project_name_prefix="" \
#--overwrite \
#--tuner_id="debug_newgridrad" \
#--dry_run \
#--nogo \


#hyperband, 1 iteration max_epochs * (math.log(max_epochs, factor) ** 2) cumulative epochs across all trials
