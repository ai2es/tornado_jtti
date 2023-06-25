#!/bin/bash

##SBATCH --partition=gpu
#SBATCH -p ai2es #gpu #
##SBATCH --nodelist=c731
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=10  #-n 20
#SBATCH --mem=32G
#SBATCH --job-name=tuner__test_resample
#SBATCH --chdir=/home/momoshog/Tornado/tornado_jtti
#SBATCH --output=/home/momoshog/Tornado/slurm_out/tornado_jtti/%x_%j.out
#SBATCH --error=/home/momoshog/Tornado/slurm_out/tornado_jtti/%x_%j.err
#SBATCH --mail-user=monique.shotande@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --verbose
##SBATCH --profile=Task   #data collection on  (I/O, Memory, ...) data is collected. stored in an HDF5 file for the job

##########################################################

# Source Andy's env
. /home/fagg/tf_setup.sh
conda activate tf
#conda env export --from-history > fagg_env.yml

# Source conda
#source ~/.bashrc
#bash 
# if conda env tornado does not exist
#conda env create --name tornado --file environment.yml
#conda activate tf_experiments #tf_tornado
#conda install -c conda-forge wandb
#conda install -c anaconda -y tensorflow-gpu 

#conda activate tf_tornado 
#conda uninstall protobuf
#>conda install -c conda-forge tensorflow
#conda install -c conda-forge tensorflow-gpu
#conda install -c conda-forge tensorboard
#>conda install -c conda-forge keras-tuner
#conda install -c conda-forge wandb
#conda clean --all -v


python --version
echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK" # # threads job has bee allocated
echo "SLURM_JOB_NUM_NODES=$SLURM_JOB_NUM_NODES" # count of nodes actually allocated
echo "SLURM_CPUS_ON_NODE=$SLURM_CPUS_ON_NODE" # # CPUs allocated
echo "SLURM_JOB_CPUS_PER_NODE=$SLURM_JOB_CPUS_PER_NODE" # # available CPUs to  the job on the allocated nodes. format: CPU_count[(xnumber_of_nodes)][,CPU_count [(xnumber_of_nodes)] ...].
echo "SLURM_GPUS=$SLURM_GPUS"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
#echo "SBATCH_MEM_PER_CPU=$SBATCH_MEM_PER_CPU"
#echo "SBATCH_MEM_PER_GPU=$SBATCH_MEM_PER_GPU"
#echo "SBATCH_MEM_PER_NODE=$SBATCH_MEM_PER_NODE" # Same as --mem
#SBATCH_PROFILE
#SLURM_JOB_NODELIST
#set -x; cat /proc/cpuinfo | grep processor | wc


# Run hyperparameter search
#--in_dir="/ourdisk/hpc/ai2es/tornado/learning_patches/tensorflow/3D_light/validation_int_nontor_tor/validation1_ZH_only.tf" \
#--in_dir_val="/ourdisk/hpc/ai2es/tornado/learning_patches/tensorflow/3D_light/training_int_nontor_tor/training_ZH_only.tf" \
python -u lydia_scripts/scripts_tensorboard/unet_hypermodel.py \
--in_dir="/ourdisk/hpc/ai2es/tornado/learning_patches_V2/tensorflow/3D_light/train_int_nontor_tor/train_ZH_only_2013_2016.tf" \
--in_dir_val="/ourdisk/hpc/ai2es/tornado/learning_patches_V2/tensorflow/3D_light/val_int_nontor_tor/val_ZH_only_2017.tf" \
--in_dir_test="/ourdisk/hpc/ai2es/tornado/learning_patches_V2/tensorflow/3D_light/test_int_nontor_tor/test_ZH_only_2018.tf" \
--out_dir="/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/unet/ZH_only/tuning" \
--out_dir_tuning="/scratch/momoshog/Tornado/tornado_jtti/tuning" \
--epochs=10 \
--batch_size=1024 \
--lrate=1e-3 \
--wandb_tags test resample \
--number_of_summary_trials=3 \
--gpu \
--save=0 \
--overwrite \
--tuner_id="debug_wandb" \
--dry_run \
hyper \
--max_epochs=5 \
--factor=2
#--hyperband_iterations=2
#--hps_index=1 \

#--resample .9 .1 \
#--class_weight .1 .9 \

#--max_epochs=320 \
#--factor=10

#--overwrite \
#--tuner_id="debug_newgridrad" \
#--dry_run \
#--nogo \


#hyperband, 1 iteration max_epochs * (math.log(max_epochs, factor) ** 2) cumulative epochs across all trials
#natural_validation_ZH_only.tf

#--out_dir="/home/momoshog/Tornado/test_data/tmp" \
#tornado_jtti/unet/ZH_only/initialrun_model8/' \
#python -u lydia_scripts/scripts_tensorboard/tuning_JTTI.py \
#--input_dir="/ourdisk/hpc/ai2es/tornado/wofs_patched/size_32/" \
#--output_dir='/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/unet/' \
#--training_data_metadata_path='/ourdisk/hpc/ai2es/tornado/learning_patches/tensorflow/3D_light/training_onehot_tor/training_metadata_ZH_only.nc' \
#--path_to_wofs='/ourdisk/hpc/ai2es/wofs/' \
#--wofs_day_idx=$SLURM_ARRAY_TASK_ID \
#--patch_size=32  \
#--dry_run
