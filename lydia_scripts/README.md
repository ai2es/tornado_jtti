Dataset Name: Tornado JTTI Code Lydia  
Author: Randy J. Chase  
Date Created: 01 Dec 2022  
Email: randychase 'at' ou.edu  

UPDATED: 05 Oct 2023 by Monique Shotande (mo dot shotande `at` gmail)


Description:

The point of the files in this directory are to provide the code for training the machine learning model associated with the NOAA JTTI project. This project is to train an ML model to identify tornadic storms from observed radar data, then apply its learned features into the in numerical weather prediction model data without explicitly resolving the tornadoes. The goal is that the ML product will be better than other tornado proxies in 3km model data (i.e., UH). 

The code in this directory is from Lydia Spychalla, who wrote this code as an undergraduate researcher at OU (while doing her BS at UIUC). Now she is at Penn State working for Matt Kumjian. Please email her at lks5850 'at' psu.edu if you have questions. Note, I (Randy Chase) have not edited the scripts in anyway. So they are still pointing to Lydia's original directory on schooner. Eventually this directory might be deleted. In that case, we have made a backup of her tornado_project dir here: ```/ourdisk/hpc/ai2es/tornado/Lydia_JTTI_Backup2022.zip```

Directory Contents:

```Tornado_Project_Flowchart.pdf```

- This pdf contains the flow through the scripts in this directory to train the model that was used in Summer 2022 (REU; Alex Nozka). It is known that this model has its issues, but this is so we can re-do the pipeline and make the model (hopefully) better. 

```scripts_data_pipeline```

- This directory is for scripts that help make the features and labels for the UNET. Please note, this requires the GridRad processing pipeline that takes a non-trival amount of time to run. This directory also has the wofs processing scripts. 

```scripts_ml_model```

- This directory is for the evaluating trained machine learning models. 

```scripts_tensorboard```

- This directory contains the scripts to do the hyperparameter search. Including a tensorboard script (to monitor training/overfiting)

Required Software: 

[GewitterGefahr](https://github.com/dopplerchase/GewitterGefahr) 

- This code base was orginially developed by Ryan Lagerquist (now at CIRA/CSU) then edited by Randy Chase to run on a GridRad Dataset that was 10x larger than the original GridRad dataset used in [Lagerquist et al. (2020)](https://journals.ametsoc.org/view/journals/mwre/148/7/mwrD190372.xml). This is for creating the training data
 
Tensorflow & [keras_unet_collection](https://github.com/ai2es/keras-unet-collection)

- In order train or run the trained models you will need tensorflow and the keras_unet_collection. Please note the linked github above because we have made some changes to the original repo, like being able to change the kernel size (i.e., the preceptive field)



# UPDATES 
## General Workflow
1. pre-process GridRad data for training, validation, and testing (directory: `scripts_data_pipeline`)   
    a. convert Ryan's storm objects and tracks into arrays and labels for the U-Nets   
    b. generate patches   
    c. convert to tf.Datasets   
    d. generate mean and standard deviation of the training data   
2. run hyperparameter search (scripts: `scripts_tensorboard/unet_hypermodel.py` and `scripts_tensorboard/unet_hypermodel.sh`)
3. create calibration model from GridRad training set predictions (scripts: `scripts_tensorboard/train_calibration_model.py` and `scripts_tensorboard/evaluate_models.sh`)
4. generate WoFS data predictions (script: `wofs_raw_predictions.py`, `wofs_raw_predictions.sh` [for single wofs file], and `wofs_raw_predictions_array_ens.sh` [for multiple wofs files])   
    a. interpolate to GridRad domain   
    b. normalize using GridRad training set mean and standard deviation   
    c. use model to generate predictions   
    d. recalibrate predictions using calibration model   
    e. interpolate predictions to WoFS domain   
5. evaluate WoFS performance (directory: `tornado_jtti/wofs_evaluations`)


## Pre-processing GridRad Data
Dependency: [GewitterGefahr](https://github.com/thunderhoser/GewitterGefahr) by Dr. Ryan Lagerquist   
1) Download repo
2) `cd` to top level GewitterGefahr directory
3) Create conda env   
    In the environment.yml:   
    a) change `python` from `3.8` to `3.9`   
    b) change `numpy` from `=1.18.*` to just `numpy`.   
    Then execute `conda env create -f environment.yml`   
    There is an issue with the pandas.errors module. So it was easiest to just 
    upgrade all the modules to resolve the conflicts.
4) Activate the environment `conda activate gewitter`
5) Install gewittergefahr with `pip install .`
6) You can verify the GewitterGefahr install by running the pytests `pytest`. 
    All tests should pass, but you will see warnings

### PART A
Using scripts by Ryan and modified by Randy in `tornado_jtti/scripts/process_gridrad_scripts`. 
Steps 1 through 6 are the relevant steps. In particular,    
* <ins>Step 1</ins>: /ourdisk/hpc/ai2es/tornado/gridrad_gridded_V2/    
            (old /ourdisk/hpc/ai2es/tornado/gridrad_gridded/)   
* <ins>Step 4</ins>: /ourdisk/hpc/ai2es/tornado/final_tracking_V2/   
            (old /ourdisk/hpc/ai2es/tornado/final_tracking/)   
* <ins>Step 6</ins>: /ourdisk/hpc/ai2es/tornado/linked_V2/   
            (old /ourdisk/hpc/ai2es/tornado/linked/)   

These steps produce the relevant data required for PART B of the pre-processing 
that is performed by Lydia's scripts under `tornado_jtti/lydia_scripts/scripts_data_pipeline`.

### PART B
    1) Convert the storm objects and tracks into arrays and labels
       unet_linking.py
       unet_linking.sh
       input:
            linked_tornado_dir=/ourdisk/hpc/ai2es/tornado/linked_V2/
            tracking_dir=/ourdisk/hpc/ai2es/tornado/final_tracking_V2/
            radar_dir=/ourdisk/hpc/ai2es/tornado/gridrad_gridded_V2/
       output:
            out_labeled_storm_dir=/ourdisk/hpc/ai2es/tornado/labels_unet_V2/   
            (old /ourdisk/hpc/ai2es/tornado/labels_unet/)  
            out_storm_mask_dir=/ourdisk/hpc/ai2es/tornado/storm_mask_unet_V2/   
            (old /ourdisk/hpc/ai2es/tornado/storm_mask_unet/)

    2) patching.py
       patching.sh
       input:
            spc_date_index=$SLURM_ARRAY_TASK_ID
            radar_dir=/ourdisk/hpc/ai2es/tornado/gridrad_gridded_V2/
            storm_mask_dir=/ourdisk/hpc/ai2es/tornado/storm_mask_unet_V2/
       output:
            patch_dir=/ourdisk/hpc/ai2es/tornado/learning_patches_V2/xarray/3D/ 
            (old /ourdisk/hpc/ai2es/tornado/learning_patches/xarray/3D/)

    3) save_light_model_patches.py
       save_light_model_patches.sh
       input:
            /ourdisk/hpc/ai2es/tornado/learning_patches_V2/xarray/3D/size_32/forecast_window_5/
       output:
            /ourdisk/hpc/ai2es/tornado/learning_patches_V2/xarray/3D_light/size_32/forecast_window_5/

    4) Create tf.Dataset files for the the data
       save_tensorflow_datasets_ZH_only.py
       save_tensorflow_datasets_ZH_only.sh
       input:
            /ourdisk/hpc/ai2es/tornado/learning_patches_V2/xarray/3D_light/size_32/forecast_window_5/
            (metadata) /ourdisk/hpc/ai2es/tornado/learning_patches_V2/tensorflow/3D_light/training_int_nontor_tor/training_metadata_ZH_only.nc
       output:
            /ourdisk/hpc/ai2es/tornado/learning_patches_V2/tensorflow/3D_light/


## Running Hyperparameter Search
Run hyperparameter search using GridRad training and validation tf.Datasets. 
Various Keras Tuners are available for use. I (Monique) used Hyperband   

Executing code:   
* Directory: `scripts_tensorboard`   
* Python Script: `scripts_tensorboard/unet_hypermodel.py`   
* Test Script: `scripts_tensorboard/unet_hypermodel_test.py`   
* Batch Script: `scripts_tensorboard/unet_hypermodel.sh`   

For details on command line arguments, execute `unet_hypermodel.py --h`.   

Input GridRad data locations:   
These data sets are created using the scripts in the `scripts_data_pipeline` directory. 
* (training set) `/ourdisk/hpc/ai2es/tornado/learning_patches_V2/tensorflow/3D_light/train_int_nontor_tor/train_ZH_only.tf`
* (validation set) `/ourdisk/hpc/ai2es/tornado/learning_patches_V2/tensorflow/3D_light/val_int_nontor_tor/val_ZH_only.tf`
* (test set) `/ourdisk/hpc/ai2es/tornado/learning_patches_V2/tensorflow/3D_light/test_int_nontor_tor/test_ZH_only.tf`

Hyperparameter search results location:   
Models, training and validation performance figures and results csv files are saved:
* `/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/unet/ZH_only/tuning`


## Create Calibration Model
Train calibration models for a U-Net using the GridRad predictions from the 
training set. Calibration models are learned using isotonic or linear regression.

Executing code:   
* Directory: `scripts_tensorboard`   
* Python Scripts: `scripts_tensorboard/train_calibration_model.py`  
* Batch Scripts: `scripts_tensorboard/evaluate_models.sh`

For details on command line arguments, execute `train_calibration_model.py --h`. 

Input GridRad data locations:
* (Training set) `/ourdisk/hpc/ai2es/tornado/learning_patches_V2/tensorflow/3D_light/train_int_nontor_tor/train_ZH_only.tf`
* (Validation set) `/ourdisk/hpc/ai2es/tornado/learning_patches_V2/tensorflow/3D_light/val_int_nontor_tor/val_ZH_only.tf`
* (Test set) `/ourdisk/hpc/ai2es/tornado/learning_patches_V2/tensorflow/3D_light/test_int_nontor_tor/test_ZH_only.tf`   

UNet models location: `/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/unet/ZH_only/tuning/`

Output GridRad calibration results location: `/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/gridrad/preds/`. Contains calibration model as python pickle file,
and figures for reliability diagram and histograms of the prediction probabilities.


## Generate WoFS Predictions:   
To generate WoFS predictions using an ML model with or without a calibration 
model, use `wofs_raw_predictions.py`. 

Executing code:   
* Directory: `lydia_scripts`   
* Python Script: `lydia_scripts/wofs_raw_predictions.py`   
* Batch Scripts: 
    - (for single WoFS prediction file) `lydia_scripts/wofs_raw_predictions.sh` 
    - (for multiple WoFS prediction files) `lydia_scripts/wofs_raw_predictions_array_ens.sh` 

For details on command line arguments, execute `wofs_raw_predictions.py --h`.  
Additional WoFS data fields can be extracted along with the WoFS predictions and 
can be selected via a space delimited list in the command line. Common fields:    
* U 
* V 
* W
* WSPD10MAX 
* W_UP_MAX 
* P 
* PB 
* PH 
* PHB
* HGT   

(NOTE: `P PB PH PHB HGT` are REQUIRED for WoFS interpolation to the GridRad grid).   

The current top models can be found under: 
* `/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/unet/ZH_only/tuning/`
    - `tor_unet_sample50_50_classweightsNone_hyper/`
    - `tor_unet_sample50_50_classweights20_80_hyper/`
    - `tor_unet_sample90_10_classweights20_80_hyper/`
    - `tor_unet_sample50_50_classweights50_50_hyper/`

Input raw WoFS data location: 
* `/ourdisk/hpc/ai2es/wofs`   
* Command line argument is `--loc_wofs`

Input models location: 
* `/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/unet/ZH_only/tuning/`
* Command line argument is `--loc_model`   

Input calibration models location: 
* `/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/gridrad/preds/`
* Command line argument is `--loc_model_calib`   

Input GridRad normalization stats (training mean and STD): 
* `/ourdisk/hpc/ai2es/tornado/learning_patches/tensorflow/3D_light/training_int_nontor_tor/training_metadata_ZH_only.nc`
* Command line argument is `--file_trainset_stats`.   

Output WoFS prediction locations:
* (50/50 Sample model predictions) `/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_preds/`
* (all other models) `/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_preds1/`

Example model: `/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/unet/ZH_only/tuning/tor_unet_sample50_50_classweightsNone_hyper/2023_07_20_20_55_39_hp_model00.h5`   
Corresponding calibration model (as a python pickle file): `/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/gridrad/preds/calibraion_model_iso_v00.pkl`


## Evaluate WoFS Performance
Evaluate performance of the models from the WoFS predictions. Can specify time 
windows and neighborhood radii.   

Executing code:   
* Directory: `wofs_evaluations`   
* Python Scripts: 
    - (generate csv files with confusion matrix) `wofs_evaluations/wofs_raw_predictions.py`  
    - (generate figures comparing multiple models) `wofs_evaluations/wofs_performance_plots.py`   
* Batch Scripts: `wofs_evaluations/wofs_raw_predictions.sh`

For details on command line arguments, execute `wofs_raw_predictions.py --h`.  

Input WoFS predictions:
* `/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_preds/`    
    OR   
    `/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_preds1/`
* Command line arg `--dir_wofs_preds`  

Input storm reports used: 
* `/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/tornado_reports/tornado_reports_${YEAR}_spring.csv`
* Command line arg `--loc_storm_report` 

Output performance results:
* `/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_figs/2019/summary/`
* Command line arg `--out_dir`
