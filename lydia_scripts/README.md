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
    a. generate patches
    b. ...
2. generate mean and standard deviation of the training data (directory: `scripts_data_pipeline`)
3. run hyperparameter search (scripts: `scripts_tensorboard/unet_hypermodel.py` and `scripts_tensorboard/unet_hypermodel.sh`)
4. create calibration model from GridRad training set predictions (scripts: `scripts_tensorboard/train_calibration_model.py` and `scripts_tensorboard/...sh`)
5. generate WoFS data predictions (script: `wofs_raw_predictions.py`, `wofs_raw_predictions.sh` [for single wofs file], and `wofs_raw_predictions_array_ens.sh` [for multiple wofs files])
    a. interpolate to GridRad domain
    b. normalize using GridRad training set mean and standard deviation
    c. use model to generate predictions
    d. recalibrate predictions using calibration model
    e. interpolate predictions to WoFS domain
6. evaluate WoFS performance (directory: `tornado_jtti/wofs_evaluations`)

## Running Hyperparameter Search
Run hyperparameter search using GridRad training and validation data sets. 
Various Keras Tuners are available for use. I (Monique) used Hyperband   

Executing code:   
* Directory: `scripts_tensorboard`   
* Python Script: `scripts_tensorboard/unet_hypermodel.py`   
* Test Script: `scripts_tensorboard/unet_hypermodel_test.py`   
* Batch Script: `scripts_tensorboard/unet_hypermodel.sh`   

For details on command line arguments, execute `unet_hypermodel.py -h`.   

Input GridRad data locations:   
These data sets are created using the scripts in the `scripts_data_pipeline` directory. 
* (training set) `/ourdisk/hpc/ai2es/tornado/learning_patches_V2/tensorflow/3D_light/train_int_nontor_tor/train_ZH_only.tf`
* (validation set) `/ourdisk/hpc/ai2es/tornado/learning_patches_V2/tensorflow/3D_light/val_int_nontor_tor/val_ZH_only.tf`
* (test set) `/ourdisk/hpc/ai2es/tornado/learning_patches_V2/tensorflow/3D_light/test_int_nontor_tor/test_ZH_only.tf`

Hyperparameter search results location:   
Models, training and validation performance figures and results csv files are saved.
* `/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/unet/ZH_only/tuning`


## Create Calibration Model
Train calibration models for a U-Net using the GridRad predictions from the 
training set. Calibration models are learning using isotonic or linear regression.

Executing code:   
* Directory: `scripts_tensorboard`   
* Python Scripts: `scripts_tensorboard/train_calibration_model.py`  
* Batch Scripts: `scripts_tensorboard/evaluate_models.sh`

For details on command line arguments, execute `train_calibration_model.py -h`. 

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

For details on command line arguments, execute `wofs_raw_predictions.py -h`.  
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

Example model: `/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/unet/ZH_only/tuning/tor_unet_sample50_50_classweightsNone_hyper/2023_07_20_20_55_39_hp_model00.h5`   
Corresponding calibration model (as a python pickle file): `/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/gridrad/preds/calibraion_model_iso_v00.pkl`

## Evaluate WoFS Performance
Evaluate performance of the models from the WoFS predictions. 

Executing code:   
* Directory: `wofs_evaluations`   
* Python Scripts: 
    - (generate csv files with confusion matrix) `wofs_evaluations/wofs_raw_predictions.py`  
    - (generate figures comparing multiple models) `wofs_evaluations/wofs_performance_plots.py`   
* Batch Scripts: `wofs_evaluations/wofs_raw_predictions.sh`

For details on command line arguments, execute `wofs_raw_predictions.py -h`.  
