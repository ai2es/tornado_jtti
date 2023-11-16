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



UPDATES REGARDING WOFS PREDICTIONS:   
To generate WoFS predictions using an ML model with or without a calibration model, use `wofs_raw_predictions.py`.   
Example use case with command line arguments is in `wofs_raw_predictions.sh`.   
Additional WoFS fields are extracted along with the predictions and can be selected via the command line:   
`U V W WSPD10MAX W_UP_MAX P PB PH PHB HGT`. (NOTE: `P PB PH PHB HGT` are REQUIRED for WoFS interpolation to the GridRad grid).   
Current top model: /ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/unet/ZH_only/tuning/tor_unet_sample50_50_classweightsNone_hyper/2023_07_20_20_55_39_hp_model00.h5   
Current calibration model (python pickle files): /ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/gridrad/preds/calibraion_model_iso_v00.pkl

