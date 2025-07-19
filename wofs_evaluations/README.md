author: Monique Shotande

The `wofs_evaluations` directory contains scripts for evaluating models on wofs 
data.   

U-Net hyperparameter search code is in `lydia_scripts/scripts_tensorboard` to 
generate models. Specifically, the python script is `unet_hypermodel.py` and the 
executing batch script is `unet_hypermodel.py`. A limited set of unit tests are 
in `unet_hypermodel_test.py`.   

The python script to generate wofs predictions from learned models is 
`wofs_raw_predictions.py` and the batch script is `wofs_raw_predictions.py`. 
To generate wofs predictions for multiple forecast times and ensembles for a 
particular day use batch script `wofs_raw_predictions_array_ens.sh`.   


## UNet Models Directory:
(models learned from different hyperparameter searches)
`/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/unet/ZH_only/tuning`

## Isotonic Regression Models Directory:
(these models and outputs are for recalibrating the original UNet outputs)   
`/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/gridrad/preds`

## UNet model training and hyperparameter search scripts:
`/home/momoshog/Tornado/tornado_jtti/lydia_scripts/scripts_tensorboard/unet_hypermodel.sh`
`/home/momoshog/Tornado/tornado_jtti/lydia_scripts/scripts_tensorboard/unet_hypermodel.py`
`/home/momoshog/Tornado/tornado_jtti/lydia_scripts/scripts_tensorboard/unet_hypermodel_test.py`

## GridRad tuner results output:
(Contains models and some gridad evaluation results from the hyperparameter search)
`/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/unet/ZH_only/tuning`

## GridRad prediction outputs:
(generated from the hyperparameter search)   
`/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/gridrad/preds`

## WoFS prediction outputs:
(output for unet model learned with 50-50 patch sampling and no class weighting)   
`/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_preds/`   
(output for all other unet models)   
`/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_preds1/`

## WoFS evaluation results:
(evaluation results as csv files and figures for the 2019 season for top 4 models)   
`/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_figs/2019/summary/<TUNER>`

## Storm reports used for WoFS evaluations:
`/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/tornado_reports/tornado_reports_2019_spring.csv`
