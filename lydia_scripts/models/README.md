# Description of Files 
Update by: Monique Shotande

As of July 2023 there are 3 newer U-Net models found from hyperparameter 
searches using Keras Hyperband Tuner. These models are dated according to when 
the hyperparameter search was performed. Additional, slightly later models are 
found on schooner under 
`/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/unet/ZH_only/tuning` along 
with corresponding results csv files and performance figures on the GridRad data.   
These models were created using the python script 
`lydia_scripts/scripts_tensorboard/unet_hypermodel.py` and batch script 
`lydia_scripts/scripts_tensorboard/unet_hypermodel.sh`

--- 
Author: Randy J. Chase 

Inside this directory are the ML models trained to the the tornado prediction task on WoFS. As of writing this README (Feb 2023), there are 2 models that were found to be 'best' when Lydia Spychalla did a simple hyperparameter search. 

They should be h5 models, so you can change the load function in Lydia's code to 

``` model = tf.keras.models.load_model('Path2Model',compile=False) ```

the ``compile = False`` is important because of the custom metrics used in training the networks (maxCSI). 

The plan is to redo model training on more data. The model should be able to be loaded in the same manner. 


