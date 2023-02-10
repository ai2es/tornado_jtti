# Description of Files 
Author: Randy J. Chase 

Inside this directory are the ML models trained to the the tornado prediction task on WoFS. As of writing this README (Feb 2023), there are 2 models that were found to be 'best' when Lydia Spychalla did a simple hyperparameter search. 

They should be h5 models, so you can change the load function in Lydia's code to 

``` model = tf.keras.models.load_model('Path2Model',compile=False) ```

the ``compile = False`` is important because of the custom metrics used in training the networks (maxCSI). 

The plan is to redo model training on more data. The model should be able to be loaded in the same manner. 


