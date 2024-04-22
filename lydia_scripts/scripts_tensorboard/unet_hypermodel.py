"""
author: Monique Shotande (monique . shotande a ou . edu)

UNet Hyperparameter Search using keras tuners. 
A keras_tuner.HyperModel subclass is defined as UNetHyperModel
that defines how to build various versions of UNet models

Expected working directory is tornado_jtti/

Models are trained on GridRad data.

Windows tensorboard
python -m tensorboard.main --logdir=[PATH_TO_LOGDIR] [--port=6006]

"""

import wandb
from wandb.keras import WandbMetricsLogger, WandbCallback
#wandb.login()

import os, io, sys, random, shutil
import pickle, copy
import time, datetime
import math
#from absl import app
#from absl import flags
import argparse
import numpy as np
#print("np version", np.__version__)
import pandas as pd
#print("pd version", pd.__version__)
import xarray as xr 
#print("xr version", xr.__version__)
#import scipy
#print("scipy version", scipy.__version__)
#import seaborn as sns
#print(f"seaborn {sns.__version__}")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits import mplot3d
import matplotlib.patheffects as path_effects
import matplotlib
font = {#'family' : 'normal',
        #'weight' : 'bold',
        'size'   : 14}
matplotlib.rc('font', **font)

# Display all pd.DataFrame columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
from sklearn.calibration import calibration_curve

import tensorflow as tf
#print("tensorflow version", tf.__version__)
from tensorflow import keras
#print("keras version", keras.__version__)
import tensorflow.keras.backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint 
from tensorflow.keras.optimizers import Adam #, SGD, RMSprop, Adagrad
from tensorflow.keras.metrics import AUC
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import Model

from keras_tuner import HyperModel, Tuner
from keras_tuner.tuners import RandomSearch, Hyperband, BayesianOptimization
#from tensorboard.plugins.hparams import api as hp
from keras_tuner.engine import tuner as tuner_module
from keras_tuner.engine import oracle as oracle_module
from keras_tuner.engine import tuner_utils

from tensorboard.plugins.hparams import api
from tensorboard.plugins.hparams import api_pb2
from tensorboard.plugins.hparams import summary as hp_summary

#sys.path.append("../")
sys.path.append("lydia_scripts")
from custom_losses import make_fractions_skill_score
from custom_metrics import MaxCriticalSuccessIndex
#sys.path.append("../../../keras-unet-collection")
sys.path.append("../keras-unet-collection")
from keras_unet_collection import models

# Cause any Tensor allocations or operations to be printed
# Display which devices operations and tensors are assigned to
tf.debugging.set_log_device_placement(True)
tf.config.run_functions_eagerly(True)
#tf.data.experimental.enable_debug_mode()
ppolicy = "float32"
precision_policy_map = {"mixed_float16": tf.float16, 
                        "mixed_float32": tf.float32,
                        "float32": tf.float32}
tfdtype = precision_policy_map[ppolicy]
print("Mixed precision", ppolicy, tfdtype)
tf.keras.mixed_precision.set_global_policy(ppolicy)
tf.keras.mixed_precision.global_policy()
print(' ')



class UNetHyperModel(HyperModel):
    """
    Hypermodel for tuning unet architectures
    @param input_shape: tuple for the shape of the input data
    @param n_labels: number of class labels. If None, tune as a hyperparameter in 
                that can either be 1 or 2.
    @param name: prefix of the layers and model. Use keras.models.Model.summary to 
                identify the exact name of each layer. Empty by default
    @param tune_optimizer: Whether to tune the choice for the learning 
                optimizer. False by default and uses Adam
    @param distribution_strategy: Optional instance of tf.distribute.Strategy. 
                Same as that used by the tuner object. Here it is used for any 
                custom metrics or losses such that they are declared under the 
                same scope as the model. See Keras Tuner for details.
    """
    def __init__(self, input_shape, n_labels=None, name='', tune_optimizer=False, 
                 distribution_strategy=None, DB=False):
        '''
        Class constructor
        '''
        #TODO exception handling
        self.input_shape = input_shape
        #TODO exception handling
        self.n_labels = n_labels if not n_labels is None or n_labels > 0 else 1
        self.name = name
        self.tune_optimizer = tune_optimizer
        self.distribution_strategy = distribution_strategy
        self.DB = DB
        super().__init__(name=name) #, tunable=True)

    def save_model(self, filepath, weights=True, model=None, save_traces=True):
        '''
        Save this or some other model
        @params filepath: path to save the model
        @params weights: Save just the model weights when True, otherwise save
                        the whole model.
        @params model: if None, save this model, otherwise save the provided model
        @params save_traces: 

        @return: True if the save occured, False otherwise
        '''
        if model is None:
            print(f"Saving this model to {filepath} (save_weights={weights})")
            if weights: self.save_weights(filepath)
            else: self.save(filepath, save_traces=save_traces)
            return True
        else:
            print(f"Saving the provided model to {filepath} (save_weights={weights})")
            if weights: model.save_weights(filepath)
            else: model.save(filepath, save_traces=save_traces)
            return True
        return False

    def build(self, hp, seed=None):
        """
        Build Keras model with the given hyperparameters.
        @params hp: kareas_tuner.HyperParameters object.
        @params seed: A hashable object to be used as a random seed (e.g., to
            construct dropout layers in the model).

        @return: compiled Keras model.
        """
        if self.DB: print("BUILDING UNET HYPERMODEL")
        # Set our random seed
        rng = None if seed is None else random.Random(seed)

        in_shape = self.input_shape

        num_layers = hp.Int("num_layers", min_value=4, step=1, max_value=6)
        latent_dim = hp.Int("latent_dim", min_value=28, step=2, max_value=256)
        #num = hp.Int("n_conv_down", min_value=3, step=1, max_value=10) # number of layers
        nfilters_per_layer6 = np.around(np.linspace(8, latent_dim, num=num_layers)).astype(int).tolist()
        #nfilters_per_layer = list(range(8, latent_dim, 2))
        #nfilters_per_layer2 = [2**i for i in range(3, int(np.log2(latent_dim)))]
        nfilters_per_layer2 = np.logspace(3, np.log2(latent_dim), num=num_layers, endpoint=True, base=2, dtype=int)
        mask = nfilters_per_layer2 % 2
        mask[-1] = 0
        nfilters_per_layer2[mask] += 1 # make odd layers even

        # List defining number of conv filters per down/up-sampling block
        nfilters_type = hp.Choice("nfilters_type", values=['linear', 'log'])
        #filter_num = hp.Choice("nfilters_per_layer", values=[nfilters_per_layer2, nfilters_per_layer6])
        filter_num = nfilters_per_layer2
        if nfilters_type == 'linear':
            filter_num = nfilters_per_layer6

        kernel_size = hp.Choice("kernel_size", [3, 5, 7])
        
        # Number of convolutional layers per downsampling level
        stack_num_down = hp.Int("n_conv_down", min_value=1, step=1, max_value=5)
        # Configuration of downsampling (encoding) blocks
        pool = hp.Choice("pool_down", values=['False', 'ave', 'max'])
        pool = False if pool == 'False' else pool
        activation = hp.Choice("in_activation", values=['PReLU', 'ELU', 'GELU', 'Snake']) #'ReLU', 'LeakyReLU'

        # Number of conv layers (after concatenation) per upsampling level
        stack_num_up = hp.Int("n_conv_up", min_value=1, step=1, max_value=5)
        # Configuration of upsampling (decoding) blocks
        unpool = hp.Choice("pool_up", values=['False', 'bilinear', 'nearest'])
        unpool = False if unpool == 'False' else unpool

        # Select appropriate output activation based on loss and number of output nodes
        #has_l2 = hp.Boolean("has_l2")
        n_labels = self.n_labels
        if n_labels is None:
            with hp.conditional_scope("n_labels", [None]): 
                n_labels = hp.Int("n_labels", min_value=1, step=1, max_value=2)

        loss = 'binary_focal_crossentropy'
        #MCSI = MaxCriticalSuccessIndex() #, MCSI.tp, MCSI.fp, MCSI.fn
        metrics = [MaxCriticalSuccessIndex(), AUC(num_thresholds=100, curve='ROC', name='auc_roc'),
                    AUC(num_thresholds=100, curve='PR', name='auc_pr')] #"categorical_accuracy"
        if n_labels == 1: 
            #TODO: keras.activations.linear
            output_activation = hp.Choice("out_activation", values=['Sigmoid']) #, 'Snake'
            metrics.append("binary_accuracy")
        else:
            with hp.conditional_scope("n_labels", ['>=2']): 
                #TODO: keras.activations.linear
                output_activation = hp.Choice("out_activation", values=['Softmax']) #, 'Snake', 'Sigmoid'
                loss = hp.Choice("loss", ["binary_focal_crossentropy", "categorical_crossentropy", "fractions_skill_score"])
                #make_fractions_skill_score(3, 2, c=1.0, cutoff=0.5, want_hard_discretization=False)
                # TODO: binary v categorical crossentropy math
                if loss == "binary_focal_crossentropy":
                    metrics.append(["binary_accuracy"])
                else:
                    metrics.append(["categorical_accuracy"])
                    if loss == "fractions_skill_score": metrics.append(["categorical_crossentropy"])
        # Activation function of the output layer
        #output_activation = hp.Choice("out_activation", values=['Sigmoid', 'Softmax', None, 'Snake'])

        # Regularization
        ## all conv layers configured as stacks of "Conv2D-BN-Activation"
        batch_norm = hp.Boolean("has_batch_norm") 
        has_l1 = hp.Boolean("has_l1")
        l1 = None
        if has_l1:
            with hp.conditional_scope("has_l1", [True]): 
                #l2 = keras.regularizers.l1(hp.Choice("l1", values=[1e-3, 1e-4, 1e-5, 1e-6]))
                l1 = hp.Choice("l1", values=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
        has_l2 = hp.Boolean("has_l2")
        l2 = None
        if has_l2:
            with hp.conditional_scope("has_l2", [True]): 
                #l2 = keras.regularizers.l2(hp.Choice("l2", values=[1e-3, 1e-4, 1e-5, 1e-6]))
                l2 = hp.Choice("l2", values=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6])

        # TODO: Dropout??

        # Choose the type of unet
        #att_unet_2d (single regression), r2_unet_2d >=2 layers
        #resunet_a_2d, u2net_2d >= 3
        unet_type = hp.Choice("unet_type", values=['unet_2d', 'unet_plus_2d', 'unet_3plus_2d', 'r2_unet_2d'])

        #TODO: pretrain_weights = hp.Choice("pretrain_weights", values=['None', 'imagenet'])

        if unet_type == 'unet_2d':
            model = models.unet_2d(in_shape, 
                            filter_num, 
                            kernel_size=kernel_size,
                            stack_num_down=stack_num_down, 
                            n_labels=n_labels,
                            stack_num_up=stack_num_up,
                            activation=activation, output_activation=output_activation, 
                            pool=pool, unpool=unpool,
                            l1=l1, l2=l2, weights=None,
                            batch_norm=batch_norm, name='unet')
        elif unet_type == 'unet_plus_2d':
            model = models.unet_plus_2d(in_shape, 
                            filter_num, 
                            kernel_size=kernel_size,
                            stack_num_down=stack_num_down,  
                            n_labels=n_labels,
                            stack_num_up=stack_num_up,
                            activation=activation, output_activation=output_activation, 
                            pool=pool, unpool=unpool, 
                            l1=l1, l2=l2, weights=None, 
                            batch_norm=batch_norm, name='unetplus')
        elif unet_type == 'unet_3plus_2d':
            model = models.unet_3plus_2d(in_shape, 
                            filter_num_down=filter_num, 
                            kernel_size=kernel_size,
                            stack_num_down=stack_num_down, 
                            n_labels=n_labels,
                            stack_num_up=stack_num_up, 
                            activation=activation, output_activation=output_activation, 
                            pool=pool, unpool=unpool,
                            l1=l1, l2=l2, weights=None,
                            batch_norm=batch_norm, name='unet3plus')
        elif unet_type == 'r2_unet_2d':
            recur_num = hp.Int("recur_num", min_value=1, step=1, max_value=2)
            model = models.r2_unet_2d(in_shape, filter_num, #(None, None, 3), [64, 128, 256, 512]
                            n_labels=n_labels,
                            stack_num_down=stack_num_down, 
                            stack_num_up=stack_num_up,
                            activation=activation, 
                            output_activation=output_activation, 
                            batch_norm=batch_norm, recur_num=recur_num, 
                            pool=pool, unpool=unpool, name='r2unet')
        '''
        elif unet_type == 'vnet_2d':
            model = models.vnet_2d((256, 256, 1), filter_num=[16, 32, 64, 128, 256], n_labels=2,
                      res_num_ini=1, res_num_max=3, 
                      activation='PReLU', output_activation='Softmax', 
                      batch_norm=True, pool=False, unpool=False, name='vnet')
        elif unet_type == 'u2net_2d':
            model = models.u2net_2d((128, 128, 3), n_labels=2, 
                        filter_num_down=[64, 128, 256, 512], filter_num_up=[64, 64, 128, 256], 
                        filter_mid_num_down=[32, 32, 64, 128], filter_mid_num_up=[16, 32, 64, 128], 
                        filter_4f_num=[512, 512], filter_4f_mid_num=[256, 256], 
                        activation='ReLU', output_activation=None, 
                        batch_norm=True, pool=False, unpool=False, deep_supervision=True, name='u2net')
        '''

        # Insert BatchNormalization layer after Input layer
        if not batch_norm:
            bn_layer = BatchNormalization()
            #synchronized=self.distribution_strategy, #set and if this layer is used within a tf.distribute strategy
            #TODO: model = insert_batchnorm_after_input(model, bn_layer) #, DB=False)

        # TODO: Class weighting
        #w_class0 = hp.Float("w_class0", min_value=.01, max_value=.99, sampling="log") #sampling=linear, step=.1

        # Optimization
        lr = hp.Choice("learning_rate", values=[1e-3, 1e-4, 1e-5, 1e-6])
        #hp.Float('learning_rate',min_value=1e-4, max_value=1e-2, sampling='LOG', default=1e-3)
        optimizer = Adam(learning_rate=lr) #TODO (MAYBE): if not self.tune_optimizer else hp.Choice("optimizer", values=[Adagrad(learning_rate=lr), SGD(learning_rate=lr), RMSprop(learning_rate=lr)])
        # TODO ?
        # hp.Choice("optimizer", values=[Adagrad(learning_rate=lr),
        #                                    SGD(learning_rate=lr),
        #                                    RMSprop(learning_rate=lr)]

        # Build and return the model
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics, 
                      weighted_metrics=[]) #, run_eagerly=True) #TODO: weighted_metrics
        if self.DB: model.summary()
        return model

class HyperbandWAB(Hyperband):
    """ 
    Custom Tuner for Weights and Biases
    """
    def __init__(self, hypermodel=None, cli_args=None, wandb_path=None, **kwargs): #*tuner_args, 
        '''
        Class constructor
        :param cli_args: command line interface (CLI) arguments
        '''
        # Command line args to dict
        args_dict = copy.deepcopy(vars(cli_args))
        self.tags = args_dict['wandb_tags']
        for k in ['in_dir', 'in_dir_val', 'in_dir_test', 'dry_run', 'nogo', 
                  'overwrite', 'save', 'wandb_tags']: #, 'out_dir', 'out_dir_tuning']: #'project_name_prefix']:
            args_dict.pop(k, None)
        self.cli_args = args_dict
        self.wandb_path = wandb_path
        super().__init__(hypermodel=hypermodel, **kwargs) #*tuner_args, 
  
    def run_trial(self, trial, *args, **kwargs):
        '''
        The overridden `run_trial` function

        Args:
            trial: The trial object that holds information for the
            current trial.
            *args: positional args for
            **kwargs: keyword args for
        '''
        # Not using `ModelCheckpoint` to support MultiObjective.
        # It can only track one of the metrics to save the best model.
        model_checkpoint = tuner_utils.SaveBestEpoch(
            objective=self.oracle.objective,
            filepath=self._get_checkpoint_fname(trial.trial_id),
        )

        # WANDB INITIALIZATION
        # Pass configuration so the runs are tagged with the hyperparams
        # Enables use of the comparison UI widgets in the wandb dashboard off the shelf.
        cargs = copy.deepcopy(self.cli_args)

        #PROJ_NAME_PREFIX = cargs['project_name_prefix']
        #PROJ_NAME = f'{PROJ_NAME_PREFIX}_{cargs["tuner"]}'
        #PROJ_DATE = cargs['cdatetime']
        #tuner_dir = cargs['out_dir_tuning'] if not cargs['out_dir_tuning'] is None  else cargs['out_dir']

        cargs.pop('out_dir')
        cargs.pop('out_dir_tuning')
        config = cargs
        hp = trial.hyperparameters
        config.update(hp.values)

        #tb_path = os.path.join(tuner_dir, f'{PROJ_NAME}_{PROJ_DATE}_tb')
        wandb_path = self.wandb_path #os.path.join(tuner_dir, f'{PROJ_NAME}_{PROJ_DATE}_wandb')
        #wandb_ckpt_path = os.path.join(tuner_dir, f'{PROJ_NAME}_wandb_model_ckpts{PROJ_DATE}')
        if not os.path.exists(wandb_path):
            try:
                os.mkdir(wandb_path)
                print(f"Making dir {wandb_path}")
            except Exception as err:
                print(f"CAUGHT:: {err}")
                wandb_path = tuner_dir

        run = wandb.init(project='unet_hypermodel_run0', config=config, 
                         #sync_tensorboard=True, 
                         dir=wandb_path, tags=self.tags) 
                         #resume='auto', entity='ai2es',  

        original_callbacks = kwargs.pop("callbacks", []) + [WandbMetricsLogger()] 
        #WandbMetricsLogger()] #WandbCallback(save_model=False, compute_flops=True)]

        # From Hyperband superclass
        if "tuner/epochs" in hp.values:
            kwargs["epochs"] = hp.values["tuner/epochs"]
            kwargs["initial_epoch"] = hp.values["tuner/initial_epoch"]
        #histories = super().run_trial(trial, *args, **kwargs)

        # Run the training process multiple times
        histories = []
        for execution in range(self.executions_per_trial):
            copied_kwargs = copy.copy(kwargs)
            callbacks = self._deepcopy_callbacks(original_callbacks)
            self._configure_tensorboard_dir(callbacks, trial, execution)
            callbacks.append(tuner_utils.TunerCallback(self, trial))
            # Only checkpoint the best epoch across all executions
            callbacks.append(model_checkpoint)
            copied_kwargs["callbacks"] = callbacks

            obj_value = self._build_and_fit_model(trial, *args, **copied_kwargs)
            histories.append(obj_value)
            print("obj_value.history", obj_value.history)
            # Log the history for WANDB
            #hist_dict = print({f'tn_{k}': v for k, v in obj_value.history.items()})
            df = pd.DataFrame(obj_value.history)
            df = df.assign(execution_index=[execution] * df.shape[0])
            print("df", df)
            wb_table = wandb.Table(dataframe=df)
            run.log({"history": wb_table})
            # Convert DataFrame to a list of dictionaries to log
            #>_historys = df.to_dict(orient='records')
            #>for _hist in _historys:
            #>    run.log(_hist) #run.log({'epoch_loss':epoch_loss, 'epoch':epoch})

        # Finish the wandb run
        run.finish()
        return histories

def insert_batchnorm_after_input(model, bn_layer, DB=False):
    '''
    Insert a BatchNormalization layer after the input layer of an existing 
    model
    :param model: the existing Keras model
    :param bn_layer: BatchNormalization layer
    '''
    #bn_layer = BatchNormalization(**bnargs)
    layers_dict = {layer.name: layer for layer in model.layers}
    new_layers = {model.layers[0].name: None}
    layer_outputs = []

    x = model.layers[0]
    prior_layer_name = ''
    for i, layer in enumerate(model.layers):
        if DB:
            print(f"[{i}] x={x}; ({type(layer)})")
            print(f"      layer.name={layer.name}; layer={layer}")
            print(f"      layer.input={type(layer.input)}")
            print(f"      layer.input={layer.input}")
            print(f"      layer.output={layer.output}")

        # Insert BN layer after the input layer
        if i == 0:
            x = bn_layer(layer.output)
            new_layers[layer.name] = layer.output
            new_layers[bn_layer.name] = x
            layer_outputs.append(x)
        else: 
            # Handle layers that also take skip connections as input
            if isinstance(layer.input, list): # or 'Concatenate' in str(type(layer)):
                ins = []
                for j, inp in enumerate(layer.input):
                    name = layer.input[0].name.split('/')[0]
                    if DB:
                        print(f"  {j} {name}")
                        print(f"  {j} {new_layers[name]}") #=inp
                        print(f"  inp {inp}")
                    ins.append(new_layers[name])
                if DB:
                    print("  x", x)
                x = layer(ins)
            else: 
                inlayer_name = layer.input.name.split('/')[0]
                if DB:
                    print(f"  e {inlayer_name}")
                    print(f"  e {new_layers[inlayer_name]}")

                if 'Concatenate' in str(type(layer)):
                    x = [x]

                # Use output of BN layer for layers whose input comes from the Input layer
                if 'Input' in inlayer_name: # or name != prior_layer:
                    bn_layer_out = new_layers[bn_layer.name]
                    x = layer(bn_layer_out)
                elif inlayer_name == prior_layer_name:
                    x = layer(x)
                else:
                    inp =  new_layers[inlayer_name]
                    x = layer(inp)
            
            new_layers[layer.name] = x
            layer_outputs.append(x)

        prior_layer_name = layer.name

    new_model = Model(model.layers[0].output, layer_outputs, name=f'{model.name}_bn_after_input')
    new_model.summary()
    for k, v in new_layers.items():
            print(f"{k}: {v}")
    return new_model

def create_tuner(args, wandb_path=None, strategy=None, DB=1, **kwargs):
    '''
    Create the Keras Tuner. Tuner can be instance of RandomSearch, Hyperband, 
    BayeOpt, custom or Tuner (the base Tuner class) when no tuner is selected.
    @param args: the command line args object. See create_argsparser() for
            details about the command line arguments
            Command line arguments relevant to this method:
                x_shape
                n_labels
                objective
                max_trials
                max_retries_per_trial
                max_consecutive_failed_trials
                executions_per_trial
                tuner_id
                overwrite
                out_dir
                project_name_prefix
                tuner
                max_epochs
                factor
                hyperband_iterations
                num_initial_points
                alpha,
                beta
    @param DB: debug flag to print the resulting hyperparam search space

    @return: tuple containing the configured tuner object and the hypermodel
    '''
    hypermodel = UNetHyperModel(input_shape=args.x_shape, n_labels=args.n_labels, 
                                distribution_strategy=strategy, DB=0)
    #weights = dummy_loader(model_old_path)
    #model_new = swin_transformer_model(...)
    #model_new.set_weights(weights)

    tuner_dir = args.out_dir_tuning if not args.out_dir_tuning is None  else args.out_dir
    
    PROJ_NAME_PREFIX = args.project_name_prefix
    PROJ_NAME = f'{PROJ_NAME_PREFIX}_{args.tuner}'   

    tuner_args = {
        'distribution_strategy': strategy, #TODO: for multi-gpu 
        'objective': args.objective, #'val_MaxCriticalSuccessIndex', name of objective to optimize (whether to minimize or maximize is automatically inferred for built-in metrics)
        #'max_retries_per_trial': args.max_retries_per_trial,
        #'max_consecutive_failed_trials': args.max_consecutive_failed_trials,
        'executions_per_trial': args.executions_per_trial, 
        'logger': None, #TODO Optional instance of kerastuner.Logger class for streaming logs for monitoring.
        'tuner_id': args.tuner_id, # Optional string, used as ID of this Tuner.
        'overwrite': args.overwrite, #If False, reload existing project. Otherwise, overwrite project
        'directory': tuner_dir #wandb.run.dir #args.out_dir #TODO
    }
        #'seed': None,
        #'hyperparameters': None,
        #'tune_new_entries': True,
        #'allow_new_entries': True,

    MAX_TRIALS = args.max_trials  

    # Select tuner
    tuner = None
    if args.tuner in ['rand', 'random']:
        tuner = RandomSearch(
            hypermodel,
            max_trials=MAX_TRIALS,
            project_name=PROJ_NAME, #TODO prefix for files saved by this Tuner.
            **tuner_args
        )
    elif args.tuner in ['hyper', 'hyperband']:
        tuner = HyperbandWAB( #Hyperband( #
            hypermodel,
            cli_args=args,
            wandb_path=wandb_path,
            max_epochs=args.max_epochs, #10, #max train epochs per model. recommended slightly higher than expected epochs to convergence 
            factor=args.factor, #3, #int reduction factor for epochs and number of models per bracket
            hyperband_iterations=args.hyperband_iterations, #2, #>=1,  number of times to iterate over full Hyperband algorithm. One iteration will run approximately max_epochs * (math.log(max_epochs, factor) ** 2) cumulative epochs across all trials. set as high a value as is within your resource budget
            project_name=PROJ_NAME,
            **tuner_args
        )
        print("Hyperband Trials (1 iteration is max_epochs * (math.log(max_epochs, factor) ** 2))", args.hyperband_iterations * args.max_epochs * (math.log(args.max_epochs, args.factor) ** 2))
    elif args.tuner in ['bayes', 'bayesian']:
        tuner = BayesianOptimization(
            hypermodel,
            max_trials=MAX_TRIALS, # # of hyperparameter combinations tested by tuner
            num_initial_points=args.num_initial_points, #2, 
            alpha=args.alpha, #0.0001, #added to diagonal of kernel matrix during fitting. represents expected noise in performances
            beta=args.beta, #2.6, #balance exploration v exploitation. larger more explorative
            project_name=PROJ_NAME,
            **tuner_args
        )
    elif args.tuner == 'custom': 
        #TODO
        pass
    else:
        tuner = tuner_module.Tuner(
            oracle=oracle_module.Oracle(),
            hypermodel=hypermodel,
            #optimizer=None,
            #loss=None,
            #distribution_strategy=None,
            project_name=PROJ_NAME,
            #overwrite=args.overwrite,
            directory=args.out_dir
            #**tuner_args
        )

    print('\n==============================')
    tuner.search_space_summary()
    print(' ')

    return tuner, hypermodel

def execute_search(args, tuner, X_train, X_val=None, callbacks=[], 
                   train_val_steps=None, cdatetime='', DB=0, **kwargs):
    '''
    Execute the hyperparameter search. Calls tuner.search()
    @param args: the command line args object. See create_argsparser() for
            details about the command line arguments
            Command line arguments relevant to this method:
                batch_size 
                epochs 
                project_name_prefix 
                tuner objective 
                patience 
                min_delta 
                out_dir 
    @param tuner: Keras Tuner
    @param X_train: Tensorflow Dataset
    @param X_val: optional Tensorflow Dataset
    @params callbacks: overwrite the default callbacks. Default callbacks are 
            EarlyStopping.
    @param train_val_steps: dict with keys 'steps_per_epoch' and val_steps
    @param cdatetime: formatted datetime string appended to Tensorboard directory name
    @param DB: debug flag to print the resulting hyperparam search space
    @param kwargs: additional keyword arguments
    '''
    BATCH_SIZE = args.batch_size 
    NEPOCHS = args.epochs 
    PROJ_NAME_PREFIX = args.project_name_prefix
    PROJ_NAME = f'{PROJ_NAME_PREFIX}_{args.tuner}'
    
    tuner_dir = args.out_dir_tuning if not args.out_dir_tuning is None  else args.out_dir


    if len(callbacks) == 0:
        # TODO: separate arg for objective and monitor?
        es = EarlyStopping(monitor=args.objective, #start_from_epoch=10, 
                            patience=args.patience, min_delta=args.min_delta, 
                            restore_best_weights=True)
        #tb_path = os.path.join(wandb.run.dir, f'{PROJ_NAME}_tb') #os.path.join(tuner_dir, f'{PROJ_NAME}_{cdatetime}_tb')
        #print(" tb path", tb_path)
        #tb = TensorBoard(tb_path) #, histogram_freq=2) #--logdir=
        #cp = ModelCheckpoint(filepath=f"{tuner_dir}/checkpoints/{PROJ_NAME}', verbose=1, save_freq=5*BATCH_SIZE) #save_weights_only=True, 
        #manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=3)
        callbacks = [es] #, tb] #

    if train_val_steps is None:
        train_val_steps = {'steps_per_epoch': None, 'val_steps': None}
    print(" exe_Search:: train_val_steps", train_val_steps)

    # Perform the hyperparameter search
    print("\nExecuting hyperparameter search...")
    tuner.search(X_train, 
                validation_data=X_val, 
                validation_batch_size=None, 
                #batch_size=BATCH_SIZE, 
                epochs=NEPOCHS, 
                shuffle=False, callbacks=callbacks,
                steps_per_epoch=train_val_steps['steps_per_epoch'], #5 if DB else train_val_steps['steps_per_epoch'], #None, 
                validation_steps=train_val_steps['val_steps'], #5 if DB else train_val_steps['val_steps'], #None,
                #verbose=2, #max_queue_size=10, 
                workers=2, use_multiprocessing=True)
    
    # Finish the wandb run
    #run.finish()

def get_rotations_all(nfolds, DB=0):
    ''' TODO TEST
    Get the fold indices for each rotations.
    @param nfolds: number of folds for the data
    @return: 3-tuple with the 2D numpy arrays of the train, val, and test sets.
            Each column is a fold index. Each row is a rotation.
    '''
    # List of fold indices
    folds = np.arange(nfolds).reshape(1, -1)

    # Matrix of rotations of the folds where each row is a rotation
    folds_mesh = np.repeat(folds, nfolds, axis=0)

    # Rotate folds
    offset = np.arange(nfolds).reshape(-1,1)
    folds_mesh += offset # shift fold index based on rotation (i.e. row)
    folds_mesh = folds_mesh % nfolds # wrap fold indicies

    train_inds = folds_mesh[:, :-2]
    val_inds = folds_mesh[:, -2].reshape(-1, 1)
    test_inds = folds_mesh[:, -1].reshape(-1, 1)

    if DB:
        print("TRAIN SET\n", train_inds)
        print("VAL SET\n", val_inds)
        print("TEST SET\n", test_inds)
    return train_inds, val_inds, test_inds

def get_rotation(nfolds, r, DB=0):
    ''' TODO TEST
    Get the fold indices for a single rotation
    @param nfolds: number of folds for the data
    @param r: int for the rotation index. used as the offset
    @return: 3-tuple with the 2D numpy arrays of the train, val, and test sets.
            Each column is a fold index. Each row is a rotation.
    '''
    # List of fold indices
    folds = np.arange(nfolds).reshape(1, -1)

    # Rotate folds: shift index based on rotation and wrap fold indices
    folds = (folds + r) % nfolds 

    train_inds = folds[:, :-2]
    val_inds = folds[:, -2]
    test_inds = folds[:, -1]

    if DB:
        print("TRAIN SET\n", train_inds)
        print("VAL SET\n", val_inds)
        print("TEST SET\n", test_inds)
    return train_inds, val_inds, test_inds

def take_rotation_data(args, data, folds, r=0):
    ''' TODO
    #def take_rotation_date(args, itrain, ival, itest):
    Grab the actual data based on the fold indicies for a specific rotation
    @param args: the command line args. See create_argsparser() for more details
    @param data: the data to partition
    @param folds: the 2D array of fold indicies. row is the rotation. col is the
            fold index
    @param r: the rotation to select
    '''
    #ins_validation = np.concatenate(np.take(ins, folds_validation), axis=0)
    #outs_validation = np.concatenate(np.take(outs, folds_validation), axis=0)
    return np.concatenate(np.take(data, folds[r]), axis=0)        

def get_dataset_size(ds):
    return ds.reduce(0, lambda x, _: x + 1, name="num_elements").numpy()

def resample_dataset(args, ds_list, weights, stop_on_empty_dataset=True, 
                     method='sample', init_dist=None, ntake=10000, DB=False):
    ''' TODO
    Resample the dataset using one of three approaches
    @param weights: list of sample weights for each Dataset in the ds_list; target distribution
    @param method: string, either 'sample' to use tf.Dataset.sample_from_dataset()
            or 'resample' to use tf.Dataset.rejection_resample()
    @return: tf.data.Dataset
    '''
    ds = None

    if method == 'take':
        for i, _ds in enumerate(ds_list):
            count = int(weights[i] * ntake)
            _ds = _ds.repeat().take(count)
            if ds is None: ds = _ds
            else: ds = ds.concatenate(_ds)

        '''
        if DB:
            nneg = ds.reduce(0, lambda res, x,y: res + tf.math.reduce_any(y <= 0), name="num_elements").numpy()
            npos = ds.reduce(0, lambda res, x,y: res + tf.math.reduce_any(y > 0), name="num_elements").numpy()
            pneg = nneg / (nneg + npos)
            ppos = npos / (nneg + npos)
            print(f" resmaple:: #neg={nneg} ({pneg}) #pos={npos} ({ppos})")

            #checking ratio
            #_neg = ds_val.filter(filter_neg, name='_nt_val') 
            #_pos = ds_val.filter(filter_pos, name='_t_val')
            #nn = get_dataset_size(_neg) 
            #np = get_dataset_size(_pos) 
            #print(f"(resampled) n neg {nn} ({nn / (nn + np)}) n pos {np} ({np / (nn + np)}) ")
        '''

    elif method == 'sample':
        for i, _ds in enumerate(ds_list):
            ds_list[i] = _ds#.repeat()
        
        ds = tf.data.Dataset.sample_from_datasets(ds_list, #[ds_neg, ds_pos],
                                                  weights=weights,
                                                  stop_on_empty_dataset=stop_on_empty_dataset)
    else: 
        def which_class(x, y):
            return tf.cast(tf.math.reduce_any(y > 0), dtype=tf.int32)
        
        ds = ds_list[0].rejection_resample(
                class_func=which_class, 
                #class_func=lambda f, l: which_class(f),
                #class_func=lambda x, y: tf.py_function(which_class, [x, y], [tf.int32]), 
                #class_func=lambda x, y: tf.cast(tf.math.reduce_any(y > 0), dtype=tf.int32), #which_class, #map input dataset to scalar tf.int32. Values in [0, num_classes).
                target_dist=weights, #float type tensor, shaped [num_classes]
                initial_dist=init_dist, #(Optional.) float tensor, shaped [num_classes]. If not provided, true class distribution estimated live
                name='resample') # returns ds of tuples (label, (feat, label))
        if DB:
            print(" spec:", ds.element_spec)
            _ds = list(ds.as_numpy_iterator())
            _ds = [lst[0] for lst in _ds]
            print(" rejection_resample:: _ds[0]", _ds[:5])
            #zero, one = np.bincount(_ds) / len(_ds) 
            #print("rejection sample 1", len(_ds), zero, one)

        # Remove resample class_func result
        #(<tf.Tensor 'args_1:0' shape=(32, 32, 12) dtype=float32>, <tf.Tensor 'args_2:0' shape=(32, 32, 1) dtype=int64>)
        specs = (tf.TensorSpec(shape=(32, 32, 12), dtype=tf.float32), 
                     tf.TensorSpec(shape=(32, 32, 1), dtype=tf.int64))
        #@tf.autograph.experimental.do_not_convert
        def identity(class_func_result, data):
            return data
        ds = ds.map(lambda class_func_result, data: data) # returns ds of tuples (feat, label)
        '''
        if DB:
            ds_np = list(ds.as_numpy_iterator())
            ds_np = [lst[1] for lst in ds_np]
            print(" map:: ds_np[0]", ds_np[:5])
            #zero, one = np.bincount(ds_np) / len(ds_np)  #n
            #print("rejection sample 2", zero, one, len(ds_np))
        '''

        # Repeat resampled dataset so it's at least as big as the original dataset
        print("ds.cardinality().numpy()", ds.cardinality().numpy())
        ds = ds.repeat(2).take(ds.cardinality().numpy())
        if DB:
            ds_np = list(ds.as_numpy_iterator())
            ds_np = [lst[1] for lst in ds_np]
            print(" take:: ds_np[0]", ds_np[:5])
            #zero, one = np.bincount(ds_np) / len(ds_np)  #n
            #print("rejection sample 3", zero, one, len(ds_np))
    return ds

def prep_data(args, n_labels=None, sample_method='sample', DB=1):
    """ 
    Load and prepare the data.
    dataset split https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
    @param args: the command line args object. See create_argsparser() for
            details about the command line arguments
            Command line arguments relevant to this method:
                    in_dir
                    in_dir_val
                    in_dir_test
                    lscratch
                    batch_size
                    class_weight
                    resample

    @param n_labels: number of class labels. If None, tune as a hyperparameter in 
                that can either be 1 or 2.
    @param method: string, either 'sample' to use tf.Dataset.sample_from_dataset()
            or 'resample' to use tf.Dataset.rejection_resample()

    @return: tuple with the training and validation sets as Tensorflow Datasets
    """ 
    # Dataset size
    x_shape = args.x_shape #(None, *args.x_shape) #model-->(None, 32, 32, 12)
    x_shape_val = args.x_shape #(None, *args.x_shape) #(32, 32, 12)
    
    y_shape = args.y_shape #(None, 32, 32, 1) #(None, *args.y_shape)
    y_shape_val = args.y_shape

    if n_labels == 1:        
        #y_shape = (None, *args.y_shape) #(None, 32, 32, 1)
        specs = (tf.TensorSpec(shape=x_shape, dtype=tf.float64, name='X'), 
                     tf.TensorSpec(shape=y_shape, dtype=tf.int64, name='Y'))

        #y_shape_val = args.y_shape #(32, 32, 1)
        specs_val = (tf.TensorSpec(shape=x_shape_val, dtype=tf.float64, name='X'), 
                         tf.TensorSpec(shape=y_shape_val, dtype=tf.int64, name='Y'))

    else:
        # Define the dataset size
        #y_shape = (None, 32, 32, 2)
        specs = (tf.TensorSpec(shape=x_shape, dtype=tf.float64, name='X'), 
                     tf.TensorSpec(shape=y_shape, dtype=tf.float32, name='Y'))

        #y_shape_val = y_shape[1:] #(32, 32, 2)
        specs_val = (tf.TensorSpec(shape=x_shape_val, dtype=tf.float64, name='X'), 
                         tf.TensorSpec(shape=y_shape_val, dtype=tf.float32, name='Y'))

    # tf.Dataset helper methods
    #@tf.function
    #@tf.autograph.experimental.do_not_convert
    def filter_neg(x, y):
        # Get non tornadic storms
        return tf.math.reduce_all(y <= 0)
    
    #@tf.function
    #@tf.autograph.experimental.do_not_convert
    def filter_pos(x, y):
        # Get tornadic storms
        return tf.math.reduce_any(y > 0)
    
    def identity(x, y):
        return x, y
    
    def change_spec(x, y):
        x = tf.cast(x, tfdtype, name='X') #tf.float16
        y = tf.cast(y, tf.int16, name='Y')
        return x, y
    
    #@tf.function
    def add_sample_weight(x, y):
        # Include a sample weight
        label = tf.cast(tf.math.reduce_any(y > 0), dtype=tfdtype) #tf.float16
        weight = (1. - label) * (args.class_weight[0]) + label * (1. - args.class_weight[0])
        weight = tf.cast(weight, dtype=tfdtype) #tf.float16
        return x, y, tf.reshape(weight, (1,))

    # Dict with steps per epoch
    train_val_steps = {}

    # TRAIN SET
    ds_train = tf.data.Dataset.load(args.in_dir) #, specs)
    ds_train = ds_train.map(change_spec, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train_og = ds_train.take(-1) 
    print("Train Dataset (load):", ds_train)

    ds_neg = ds_train.filter(filter_neg, name='nontor') 
    ds_pos = ds_train.filter(filter_pos, name='tor') 

    nneg = get_dataset_size(ds_neg) 
    npos = get_dataset_size(ds_pos)
    ntrain = nneg + npos
    nsteps = np.max([25, ntrain / args.batch_size // args.epochs])
    train_val_steps['steps_per_epoch'] = nsteps #`steps_per_epoch * epochs`= batches
    print(f"n neg {nneg} ({nneg / ntrain}) n pos {npos} ({npos / ntrain}) ")

    if args.lscratch is not None:
        cache_dir = os.path.join(args.lscratch, 'tornado_jtti')
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            print(f"Making dir: {cache_dir}")
        else:
            print(f"Dir exists: {cache_dir}")
        
        cache_file = os.path.join(cache_dir, 'nontor')
        ds_neg = ds_neg.take(-1).cache(filename=cache_file).repeat() #.take(-1).cache().repeat()
        print(f'CACHE FILE NAME: {cache_file}')

        cache_file = os.path.join(cache_dir, 'tor')
        ds_pos = ds_pos.take(-1).cache(filename=cache_file).repeat() 
        print(f'CACHE FILE NAME: {cache_file}')
    else:
        print("No LSCRATCH, no caching")
        ds_neg = ds_neg.repeat()
        ds_pos = ds_pos.repeat()

    # Use inverse natural distribution ratio for the class_weight
    print(f"Class weights current {args.class_weight}")
    if args.class_weight == [-1]:
        args.class_weight = [npos / ntrain, nneg / ntrain]
        print(f"Class weights set to {args.class_weight}")

    # Resample dataset if resampling weights are provided
    if not args.resample is None and nneg > 0 and npos > 0: 
        print("Producing sample tf Dataset", args.resample)
        ds_list = [ds_neg, ds_pos] if sample_method in ['sample', 'take']  else [ds_train]
        ds_train = resample_dataset(args, ds_list, weights=args.resample,
                                    method=sample_method, 
                                    init_dist=[nneg / ntrain, npos / ntrain], DB=DB)
        print("Train Dataset (resample):", ds_train)
    else:
        ds_train = ds_train.repeat()

    # Apply the class weights to the samples as sample weights
    if not args.class_weight is None: 
        ds_train = ds_train.map(add_sample_weight, name='weighted', 
                                num_parallel_calls=tf.data.AUTOTUNE)
        ds_train_og = ds_train_og.map(add_sample_weight, name='weighted_og', 
                                num_parallel_calls=tf.data.AUTOTUNE)
        print("Train Dataset (map::class_weight):", ds_train)

    ds_train = ds_train.batch(args.batch_size, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train_og = ds_train_og.batch(args.batch_size, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.prefetch(6)
    ds_train_og = ds_train_og.prefetch(6)
    print("Train Dataset:", ds_train)

    # VAL SET
    ds_val = tf.data.Dataset.load(args.in_dir_val)
    ds_val = ds_val.map(change_spec, num_parallel_calls=tf.data.AUTOTUNE)
    ds_val_og = ds_val.take(-1) #map(identity, num_parallel_calls=tf.data.AUTOTUNE)
    print("Val Dataset:", ds_val)
    
    ds_val_neg = ds_val.filter(filter_neg, name='nontor_val') 
    ds_val_pos = ds_val.filter(filter_pos, name='tor_val')

    nneg = get_dataset_size(ds_val_neg) 
    npos = get_dataset_size(ds_val_pos) 
    nval = nneg + npos
    nsteps = np.max([25, nval / args.batch_size // args.epochs])
    train_val_steps['val_steps'] = nsteps
    print(f"n neg {nneg} ({nneg / nval}) n pos {npos} ({npos / nval}) nval {nval}")
    
    # Resample 
    if not args.resample is None and nneg > 0 and npos > 0:
        ds_list = [ds_val_neg, ds_val_pos] if sample_method in ['sample', 'take']  else [ds_val]
        ds_val = resample_dataset(args, ds_list, weights=args.resample,
                                  method=sample_method)
        print("Val Dataset (resample):", ds_val)
    else:
        ds_val = ds_val.repeat()

    ds_val = ds_val.batch(args.batch_size, num_parallel_calls=tf.data.AUTOTUNE)
    ds_val_og = ds_val_og.batch(args.batch_size, num_parallel_calls=tf.data.AUTOTUNE)
    ds_val = ds_val.prefetch(6)
    ds_val_og = ds_val_og.prefetch(6)
    print("Val Dataset:", ds_val)

    # TEST SET
    ds_test = None
    if not args.in_dir_test is None:
        ds_test = tf.data.Dataset.load(args.in_dir_test)
        ds_test = ds_test.map(change_spec, num_parallel_calls=tf.data.AUTOTUNE)
        print("Test Dataset:", ds_test)

        ds_test_neg = ds_test.filter(filter_neg, name="nontor_test")
        ds_test_pos = ds_test.filter(filter_pos, name="tor_test")

        nneg = get_dataset_size(ds_test_neg)
        npos = get_dataset_size(ds_test_pos) 
        print(f"n neg {nneg} ({nneg / (nneg + npos)}) n pos {npos} ({npos / (nneg + npos)}) ")

        ds_test = ds_test.batch(args.batch_size)
        ds_test = ds_test.prefetch(4)
        print("Test Dataset:", ds_test)

    return (ds_train, ds_val, ds_test, train_val_steps, ds_train_og, ds_val_og)

def fvaf(y_true, y_pred):
    ''' TODO
    Fraction of variance accounted for (FVAF) ranges (-inf, 1]. 
    1 FVAF represents a total reconstruction. 
    0 represents a reconstruction that's as good as using the average of the 
    signal as predictor
    negative FVAF even worse reconstructions than using the average signal as 
    the predictor
    1 - MSE / VAR =
    1 - (sum[(y_true - y_pred)**2]) / (sum[(y_true - y_mean)**2])
    '''
    tf_mse = tf.keras.losses.MeanSquaredError()
    MSE = tf_mse(y_true, y_pred).numpy()
    VAR = np.var(y_true.flatten()) #tf.math.reduce_variance(y_true)
    return 1. - MSE / VAR

def compute_sr(tps, fps):
    '''
    Compute the SR (success ration) from a scalars or lists of true positives
    (TPs) and false positives (FPs)
    @param tps: scalar or numpy array of true positives
    @param fps: scalar or numpy array of false positives
    @return: scalar or numpy array for the SR
    '''
    return tps / (tps + fps)

def compute_pod(tps, fns):
    '''
    Compute the probability of detection (POD) also the true positive rate (TPR)
    @param tps: scalar or numpy array of true positives
    @param fns: scalar or numpy array of false negatives
    @return: scalar or numpy array for the POD
    '''
    return tps / (tps + fns)

def compute_csi(tps, fns, fps):
    '''
    Compute the CSI (crtical success index) from a scalars or lists of the true 
    positives (TPs), false negatives (FNs), and the false positives (FPs)
    @param tps: scalar or numpy array of true positives
    @param fns: scalar or numpy array of false negatives
    @param fps: scalar or numpy array of false positives
    @return: scalar or numpy array for the CSI
    '''
    return tps / (tps + fns + fps)

def csi_from_sr_and_pod(success_ratio_array, pod_array):
    """
    Based on method from Dr. Ryan Laguerquist and found originally in his 
    gewitter repo (https://github.com/thunderhoser/GewitterGefahr).
    Computes CSI (critical success index) from success ratio and POD.
    POD = probability of detection
    :param success_ratio_array: np array (any shape) of success ratios.
    :param pod_array: np array (same shape) of POD values.
    :return: csi_array: np array (same shape) of CSI values.
    """
    return (success_ratio_array ** -1 + pod_array ** -1 - 1.) ** -1

def frequency_bias_from_sr_and_pod(success_ratio_array, pod_array):
    """
    Based on method from Dr. Ryan Laguerquist and found originally in his 
    gewitter repo (https://github.com/thunderhoser/GewitterGefahr).
    Computes frequency bias from success ratio and POD.
    POD = probability of detection
    :param success_ratio_array: np array (any shape) of success ratios.
    :param pod_array: np array (same shape) of POD values.
    :return: frequency_bias_array: np array (same shape) of frequency biases.
    """
    return pod_array / success_ratio_array

def contingency_curves(y, y_preds, threshs):
    '''
    Compute the contingency/confusion matrix for multiple thresholds to use in
    various types of performance curves
    '''
    tp = tf.keras.metrics.TruePositives(thresholds=threshs)
    fp = tf.keras.metrics.FalsePositives(thresholds=threshs)
    fn = tf.keras.metrics.FalseNegatives(thresholds=threshs)
    tn = tf.keras.metrics.TrueNegatives(thresholds=threshs)

    # Get tp, fp and fn 
    tp.reset_state()
    fp.reset_state()
    fn.reset_state()
    tn.reset_state()

    tps = tp(y, y_preds)
    fps = fp(y, y_preds)
    fns = fn(y, y_preds)
    tns = tn(y, y_preds)
    return tps, fps, fns, tns

def get_max_csi(y, y_preds, thresh):
    ''' TODO
    @param y: true output
    @param y_preds: predicted output
    @param thresh: probability threholds #thresh=np.arange(0.05, 1.05, 0.05)
    '''
    #tp = tf.keras.metrics.TruePositives(thresholds=thresh.tolist())
    #fp = tf.keras.metrics.FalsePositives(thresholds=thresh.tolist())
    #fn = tf.keras.metrics.FalseNegatives(thresholds=thresh.tolist())
    #mcsi = MaxCriticalSuccessIndex(tp, fp, fn)
    mcsi = MaxCriticalSuccessIndex()
    mcsi.reset_state()
    mcsis = mcsi(y, y_preds).numpy()

    # TODO
    tp = mcsis.tp
    fp = mcsis.fp
    fn = mcsis.fn
    tn = 0 #TODO mcsis.tn  thresholds

    return mcsis, tp, fp, fn, tn

def plot_learning_loss(history, fname, save=False, dpi=180):
    '''
    Plot the juxtaposition of the training and validation loss
    @param history: history object returned from model.fit()
    @return: the figure
    '''
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel("Learning Epoch")
    plt.ylabel("Loss")
    plt.legend()

    if save:
        print("Saving history plot")
        print(fname)
        plt.savefig(fname, dpi=dpi)

    return plt.gcf()

def plot_predictions(y_preds, y_preds_val, fname, use_seaborn=True, fig_ax=None, 
                     figsize=(10, 8), alpha=.5, save=False, dpi=160):
    '''
    Plot histograms of the distribution of prediction values
    @return: tuple with the fig and axes objects
    '''
    fig = None
    axs = None
    if fig_ax is None:
        fig, axs = plt.subplots(2, 1, figsize=figsize)
        axs = axs.ravel()
    else:
        fig, axs = fig_ax

    if use_seaborn:
        from seaborn import histplot

        Y = {'Train': y_preds, 'Val': y_preds_val}

        histplot(data=Y, stat='probability', legend=True, #, label='Train Set'
                 ax=axs[0], alpha=alpha, common_norm=False)
        axs[0].set_xlabel('') #Tornado Predicted Probability
        axs[0].set_xlim([0, 1])
        axs[0].legend(list(Y.keys()), loc='center right')

        histplot(data=Y, stat='probability', legend=True, #, label='Train Set'
                 ax=axs[1], alpha=alpha, common_norm=False, cumulative=True, 
                 element="step", fill=False)
        axs[1].set_xlabel('Tornado Predicted Probability')
        axs[1].set_ylabel('Cumulative Probability')
        axs[1].set_xlim([0, 1])
        axs[1].legend(list(Y.keys()), loc='center right')
        '''
        histplot(data=y_preds_val, stat='probability', label='Val Set', legend=True, ax=axs[2])
        axs[2].set_xlabel('Probability')
        axs[2].set_xlim([0, 1])

        histplot(data=y_preds_val, stat='probability', label='Val Set', legend=True, ax=axs[3], cumulative=True, element="step", fill=False)
        axs[3].set_xlabel('Probability')
        axs[3].set_xlim([0, 1])
        '''
    else:
        from matplotlib import colors

        # TRAIN SET
        y_train = y_preds.ravel()
        axs[0].hist(y_train, density=True, label='Train Set') #, norm=colors.LogNorm())
        axs[0].set_xlabel('Probability')
        axs[0].set_ylabel('density')
        axs[0].set_xlim([0, 1])
        axs[0].legend(loc='center right')

        axs[1].hist(y_train, density=True, cumulative=True, label='Train Set', histtype='step') #, norm=colors.LogNorm())
        axs[1].set_xlabel('Probability')
        axs[1].set_ylabel('cumulative density')
        axs[1].set_xlim([0, .6])
        axs[1].legend(loc='center right')

        # VAL SET
        y_val = y_preds_val.ravel()
        axs[2].hist(y_val, density=True, label='Val Set') #, norm=colors.LogNorm())
        axs[2].set_xlabel('Probability')
        axs[2].set_ylabel('density')
        axs[2].set_xlim([0, 1])
        axs[2].legend(loc='center right')

        axs[3].hist(y_val, density=True, cumulative=True, label='Val Set', histtype='step') #, norm=colors.LogNorm())
        axs[3].set_xlabel('Probability')
        axs[3].set_ylabel('cumulative density')
        axs[3].set_xlim([0, .6])
        axs[3].legend(loc='center right')

    plt.suptitle("Tornado Prediction Probabilities")

    if save:
        print("Saving prediction histograms")
        print(fname)
        plt.savefig(fname, dpi=dpi)

    return fig, axs

def plot_preds_hists(Y, fname, use_seaborn=True, fig_ax=None, zoom_ax=False,
                     figsize=(10, 8), alpha=.5, save=False, dpi=160):
    '''
    Fancy Plot histograms of the distribution of prediction values
    @param Y: dict of arrays to plot. see the 'data' argument for 
            seaborn.histplot()
    @return: tuple with the fig and axes objects
    '''
    fig = None
    axs = None
    if fig_ax is None:
        fig, axs = plt.subplots(2, 1, figsize=figsize)
        axs = axs.ravel()
    else:
        fig, axs = fig_ax

    from seaborn import histplot

    # Distribution
    histplot(data=Y, stat='probability', legend=True, 
             ax=axs[0], alpha=alpha, common_norm=False)
    axs[0].set_xlabel('') #Tornado Predicted Probability
    axs[0].set_xlim([0, 1])
    axs[0].legend(list(Y.keys()), loc='center right', bbox_to_anchor=(1.16, 1))

    ## Zoom
    if zoom_ax:
        ax2 = plt.axes([0.4, 0.0, .15, .15]) #left, bottom, width, height
        histplot(Y, ax=ax2, stat='probability', legend=True, alpha=alpha, 
                common_norm=False) #distplot
        #ax2.set_title('zoom')
        ax2.set_xlim([0, .05])

    # Cumulative
    histplot(data=Y, stat='probability', legend=True,
             ax=axs[1], alpha=alpha, common_norm=False, 
             cumulative=True, element="step", fill=False)
    axs[1].set_xlabel('Tornado Predicted Probability')
    axs[1].set_ylabel('Cumulative Probability')
    axs[1].set_xlim([0, 1])
    axs[1].legend(list(Y.keys()), loc='center right', bbox_to_anchor=(1.16, 1))

    ## Zoom
    if zoom_ax:
        ax2 = plt.axes([0.4, 0.05, .15, .15]) #left, bottom, width, height
        histplot(Y, ax=ax2, stat='probability', legend=True, alpha=alpha, 
                common_norm=False, cumulative=True, element="step", fill=False) #distplot
        #ax2.set_title('zoom')
        ax2.set_xlim([0, .05])
        ax2.legend([])

    plt.suptitle("Tornado Prediction Probabilities")

    if save:
        print("Saving prediction histograms")
        print(fname)
        plt.savefig(fname, dpi=dpi)

    return fig, axs

def plot_confusion_matrix(y, y_preds, fname, thresh, p=.5, fig_ax=None, 
                          figsize=(5, 5), save=False, dpi=160):
    '''
    Compute and plot the confusion matrix based on the cutoff p.
    Based on method from Tensorflow docs.

    @param y: true output
    @param y_preds: predicted output
    @param fname: file name to save the figure as
    @param thresh: list of the thresholds for other performance plots #thresh=np.arange(0.05, 1.05, 0.05)
    @param p: cutoff probability above which is labelled 1
    @param figsize: tuple with the width and height of the figure
    @param fig_ax: (optional) tuple with existing figure and axes objects to use
    @param save: bool flag whether to save the figure
    @param dpi: integer resolution of the saved figure in dots per inch

    @return: tuple with the fig and axes objects
    '''
    from seaborn import heatmap
    
    fig = None
    ax = None
    if fig_ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig, ax = fig_ax
    #fig, ax = plt.subplots(1, 1, figsize=figsize)
    #axs = axs.ravel()

    cm = confusion_matrix(y, y_preds > p)
    heatmap(cm, annot=True, fmt="d", ax=ax)
    ax.set_title(f'p > {p:.2f}')
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('Actual label')
    ax.set_aspect('equal')

    print(" ")
    print('TN: ', cm[0, 0])
    print('FP: ', cm[0, 1])
    print('FN: ', cm[1, 0])
    print('TP: ', cm[1, 1])
    print('Total: ', cm.sum())

    #TODO: mcsis, tp, fp, fn, tn = get_max_csi(y, y_preds, thresh=thresh)

    if save:
        print("Saving confusion matrix")
        print(fname)
        plt.savefig(fname, dpi=dpi)

    return fig, ax

def plot_roc(y, y_preds, fname, tpr_fpr=None, fig_ax=None, figsize=(10, 10), 
             save=False, dpi=160, DB=False, plot_ann=False, return_scores=False, 
             **kwargs):
    '''
    Plot the Reciever Operating Characteristic (ROC) Curve
    @param y: true output
    @param y_preds: predicted output
    @param fname: file name to save the figure as
    @param tpr_fpr: tuple containing lists for the true positives rate, at index 
            0, and the false positive rate, at index 1. (tpr, fpr)
    @param fig_ax: (optional) tuple with existing figure and axes objects to use
    @param figsize: tuple with the width and height of the figure
    @param save: bool flag whether to save the figure
    @param dpi: integer resolution of the saved figure in dots per inch
    @param plot_ann: bool or int. include plot annotations
            1: show no skill line
            2: show AUC and max CSI value
            3: show all
    @param **kwargs: additional keyword arguments from Axes.plot()
    @return: tuple with the fig and axes objects
    '''
    fig = None
    ax = None
    if fig_ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig, ax = fig_ax

    tpr = None
    fpr = None
    if tpr_fpr:
        tpr, fpr = tpr_fpr
    else: 
        fpr, tpr, _ = roc_curve(y, y_preds)

    # AUC
    _auc = auc(fpr, tpr)
    if DB: print("AUC", _auc)

    ax.plot(fpr, tpr, linewidth=3, **kwargs)
    if plot_ann in [1, 3]: ax.plot([0, 1], [0, 1], '--', label='No skill')

    # Max PSS = TPR - FPR
    pss = tpr - fpr
    imax = np.nanargmax(pss)
    fmax = fpr[imax]
    tmax = tpr[imax]

    plt1 = ax.plot(fmax, tmax, '*', c='r', ms=15, label='Max PSS')
    if plot_ann in [2, 3]:
        text = f'{pss[imax]:.02f}'
        ax.text(fmax-0.12, tmax-0.05, text, fontsize=18, color='k')
        ax.annotate(f"AUC = {_auc:0.2f}", xy=(.55, .05), fontsize=18, color='k')

    ax.set(xlabel='FPR', ylabel='TPR')
    ax.set(xlim=[0, 1], ylim=[0, 1])
    ax.grid(True)
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.01))
    ax.set_aspect('equal')

    if save:
        print("Saving ROC plot")
        print(fname)
        plt.savefig(fname, dpi=dpi)

    if return_scores:
        return fig, ax, {'tpr': tpr, 'fpr': fpr, 'pss': pss, 'auc': _auc}
    return fig, ax

def plot_prc(y, y_preds, fname, pre_rec_posrate=None, fig_ax=None, 
             figsize=(10, 10), save=False, dpi=160, draw_ann=1, DB=False, **kwargs):
    '''
    Plot the Precision Recall Curve (PRC)
    @param y: true output
    @param y_preds: predicted output
    @param fname: file name to save the figure as
    @param pre_rec_posrate: 3-tuple containing lists for the precision, at index 
            0, the recall, at index 1, and the positive chance rate. 
            eg (precision, recall, pos_rate)
    @param fig_ax: (optional) tuple with existing figure and axes objects to use
    @param figsize: tuple with the width and height of the figure
    @param save: bool flag whether to save the figure
    @param dpi: integer resolution of the saved figure in dots per inch
    @param draw_ann: int indicating annotations to draw. 0: draw none, 1: draw
            diag, 2: draw f1 curves, 3: draw both
    @param **kwargs: additional keyword arguments from Axes.plot()
    @return: tuple with the fig and axes objects
    '''
    fig = None
    ax = None
    if fig_ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig, ax = fig_ax

    precision = []
    recall = []
    pos_chance_rate = 1
    if pre_rec_posrate:
        precision, recall, pos_chance_rate = pre_rec_posrate
    else:
        precision, recall, _ = precision_recall_curve(y, y_preds)
        pos_chance_rate = np.count_nonzero(y) / y.size

    # Max F1 score
    f1 = 2 * precision * recall / (precision + recall)
    imax = np.nanargmax(f1)
    pmax = precision[imax]
    rmax = recall[imax]

    ax.hlines(pos_chance_rate, 0, 1, linewidth=3, linestyles='--', #color='r', 
              label=f'Chance ({pos_chance_rate:.02f})')
    if draw_ann in [1, 3]: ax.plot([0, 1], [1, 0], '--')

    # F1-score curves
    if draw_ann >= 2:
        f_scores = np.linspace(0.2, 0.8, num=4)
        x = np.linspace(0.01, 1)
        for f_score in f_scores:
            y = f_score * x / (2 * x - f_score)
            ax.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.8)
            ax.annotate(f"F1 = {f_score:0.1f}", xy=(0.85, y[45] + 0.02), fontsize=6)

    ax.plot(precision, recall, linewidth=3, **kwargs)
    plt1 = ax.plot(pmax, rmax, '*', c='r', ms=15, label='Max F1')
    text = f'{f1[imax]:.02f}'
    ax.text(pmax-0.12, rmax-0.05, text, fontsize=18, color='k')

    keep = np.where(~np.isnan(precision))
    _prec = precision[keep]
    _reca = recall[keep]
    _inds = np.argsort(_prec)
    _auc = auc(_prec[_inds], _reca[_inds])
    ax.annotate(f"AUC = {_auc:0.2f}", xy=(.55, .9), fontsize=18, color='k') #, transform=ax.transData) #fig.transFigure) #
    if DB: print("AUC", _auc)

    ax.set(xlabel='Precision', ylabel='Recall')
    ax.set(xlim=[0, 1], ylim=[0, 1])
    ax.grid(True, color='k', alpha=.1, linewidth=2)
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.01)) 
    ax.set_aspect('equal')

    if save:
        print("Saving PRC plot")
        print(fname)
        plt.savefig(fname, dpi=dpi)

    return fig, ax

def plot_reliabilty_curve(y, y_preds, fname, n_bins=20, strategy='quantile', 
                          fig_ax=None, figsize=(10, 10), save=False, dpi=160, 
                          **kwargs):
    '''
    Plot the reliability curve. Perfect model follows the y = x line. This curve
    compares the quality of probabilistic predictions of binary classifiers by
    plotting the true frequency of the positive label against its predicted 
    probability. See calibration_curve() in Sci-kit (sklearn) for more details
    @param y: true output
    @param y_preds: predicted output
    @param fname: file name to save the figure as
    @param n_bins:
    @param strategy: 
    @param fig_ax: (optional) tuple with existing figure and axes objects to use
    @param figsize: tuple with the width and height of the figure
    @param save: bool flag whether to save the figure
    @param dpi: integer resolution of the saved figure in dots per inch
    @param **kwargs: additional keyword arguments from Axes.plot()
    @return: tuple with the fig and axes objects
    '''
    fig = None
    ax = None
    if fig_ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig, ax = fig_ax

    # Calculate observed frequency and predicted probabilities
    prob, prob_preds = calibration_curve(y, y_preds, n_bins=n_bins, 
                                         strategy=strategy)
    print(" reliability:: probs quantile", np.quantile(prob, [0, .5, 1]))
    print(" reliability:: probs_preds quantile", np.quantile(prob_preds, [0, .5, 1]))

    ax.plot(prob, prob_preds, **kwargs)
    ax.plot([0, 1], linestyle='--', color='k')
    ax.set_xlabel("Observed Frequency")
    ax.set_ylabel("Predicted Probability")
    ax.set_aspect('equal')
    ax.grid(True)
    ax.legend()
    #plt.tight_layout()

    if save:
        print("Saving reliability plot")
        print(fname)
        plt.savefig(fname, dpi=dpi)

    return fig, ax

def make_csi_axis(ax=None, figsize=(10, 10), show_csi=True, show_fb=True, 
                  csi_cmap='Greys_r', show_cb=True, fb_strfmt='%.2f', fb_padding=5):
    '''
    Based on make_performance_diagram_axis() from Dr. Ryan Laguerquist and found 
    originally in his gewitter repo (https://github.com/thunderhoser/GewitterGefahr).
    Helper function to make the axes for the performance plot of the CSI. 
    @param ax: matplotlib axes object to use
    @param figsize: tuple with the width and height of the figure
    @param show_csi: bool whether to plot the CSI contours
    @param show_fb: bool whether to plot the frequency bias lines
    @param csi_cmap: color map to use for the CSI contours
    @param show_cb: bool whether to show the CSI colorbar
    @param fb_strfmt: string format for the frequency bias values
    @param fb_padding: space in pixels on each side of the frequency bias labels
    '''
    # For text outlines
    import matplotlib.patheffects as path_effects
    pe = [path_effects.withStroke(linewidth=2, foreground="k")]
    pe2 = [path_effects.withStroke(linewidth=2, foreground="w")]

    if ax is None:
        fig=plt.figure(figsize=figsize)
        fig.set_facecolor('w')
        ax = plt.gca()
    
    if show_csi:
        sr_array = np.linspace(0.001, 1, 200)
        pod_array = np.linspace(0.001, 1, 200)
        X, Y = np.meshgrid(sr_array, pod_array)
        csi_vals = csi_from_sr_and_pod(X, Y)
        pm = ax.contourf(X, Y, csi_vals, levels=np.arange(0,1.1,0.1), cmap=csi_cmap)
        if show_cb: plt.colorbar(pm, ax=ax, label='CSI') #, shrink=1)
        #from mpl_toolkits.axes_grid1 import make_axes_locatable;divider = make_axes_locatable(ax1); cax1 = divider.append_axes("right", size="5%", pad=0.05)
        #https://www.geeksforgeeks.org/set-matplotlib-colorbar-size-to-match-graph/
        #https://en.moonbooks.org/Articles/How-to-match-the-colorbar-size-with-the-figure-size-in-matpltolib-/
        #https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
    
    if show_fb:
        fb = frequency_bias_from_sr_and_pod(X, Y)
        bias = ax.contour(X, Y, fb, levels=[.25,.5,.75,1,1.5,2,3,5], 
                          linestyles='--', colors='Grey')
        plt.clabel(bias, inline=True, inline_spacing=fb_padding, 
                   fmt=fb_strfmt, fontsize=15, colors='Grey')
    
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel('SR')
    ax.set_ylabel('POD')
    return ax

def plot_csi(y, y_preds, fname, label, threshs, fig_ax=None, color='dodgerblue', 
             figsize=(10, 10), save=False, dpi=160, srs_pods_csis=None, 
             return_scores=False, pt_ann=True, tight=False, draw_ann=0, **csiargs):#, **plotargs):
    '''
    Plot the performance curve. This relates to the Critical Success Index (CSI).
    The top right corner shows increasingly better predictions, and where 
    CSI = 1. (this curve is highly senstive to event freq)
    @param threshs=np.linspace(0, 1, 21)
    @param srs_pods_csis_prate: 4-tuple withe the lists for the SRs, PODs, CSIs, 
            positive chance rate (prate)
    @param return_scores: bool. whether to also return the scores in a dict
    @param pt_ann: bool whether to render threshold annotations
    @param csiargs: keyword args for make_csi_axis()
    '''
    fig = None
    ax = None
    if fig_ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig, ax = fig_ax

    # For text outlines 
    pe1 = [path_effects.withStroke(linewidth=1.5, foreground="k")]
    pe2 = [path_effects.withStroke(linewidth=1.5, foreground="w")]

    # Calculate performance diagram 
    tps = []
    fps = []
    fns = []
    tns = [] 
    srs = []
    pods = []
    csis = []
    prate = None
    if srs_pods_csis:
        srs, pods, csis, prate = srs_pods_csis
    else:
        tps, fps, fns, tns = contingency_curves(y, y_preds, threshs.tolist())
        srs = compute_sr(tps, fps) #tps / (tps + fps)
        pods = compute_pod(tps, fns) #tps / (tps + fns)
        csis = compute_csi(tps, fns, fps) #tps / (tps + fns + fps)

    # Get index where thresholds are above the max probability and POD/TP = 0
    idx_stop = pods.shape[0]
    '''
    logic = np.logical_or(threshs > np.nanmax(y_preds), pods <= 0)
    inds = np.where(logic)[0] #np.where(pods <= 0)[0] 
    if inds.size > 0:
        idx_stop = sorted(inds)[0]
        print(idx_stop, threshs[idx_stop-1:idx_stop+1], csis[idx_stop-1:idx_stop+1], 
              srs[idx_stop-1:idx_stop+1], pods[idx_stop-1:idx_stop+1])
    else:
        print(idx_stop-1, threshs[idx_stop-2:idx_stop], csis[idx_stop-2:idx_stop], 
              srs[idx_stop-2:idx_stop], pods[idx_stop-2:idx_stop])
    '''

    # Max CSI
    xi = np.argmax(csis)
    max_csi = csis[xi]
    thres_of_maxcsi = threshs[xi]
    sr_of_maxcsi = srs[xi]
    pod_of_maxcsi = pods[xi]

    nthreshs = threshs.size
    _sel = np.linspace(0, nthreshs - 1, 3, dtype=int).tolist()

    ax = make_csi_axis(ax=ax, **csiargs)
    plt0 = ax.plot(srs, pods, color=color, lw=3, label=label) #, **plotargs) #srs[:idx_stop], pods[:idx_stop]
    ax.plot(np.take(srs, _sel), np.take(pods, _sel), 's', color=color, markerfacecolor='w') 
    plt1 = ax.plot(sr_of_maxcsi, pod_of_maxcsi, '*', c='r', ms=15, label='Max CSI') 

    # Max F1 score (precision=SR; recall=POD)
    f1 = 2 * srs * pods / (srs + pods)
    imax = np.nanargmax(f1)
    pmax = srs[imax]
    rmax = pods[imax]
    #plt1 = ax.plot(pmax, rmax, '*', c='purple', ms=10, label='Max F1')

    # Annotate certain points with the corresponding threshold
    if pt_ann:
        for i, t in zip(_sel, threshs[_sel]): #enumerate(threshs): #[:idx_stop]
            #if np.isnan(srs[i]) or np.isnan(pods[i]): continue
            #if i % 4 and i != nthreshs - 1: continue # skip every other threshold except the last
            text = np.char.ljust(f'{t:.02f}', width=4, fillchar='0') #str(np.round(t, 2))
            ax.text(srs[i]+0.02, pods[i]+0.02, text, path_effects=pe1, fontsize=11, color='white')
            #ax.text(srs[i]+0.02, pods[i]+0.02, text, fontsize=9, color='white')

    # Draw the chance line
    if draw_ann in [1, 3] and prate is not None: 
        ax.hlines(prate, 0, 1, linestyles='--', #color='r', 
              label=f'Chance ({prate:.02f})')

    # Area under the curve
    keep = np.where(~np.isnan(srs))
    _prec = srs[keep]
    _reca = pods[keep]
    _inds = np.argsort(_prec)
    _auc = auc(_prec[_inds], _reca[_inds])

    if draw_ann in [2, 3]:
        ax.annotate(f"AUC = {_auc:0.2f}", xy=(.48, .9), fontsize=18, color='k')

        text = f'Max CSI = {max_csi:.02f}'
        ax.text(.48, .8, text, fontsize=18, color='k')
        #ax.text(sr_of_maxcsi-0.12, pod_of_maxcsi-0.05, text, path_effects=pe1, fontsize=18, color='white')
        
        text = f'Max F1 = {f1[imax]:.02f}'
        ax.text(.48, .7, text, fontsize=18, color='k') # path_effects=pe1,

        print(f"Max F1: {f1[imax]:.02f}. Pre={pmax:.03f}. Rec={rmax:.03f}")
        print(f"Max CSI: {max_csi:.02f}. SR={sr_of_maxcsi:.03f}. POD={pod_of_maxcsi:.03f}")
        print("AUC", _auc)

    ax.legend(loc='upper left', bbox_to_anchor=(1.26, 1.01)) #loc='lower center' (0.5, -.35)  #[plt0, plt1],  #ax.transData
    ax.set_aspect('equal')

    if tight: plt.tight_layout()
    if save:
        print("Saving performance (CSI) plot")
        print(fname)
        plt.savefig(fname, dpi=dpi, bbox_inches='tight')

    if return_scores:
        return fig, ax, {'tps':tps, 'fps':fps, 'fns':fns, 'tns':tns, 'srs':srs, 
                         'pods':pods, 'csis':csis, 'index_max_csi':xi, 
                         'f1s':f1, 'index_max_f1':imax, 'auc':_auc}
    else: return fig, ax

def create_argsparser():
    '''
    Create a command line args parser for the XOR experiment
    @return: the configured arguments parser
    '''
    parser = argparse.ArgumentParser(description='Tornado Model Hyperparameter Search',
                                     epilog='AI2ES')
    ## TODO
    parser.add_argument('--in_dir', type=str, required=True,
                        help='Input directory where the data are stored')
    parser.add_argument('--in_dir_val', type=str, required=True,
                        help='Input directory where the validation data are stored')
    parser.add_argument('--in_dir_test', type=str, #required=True,
                        help='Input directory where the test data are stored')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Output directory for results, models, hyperparameters, etc.')
    parser.add_argument('--out_dir_tuning', type=str, #required=True,
                        help='(optional) Output directory for training and tuning checkpoints. Defaults to --out_dir if not specified')
    parser.add_argument('--lscratch', type=str, default=None, #required=True,
                        help='(optional) Path to lscratch for caching data. None by default, meaning do NOT use caching. If the empty string is provided, the memory is used.')
    parser.add_argument('--ntasks', type=int, default=None, #required=True,
                        help='(optional) Number of threads, either $SLURM_NTASKS or $SLURM_CPUS_PER_TASK. None by default.')
    parser.add_argument('--hps_index', type=int, #required=True,
                        help='(optional) Index of top')
    #parser.add_argument('--hps_datetime', type=str, #required=True,
    #                     help='(optional) Datetime for the file containing the top hyperparameters')
    

    # Tuner hyperparameter search arguments
    parser.add_argument('--objective', type=str, default='val_loss', #, required=True
                        help='Objective or loss functionor value to optimize, such as val_loss. See keras tuner for more information')
    parser.add_argument('--project_name_prefix', type=str, default='tornado_unet', #required=True,
                        help='Prefix to the project name for the tuner. Used as the prefix for the name of the sub-directory where the search results are stored. See keras tuner attribute project_name for more information')
    parser.add_argument('--max_trials', type=int, default=5, #required=True,
                        help='Number of trials (i.e. hyperparameter configurations) to try. See keras tuner for more information')
    parser.add_argument('--overwrite', action='store_true', #required=True,
                        help='Include to overwrite the tuner project directory. Otherwise, reload existing project of the same name if found. See keras tuner for more information')
    parser.add_argument('--max_retries_per_trial', type=int, default=0, #required=True,
                        help='Maximum number of times to retry a Trial if the trial crashed or the results are invalid. See keras tuner for more information')
    parser.add_argument('--max_consecutive_failed_trials', type=int, default=1, #required=True,
                        help='Maximum number of consecutive failed Trials. When this number is reached, the search will be stopped. A Trial is marked as failed when none of the retries succeeded. See keras tuner for more information')
    parser.add_argument('--executions_per_trial', type=int, default=1, #required=True,
                        help='Number of executions (training a model from scratch, starting from a new initialization) to run per trial (model configuration). See keras tuner for more information')
    parser.add_argument('--tuner_id', type=str, default=None,
                        help='Name identitfying the tuner. See Keras Tuner documentation')
    #parser.add_argument('-t', '--tuner', default='none', choices=['none', 'random', 'hyperband', 'bayesian', 'custom'],
    #                    help='Include flag to run the hyperparameter tuner. Otherwise load the top five previous models')
    tunersparsers = parser.add_subparsers(title='tuners', dest='tuner', help='tuner selection')
    #prsr_none = tunersparsers.add_parser('no_tuner', help='Hyperparameter search is not performed')
    prsr_rand = tunersparsers.add_parser('random', aliases=['rand'], 
                        help='Use random search')
    # BAYESOPT
    prsr_bayes = tunersparsers.add_parser('bayesian', aliases=['bayes'], help='Use bayesian optimization for tuning')
    prsr_bayes.add_argument('--num_initial_points', type=int, default=5, #required=True,
                        help='Number of points to initialize')
    prsr_bayes.add_argument('--alpha', type=float, default=0.0001, 
                        help='Value added to the diagonal of the kernel matrix during fitting. Represents expected amount of noise in the observed performances')
    prsr_bayes.add_argument('--beta', type=float, default=2.6, 
                        help='Balance exploration v exploitation. Larger is more explorative')
    # HYPERBAND
    prsr_hyper = tunersparsers.add_parser('hyperband', aliases=['hyper'], help='Use hyperband seach for tuning')
    prsr_hyper.add_argument('--max_epochs', type=int, default=4,
                        help='max number of epochs to train one model. recommended to set slightly higher than the expected epochs to convergence ')
    prsr_hyper.add_argument('--factor', type=int, default=3, 
                        help='Reduction factor for the number of epochs and number of models for each bracket')
    prsr_hyper.add_argument('--hyperband_iterations', type=int, default=1, 
                        help='At least 1. Number of times to iterate over the full Hyperband algorithm. One iteration will run approximately max_epochs * (math.log(max_epochs, factor) ** 2) cumulative epochs across all trials. It is recommended to set this to as high a value as is within your resource budget. ')
    # TODO
    prsr_custom = tunersparsers.add_parser('custom', help='Use custom tuner class')
    prsr_custom.add_argument('--args', type=dict, 
                        help='')

    # Architecture arguments
    parser.add_argument('--n_labels', type=int, default=1, #required=True,
                        help='Number of class labels (i.e. output nodes) for classification')

    # Training arguments
    parser.add_argument('--x_shape', type=tuple, default=(32, 32, 12), #required=True,
                        help='The size of the input patches')
    parser.add_argument('--y_shape', type=tuple, default=(32, 32, 1), #required=True,
                        help='The size of the output patches')
    parser.add_argument('--epochs', type=int, default=5, #required=True,
                        help='Number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=128, #=128,required=True,
                        help='Number of examples in each training batch')
    parser.add_argument('--class_weight', type=float, nargs='+', default=None, #type=list,
        help='Space delimited list of floats for the class weights. Set equal to -1 to use the inverse natural distribution in the training set for the weighting. For instance, if the ratio of nontor to tor is 9:1, then the class weight for nontor will be set to .1 and the weight for tor will be set to .9. Ex use: --class_weight .1 .9; Ex use: --class_weight=-1')
    parser.add_argument('--resample', type=float, nargs='+', default=None, #type=list,
        help='Space delimited list of floats for the weights for tf.Dataset.sample_from_datasets(). Ex use: --resample .9 .1')
    parser.add_argument('--lrate', type=float, default=1e-3, #required=True, 
                        help='Learning rate')

    # Callbacks
    # EarlyStopping
    parser.add_argument('--patience', type=int, default=8, #required=True,
                        help='Number of epochs with no improvement after which training will be stopped. See patience in EarlyStopping')
    parser.add_argument('--min_delta', type=float, default=0, #1e-3, #required=True,
                        help='Absolute change of less than min_delta will count as no improvement. See min_delta in EarlyStopping')

    # TODO: choices? tuned hyperparam
    '''
    parser.add_argument('--dataset', type=str, default='tor', #required=True,
                        choices=['tor', 'nontor_tor'],
                        help='dataset to use')
    '''

    parser.add_argument('--wandb_tags', type=str, nargs='+', default=None, #type=list,
                        help='Space delimited list of str tags for the wandb config tags. Ex use: --wandb_tags test crossval')
    parser.add_argument('--number_of_summary_trials', type=int, default=2,
                        help='The number of best hyperparameters to save.')
    parser.add_argument('--gpu', action='store_true',
                        help='Turn on gpu')
    parser.add_argument('--dry_run', action='store_true',
                        help='For testing. Execute with only 5 steps per epoch and print some extra debugging info')
    parser.add_argument('--nogo', action='store_true',
                        help='For testing. Do NOT execute any experiements')
    parser.add_argument('--save', type=int, default=0,
                        help='Specify data to save. 0 indicates save nothing. >=1 (but not 3) to save best hyperparameters and results in text format. >=2 (but not 3) save figures. 3 to save only the best model trained from the best hyperparameters. 4 to save textual results, figures, and the best model.') 
    parser.add_argument('--save_weights', action='store_true',
                        help='If saving the model, boolean flag indicating to save just the model weights when saving the model')
    return parser

def parse_args():
    '''
    Create and parse the command line args parser for the experiment
    @return: the parsed arguments object
    '''
    parser = create_argsparser()
    args = parser.parse_args()
    return args

def args2string(args):
    '''
    Translate the current set of arguments into a string
    @param args: the command line args object. See create_argsparser() for
            details about the command line arguments
            Command line arguments relevant to this method:


    @return: string with the formated command line arguments
    '''
    cdatetime = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    args_str = '' #f'{cdatetime}_'
    for arg, val in vars(args).items(): 
        if arg in ['in_dir', 'in_dir_val', 'in_dir_test', 'out_dir', 'out_dir_tuning',
                   'wandb_tags', 'overwrite', 'dry_run', 'nogo', 'save', 'lscratch']:
            continue
        if isinstance(val, bool):
            args_str += f'{arg}={val:1d}_'
        elif isinstance(val, int):
            if arg in ['max_retries_per_trial', 'max_consecutive_failed_trials', 'executions_per_trial', 
            'number_of_summary_trials', 'n_labels', 'factor', 'hps_index']:
                args_str += f'{arg}={val:02d}_'
            elif arg == 'patience':
                args_str += f'{arg}={val:03d}_'
            else:
                args_str += f'{arg}={val:04d}_'
        elif isinstance(val, float):
            args_str += f'{arg}={val:06f}_'
        elif isinstance(val, list) or isinstance(val, tuple):
            valstrs = [f'{i:03d}' if isinstance(i, int) else f'{i:.02f}'  for i in val]
            fullstr = '_'.join(valstrs)
            args_str += f'{arg}={fullstr}_'
        else: args_str += f'{arg}={val}_'

    args_str = args_str[:-1] # remove last underscore
    if args.dry_run:
        print(args_str, "\n")
    return cdatetime, args_str


if __name__ == "__main__":
    args = parse_args()
    cdatetime, argstr = args2string(args)
    args.cdatetime = cdatetime
    if args.dry_run: 
        print(cdatetime)
        print(argstr)
        print(args, "\n\n")

    # (OLD WAY) Grab select GPU(s)
    #if args.gpu: 
    #    print("Attempting to grab GPU")
    #    py3nvml.grab_gpus(num_gpus=1, gpu_select=[0])

    #    '''
    #    #os.get_env('CUDA_VISIBLE_DEVICES', None)
    #    physical_devices = tf.config.get_visible_devices('GPU')
    #    n_physical_devices = len(physical_devices)

    #    # Ensure all devices used have the same memory growth flag
    #    for physical_device in physical_devices:
    #        tf.config.experimental.set_memory_growth(physical_device, False)
    #    print(f'We have {n_physical_devices} GPUs\n')
    #    '''

    ndevices = 0
    devices = []
    if "CUDA_VISIBLE_DEVICES" in os.environ.keys():
        # Fetch list of allocated logical GPUs; numbered 0, 1, …
        devices = tf.config.get_visible_devices('GPU')
        ndevices = len(devices)

        # Set memory growth for each
        #config.gpu_options.allow_growth = True
        try:
            for device in devices:
                tf.config.experimental.set_memory_growth(device, True)
            print("Memory growth set")
        except Exception as err:
            print(err)
    else:
        # No allocated GPUs: do not delete this case!
        try: tf.config.set_visible_devices([], 'GPU')
        except Exception as err: print(err)

    devices_logical = tf.config.list_logical_devices('GPU')
    print(f'Visible {ndevices} devices {devices}. \nLogical devices {len(devices_logical)} {devices_logical}\n')
    print("GPUs (Phys. Devices) Available: ", tf.config.list_physical_devices('GPU'))

    #if args.cpus_per_task is not None:
    #    tf.config.threading.set_intra_op_parallelism_threads(args.cpus_per_task)
    #    tf.config.threading.set_inter_op_parallelism_threads(args.cpus_per_task)

    if ndevices == 0:
        print(f'Terminating run. nGPUs is {ndevices}')
        exit()

    if args.nogo:
        print('NOGO.')
        exit()

    ''' TODO
    if args.cpus_per_task is not None:
        tf.config.threading.set_intra_op_parallelism_threads(args.cpus_per_task)
        tf.config.threading.set_inter_op_parallelism_threads(args.cpus_per_task)
    '''


    PROJ_NAME_PREFIX = args.project_name_prefix
    PROJ_NAME = f'{PROJ_NAME_PREFIX}_{args.tuner}'
    
    tuner_dir = args.out_dir_tuning if not args.out_dir_tuning is None  else args.out_dir
    wandb_path = os.path.join(tuner_dir, f'{PROJ_NAME}_{cdatetime}_wandb')


    tuner, hypermodel = create_tuner(args, wandb_path=wandb_path, DB=args.dry_run) #, strategy=tf.distribute.MirroredStrategy())

    # Load and prepare datasets
    ds_train, ds_val, ds_test, train_val_steps, ds_train_og, ds_val_og = prep_data(
                args, n_labels=hypermodel.n_labels, sample_method='sample')
    
    # If a tuner is specified, run the hyperparameter search
    if not args.tuner is None:
        # Command line args to dict
        args_dict = copy.deepcopy(vars(args))
        tags = args_dict['wandb_tags']
        for k in ['in_dir', 'in_dir_val', 'in_dir_test', 'class_weight', 'resample',
                    'dry_run', 'nogo', 'overwrite', 'save', 'wandb_tags', 'out_dir', 
                    'out_dir_tuning']: #]: #'project_name_prefix']:
            args_dict.pop(k, None)

        execute_search(args, tuner, ds_train, X_val=ds_val, callbacks=[], 
                       train_val_steps=train_val_steps, cdatetime=cdatetime, 
                       DB=args.dry_run)

        # Report results
        print('\n=====================================================')
        print('=====================================================')

        # Init wandb summary run logging eval and figures
        run = wandb.init(project='unet_hypermodel_run0', config=args_dict, 
                         dir=wandb_path, tags=tags)
        del args_dict

        N_SUMMARY_TRIALS = args.number_of_summary_trials
        tuner.results_summary(N_SUMMARY_TRIALS)

        # Retrieve best hyperparams
        best_hps_obj = tuner.get_best_hyperparameters(num_trials=N_SUMMARY_TRIALS)
        best_hps = [hp.values for hp in best_hps_obj]
        #print("best_hp", best_hps[0].values)

        # Save best hyperparams
        df = pd.DataFrame(best_hps)
        df['args'] = [argstr] * N_SUMMARY_TRIALS
        df['tuner_id'] = tuner.tuner_id 
        df['tuner_directory'] = tuner.directory 
        df['tuner_project_name'] = tuner.project_name 
        dirpath = os.path.join(args.out_dir, PROJ_NAME)
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)
            print(f"Made dir {dirpath}")
        hp_fnpath = os.path.join(dirpath, f"{cdatetime}_hps.csv")
        #hp_fnpath = f"{args.out_dir}/{PROJ_NAME}/hps_{argstr}.csv"
        print(f"\nSaving top {N_SUMMARY_TRIALS:02d} hyperparameter")
        print(hp_fnpath)
        print(df)
        if args.save in [1, 2, 4]: #args.save > 0:
            print("Saving", hp_fnpath)
            df.to_csv(hp_fnpath)
        del df

        # Train with best hyperparameters
        print("\n-------------------------")
        print("Training Best Model")

        hp_index = args.hps_index if not args.hps_index is None  else 0
        BATCH_SIZE = args.batch_size
        FN_PREFIX = f"{cdatetime}_hp_model{hp_index:02d}"
        print(f"TOP HPs INDEX = {hp_index} ** ")

        model = tuner.hypermodel.build(best_hps_obj[hp_index])
        es = EarlyStopping(monitor=args.objective, patience=args.patience,  
                            min_delta=args.min_delta, restore_best_weights=True)
        H = model.fit(ds_train, validation_data=ds_val, 
                      #batch_size=BATCH_SIZE, 
                      steps_per_epoch=train_val_steps['steps_per_epoch'],
                      validation_steps=train_val_steps['val_steps'], 
                      epochs=args.epochs, 
                      callbacks=[es]) #, verbose=1)
        fname = os.path.join(dirpath, f"{FN_PREFIX}_learning_plot.png")
        _fg = plot_learning_loss(H, fname, save=(args.save in [2, 4])) #(args.save >= 2)
        #['loss', 'max_csi', 'auc_2', 'auc_3', 'binary_accuracy', 'val_loss', 'val_max_csi', 'val_auc_2', 'val_auc_3', 'val_binary_accuracy']
        #wandb.log({"plot_learning": wandb.Image(_fg)}) #pip install plotly
        wandb.log({"plot_learning": _fg}) #pip install plotly
        plt.close()
        del _fg

        if args.save >= 2:
            diagram_fnpath = os.path.join(dirpath, f"{FN_PREFIX}_architecture.png")
            print("Saving", diagram_fnpath)
            img_obj = plot_model(model, to_file=diagram_fnpath, show_dtype=True,
                                 show_shapes=True, expand_nested=False)
            from io import BytesIO
            from PIL import Image
            img_plt_model = Image.open(BytesIO(img_obj.data))
            arr_plt_model = np.asarray(img_plt_model)
            wandb.log({"plot_model": wandb.Image(arr_plt_model)})

        if args.save >= 3: 
            suffix = "_weights" if args.save_weights  else ""
            model_fnpath = os.path.join(dirpath, f"{FN_PREFIX}{suffix}.h5")
            hypermodel.save_model(model_fnpath, weights=args.save_weights, #argstr
                                  model=model, save_traces=True)

        # Predict with trained model
        print("\nPREDICTION")
        #print(" train size", get_dataset_size(ds_train))
        xtrain_preds = model.predict(ds_train_og, #steps=train_val_steps['steps_per_epoch'],#*BATCH_SIZE, 
                                     verbose=1, workers=2, use_multiprocessing=True)
        xval_preds = model.predict(ds_val_og, #steps=train_val_steps['val_steps'],
                                   verbose=1, workers=2, use_multiprocessing=True)
        #xtest_recon = best_model.predict(X_test)
        #print("FVAF::", fvaf(xtrain_recon, ds_train), fvaf(xval_recon, ds_val), fvaf(xtest_recon, ds_test))
        fname = os.path.join(dirpath, f"{FN_PREFIX}_preds_distr.png")
        if args.save in [2, 4]:
            sel0 = np.arange(0, xtrain_preds.size, 2)
            sel1 = np.arange(0, xval_preds.size, 2)
            _fg, _ = plot_predictions(xtrain_preds.ravel()[sel0], xval_preds.ravel()[sel1], 
                                      fname, save=True) #save>=2
            wandb.log({"plot_preds": wandb.Image(_fg)}) #
            plt.close()
            del _fg

        # Confusion Matrix
        def get_y(x, y):
            ''' Get y from tuple Dataset '''
            return y

        if args.class_weight is None:
            y_train = np.concatenate([y for x, y in ds_train_og]) #ds.map(get_y)
        else:
            y_train = np.concatenate([y for x, y, w in ds_train_og])
        y_val = np.concatenate([y for x, y in ds_val_og])

        print(f"ytrain shape: {y_train.shape} (true) {xtrain_preds.shape} (pred)")
        print(f"yval shape: {y_val.shape} (true) {xval_preds.shape} (pred)")
        threshs = np.linspace(0, 1, 51).tolist()
        tps, fps, fns, tns = contingency_curves(y_val, xval_preds, threshs)
        csis = compute_csi(tps, fns, fps) #tps / (tps + fns + fps)
        xi = np.argmax(csis)
        cutoff_probab = threshs[xi] # cutoff with heightest CSI
        print(f"Max CSI: {csis[xi]}  Thres: {cutoff_probab}  Index: {xi}")

        if args.save in [2, 4]:
            fname = os.path.join(dirpath, f"{FN_PREFIX}_confusion_matrix_train_val.png")
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            axs = axs.ravel()
            #plt.subplots_adjust(wspace=.1)
            cthresh = np.arange(0.05, 1.05, 0.05), 
            plot_confusion_matrix(y_train.ravel(), xtrain_preds.ravel(), 
                                  fname, p=cutoff_probab, thresh=cthresh, 
                                  fig_ax=(fig, axs[0]), save=False)        
            plot_confusion_matrix(y_val.ravel(), xval_preds.ravel(),  
                                  fname, p=cutoff_probab, thresh=cthresh, 
                                  fig_ax=(fig, axs[1]), save=True) #args.save >= 2
            wandb.log({"confusion_mtx": wandb.Image(fig)})
            plt.close(fig)
            del fig, axs

            # ROC
            fname = os.path.join(dirpath, f"{FN_PREFIX}_roc_train_val.png")
            fig, ax = plot_roc(y_train.ravel(), xtrain_preds.ravel(), fname, 
                            save=False, label='Train')
            plot_roc(y_val.ravel(), xval_preds.ravel(), fname, fig_ax=(fig, ax), 
                            save=True, label='Val', c='orange') #args.save in [2, 4] #args.save >= 2
            wandb.log({"plot_roc": fig}) 
            plt.close(fig)
            del fig, ax

            # PRC
            fname = os.path.join(dirpath, f"{FN_PREFIX}_prc_train_val.png")
            fig, ax = plot_prc(y_train.ravel(), xtrain_preds.ravel(), fname, 
                            save=False, label='Train')
            plot_prc(y_val.ravel(), xval_preds.ravel(), fname, fig_ax=(fig, ax), 
                            save=True, label='Val', c='orange') #args.save in [2, 4] #args.save >= 2
            wandb.log({"plot_prc": fig})
            plt.close(fig)
            del fig, ax
        
        # Evaluate trained model
        print("\nEVALUATION")
        train_eval = model.evaluate(ds_train_og, workers=2, use_multiprocessing=True)
        val_eval = model.evaluate(ds_val_og, workers=2, use_multiprocessing=True)
        if not ds_test is None:
            test_eval = model.evaluate(ds_test, workers=2, use_multiprocessing=True)

        # Construct evalutation pd.DataFrame
        metrics = H.history.keys()
        df_index = ['train', 'val']
        evals = [ {k: v for k, v in zip(metrics, train_eval)} ]
        evals.append( {k: v for k, v in zip(metrics, val_eval)} )
        if not ds_test is None:
            evals.append( {k: v for k, v in zip(metrics, test_eval)} )
            df_index.append('test')
        df_eval = pd.DataFrame(evals, index=df_index)
        print(df_eval)
        fname = os.path.join(dirpath, f"{FN_PREFIX}_eval.csv")
        if args.save in [1, 2, 4]: #args.save > 0
            print("Saving", fname)
            df_eval.to_csv(fname)
        del evals, df_eval

        if args.save in [2, 4]:
            # CSI Curve
            csithreshs = np.linspace(0, 1, 21)
            fname = os.path.join(dirpath, f"{FN_PREFIX}_csi_train_val.png")
            fig, ax = plot_csi(y_train.ravel(), xtrain_preds.ravel(), fname, 
                            threshs=csithreshs, label='Train', show_cb=False)
            plot_csi(y_val.ravel(), xval_preds.ravel(), fname, threshs=csithreshs, label='Val', 
                            color='orange', save=True, fig_ax=(fig, ax)) #args.save in [2, 4] #args.save >= 2
            wandb.log({"plot_csi": wandb.Image(fig)}) 
            plt.close(fig)
            del fig, ax

            # Reliability Curve
            fname = os.path.join(dirpath, f"{FN_PREFIX}_reliability_train_val.png")
            fig, ax = plot_reliabilty_curve(y_train.ravel(), xtrain_preds.ravel(),  
                                        fname, save=False, label='Train', strategy='uniform')
            plot_reliabilty_curve(y_val.ravel(), xval_preds.ravel(), fname, 
                                        fig_ax=(fig, ax), save=True, #args.save in [2, 4] #args.save >= 2
                                        label='Val', c='orange', strategy='uniform')
            wandb.log({"plot_reliabilty_curve": fig})
            plt.close(fig)
            del fig, ax


        '''
        # Retrieve best model
        print("\n-------------------------")
        best_model = tuner.get_best_models(num_models=1)[0]
        in_shape = ds_train.element_spec[0].shape
        best_model.build(input_shape=in_shape)
        Hb = best_model.fit(ds_train, validation_data=ds_val, callbacks=[es],
                            batch_size=args.batch_size, epochs=args.epochs)
        '''
    # Load the latest model
    else:
        PROJ_NAME_PREFIX = args.project_name_prefix
        PROJ_NAME = f'{PROJ_NAME_PREFIX}_{args.tuner}'

        #TODO
        cp_dir = args.out_dir_tuning if not args.out_dir_tuning is None  else args.out_dir
        cp_path = os.path.join(cp_dir, PROJ_NAME)
        latest = tf.train.latest_checkpoint(cp_path) 
        #latest = tf.train.latest_checkpoint(f'{args.out_dir}/{PROJ_NAME}') 
        #latest = tf.keras.models.load_model(cp_path, compile=False)

        # Load hyperparameters 
        #HyperParameters.Fixed(name, value, parent_name=None, parent_values=None)
        hps_file = args.in_hps if not args.in_hps is None  else args.out_dir
        df_hps = pd.read_csv(hps_file)
        best_hps = df_hps.drop(columns='args') #df_hps.iloc[0][:-1]
        #best_hp = df.iloc[0]
        #hypermodel.build(best_hps) 


    print('DONE.\n')
