"""
author: Monique Shotande (monique.shotande a ou.edu)

UNet Hyperparameter Search using keras tuners. 
A keras_tuner.HyperModel subclass is defined as UNetHyperModel
that defines how to build various versions of the UNet models

Expected working directory is tornado_jtti/

GridRad /ourdisk/hpc/ai2es/tornado/gridrad_gridded/ ??
/ourdisk/hpc/ai2es/tornado/storm_mask_unet/ ??

Windows tensorboard
python -m tensorboard.main --logdir=[PATH_TO_LOGDIR] [--port=6006]

"""

import os, io, sys, random, shutil
import pickle
import time, datetime
#from absl import app
#from absl import flags
import argparse
import numpy as np
print("np version", np.__version__)
import pandas as pd
print("pd version", pd.__version__)
import xarray as xr 
print("xr version", xr.__version__)
#import scipy
#print("scipy version", scipy.__version__)
#import seaborn as sns
#print(f"seaborn {sns.__version__}")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits import mplot3d
import matplotlib
font = {#'family' : 'normal',
        #'weight' : 'bold',
        'size'   : 14}
matplotlib.rc('font', **font)

import tensorflow as tf
print("tensorflow version", tf.__version__)
from tensorflow import keras
print("keras version", keras.__version__)
import tensorflow.keras.backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint 
from tensorflow.keras.optimizers import Adam #, SGD, RMSprop, Adagrad
from tensorflow.keras.metrics import AUC

from keras_tuner import HyperModel
from keras_tuner.tuners import RandomSearch, Hyperband, BayesianOptimization
#from tensorboard.plugins.hparams import api as hp
from keras_tuner.engine import tuner as tuner_module
from keras_tuner.engine import oracle as oracle_module

from tensorboard.plugins.hparams import api
from tensorboard.plugins.hparams import api_pb2
from tensorboard.plugins.hparams import summary as hp_summary

import py3nvml

#sys.path.append("../")
sys.path.append("lydia_scripts")
from custom_losses import make_fractions_skill_score
from custom_metrics import MaxCriticalSuccessIndex
#sys.path.append("../../../keras-unet-collection")
sys.path.append("../keras-unet-collection")
from keras_unet_collection import models
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
        #TODO expection handling
        self.input_shape = input_shape
        #TODO expection handling
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

        # TODO: other options?
        latent_dim = hp.Int("latent_dim", min_value=28, step=2, max_value=256)
        #num = hp.Int("n_conv_down", min_value=3, step=1, max_value=10) # number of layers
        nfilters_per_layer6 = np.around(np.linspace(8, latent_dim, num=6)).astype(int).tolist()
        #nfilters_per_layer = list(range(8, latent_dim, 2))
        #nfilters_per_layer2 = [2**i for i in range(3, int(np.log2(latent_dim)))]
        nfilters_per_layer2 = np.logspace(3, np.log2(latent_dim), num=6, endpoint=True, base=2, dtype=int)

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
        # TODO: look up GELU and Snake
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
            output_activation = hp.Choice("out_activation", values=['Sigmoid', 'Snake'])
            metrics.append("binary_accuracy")
        else:
            with hp.conditional_scope("n_labels", ['>=2']): 
                #TODO: keras.activations.linear
                output_activation = hp.Choice("out_activation", values=['Softmax', 'Snake']) #'Sigmoid'
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
        unet_type = hp.Choice("unet_type", values=['unet_2d', 'unet_plus_2d', 'unet_3plus_2d'])
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

        # Optimization
        lr = hp.Choice("learning_rate", values=[1e-3, 1e-4, 1e-5, 1e-6])
        #hp.Float('learning_rate',min_value=1e-4, max_value=1e-2, sampling='LOG', default=1e-3)
        optimizer = Adam(learning_rate=lr) #TODO (MAYBE): if not self.tune_optimizer else hp.Choice("optimizer", values=[Adagrad(learning_rate=lr), SGD(learning_rate=lr), RMSprop(learning_rate=lr)])
        # TODO ?
        # hp.Choice("optimizer", values=[Adagrad(learning_rate=lr),
        #                                    SGD(learning_rate=lr),
        #                                    RMSprop(learning_rate=lr)]

        # Build and return the model
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        if self.DB: model.summary()
        return model

def create_tuner(args, strategy=None, DB=1, **kwargs):
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

    tuner_args = {
        'distribution_strategy': strategy, #TODO 
        'objective': args.objective, #'val_MaxCriticalSuccessIndex', name of objective to optimize (whether to minimize or maximize is automatically inferred for built-in metrics)
        'max_retries_per_trial': args.max_retries_per_trial,
        'max_consecutive_failed_trials': args.max_consecutive_failed_trials,
        'executions_per_trial': args.executions_per_trial, #3
        'logger': None, #TODO Optional instance of kerastuner.Logger class for streaming logs for monitoring.
        'tuner_id': None, #TODO Optional string, used as the ID of this Tuner.
        'overwrite': args.overwrite, #TODO: If False, reload existing project. Otherwise, overwrite the project.
        'directory': tuner_dir #args.out_dir #TODO: relative path to working dir
    }
        #'seed': None,
        #'hyperparameters': None,
        #'tune_new_entries': True,
        #'allow_new_entries': True,

    MAX_TRIALS = args.max_trials 
    PROJ_NAME_PREFIX = args.project_name_prefix
    PROJ_NAME = f'{PROJ_NAME_PREFIX}_{args.tuner}'

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
        tuner = Hyperband(
            hypermodel,
            max_epochs=args.max_epochs, #10, #max train epochs per model. recommended slightly higher than expected epochs to convergence 
            factor=args.factor, #3, #int reduction factor for epochs and number of models per bracket
            hyperband_iterations=args.hyperband_iterations, #2, #>=1,  number of times to iterate over full Hyperband algorithm. One iteration will run approximately max_epochs * (math.log(max_epochs, factor) ** 2) cumulative epochs across all trials. set as high a value as is within your resource budget
            project_name=PROJ_NAME,
            **tuner_args
        )
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
    #TODO
    elif args.tuner == 'custom': 
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

    if DB:
        print(' ')
        print('==============================')
        tuner.search_space_summary()
        print(' ')

    return tuner, hypermodel

def execute_search(args, tuner, X_train, X_val=None, 
                   callbacks=None, DB=0, **kwargs):
    ''' TODO
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
            EarlyStopping and Tensorboard. (TODO: ModelCheckpoint)
    @param DB: debug flag to print the resulting hyperparam search space
    @param kwargs: additional keyword arguments
    '''
    BATCH_SIZE = args.batch_size 
    NEPOCHS = args.epochs 
    PROJ_NAME_PREFIX = args.project_name_prefix
    PROJ_NAME = f'{PROJ_NAME_PREFIX}_{args.tuner}'
    
    tuner_dir = args.out_dir_tuning if not args.out_dir_tuning is None  else args.out_dir

    if callbacks is None:
        # TODO: separate arg for objective and monitor?
        es = EarlyStopping(monitor=args.objective, #start_from_epoch=10, 
                            patience=args.patience, min_delta=args.min_delta, 
                            restore_best_weights=True)
        tb_path = os.path.join(tuner_dir, f'{PROJ_NAME}_tb')
        tb = TensorBoard(tb_path, histogram_freq=5) #--logdir=
        #cp = ModelCheckpoint(filepath=f"{tuner_dir}/checkpoints/{PROJ_NAME}', verbose=1, save_weights_only=True, save_freq=5*BATCH_SIZE)
        #manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=3)
        callbacks = [es, tb]

    # Perform the hyperparameter search
    #https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
    # TODO: dataset split
    print("\nExecuting hyperparameter search...")
    tuner.search(X_train, 
                validation_data=X_val, 
                validation_batch_size=None, 
                batch_size=BATCH_SIZE, epochs=NEPOCHS, 
                shuffle=False, callbacks=callbacks,
                steps_per_epoch=5 if DB else None, #verbose=2, #max_queue_size=10, workers=1, 
                use_multiprocessing=False) 

# TODO
def load_data(args):
    pass

def prep_data(args, n_labels=None, DB=1):
    """ 
    Load and prepare the data.
    @param args: the command line args object. See create_argsparser() for
            details about the command line arguments
            Command line arguments relevant to this method:

    @param n_labels: number of class labels. If None, tune as a hyperparameter in 
                that can either be 1 or 2.

    @return: tuple with the training and validation sets as Tensorflow Datasets
    """ 
    # Dataset size
    #height = args.elevation
    x_shape = (None, *args.x_shape) #(None, 32, 32, 12)
    x_shape_val = args.x_shape #(32, 32, 12)
    
    y_shape = (None, *args.y_shape) #(None, 32, 32, 1)
    y_shape_val = args.y_shape

    #if loss == 'binary_focal_crossentropy':
    if n_labels == 1:        
        #y_shape = (None, *args.y_shape) #(None, 32, 32, 1)
        specs = (tf.TensorSpec(shape=x_shape, dtype=tf.float64, name='X'), 
                     tf.TensorSpec(shape=y_shape, dtype=tf.int64, name='Y'))

        #y_shape_val = args.y_shape #(32, 32, 1)
        specs_val = (tf.TensorSpec(shape=x_shape_val, dtype=tf.float64, name='X'), 
                         tf.TensorSpec(shape=y_shape_val, dtype=tf.int64, name='Y'))

        # Pick out the correct dataset paths
        if args.dataset == 'tor':
            path = "int_tor"
        elif args.dataset == 'nontor_tor':
            path = "int_nontor_tor"
        else: raise ValueError(f"Arguments Error: Data set type must be either tor or nontor_tor but was {args.dataset}")
    else:
        # Define the dataset size
        #y_shape = (None, 32, 32, 2)
        specs = (tf.TensorSpec(shape=x_shape, dtype=tf.float64, name='X'), 
                     tf.TensorSpec(shape=y_shape, dtype=tf.float32, name='Y'))

        #y_shape_val = y_shape[1:] #(32, 32, 2)
        specs_val = (tf.TensorSpec(shape=x_shape_val, dtype=tf.float64, name='X'), 
                         tf.TensorSpec(shape=y_shape_val, dtype=tf.float32, name='Y'))

    # Pick out the correct dataset paths
    #hparams[HP_DATA_PATCHES_TYPE]
    if args.dataset == 'tor':
        path = "onehot_tor"
    elif args.dataset == 'nontor_tor':
        path = "onehot_nontor_tor"
    else: raise ValueError(f"Arguments Error: Data set type must be either tor or nontor_tor but was {args.dataset}")
        
    # Read tensorflow datasets
    '''
    '/ourdisk/hpc/ai2es/tornado/learning_patches/tensorflow/3D_light/training_" + path + '/training_ZH_only.tf'
    '/ourdisk/hpc/ai2es/tornado/learning_patches/tensorflow/3D_light/validation_' + path + '/validation1_ZH_only.tf'
    '''

    ds_train = tf.data.Dataset.load(args.in_dir, specs)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
    #xy_train = np.array(list(ds_train.as_numpy_iterator()))
    #ds_train = ds_train.map(drop_dim, num_parallel_calls=tf.data.AUTOTUNE)
    ds_val = tf.data.Dataset.load(args.in_dir_val, specs_val)
    ds_val = ds_val.batch(args.batch_size)
    ds_val = ds_val.prefetch(tf.data.AUTOTUNE)
    if DB:
        print("Training Dataset Specs:", specs)
        print("Validation Dataset Specs:", specs_val)

    return (ds_train, ds_val)

def fvaf(y_true, y_pred):
    ''' TODO
    Fraction of variance accounted for (FVAF) ranges (−inf, 1]. 
    1 FVAF represents a total reconstruction. 
    0 represents a reconstruction that's as good as using the average of the signal as predictor
    negative FVAF even worse reconstructions than using the average signal as the predictor
    1 - MSE / VAR =
    1 - (sum[(y_true - y_pred)**2]) / (sum[(y_true - y_mean)**2])
    '''
    tf_mse = tf.keras.losses.MeanSquaredError()
    MSE = tf_mse(y_true, y_pred).numpy()
    VAR = np.var(y_true.flatten()) #tf.math.reduce_variance(y_true)
    return 1. - MSE / VAR

def plot_learning_loss(history, fname, save=False, dpi=220):
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

def plot_predictions(y_preds, y_preds_val, fname, use_seaborn=True, figsize=(12, 8), alpha=.5, save=False, dpi=220):
    '''
    '''
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    axs = axs.ravel()

    if use_seaborn:
        from seaborn import histplot

        Y = {'Train': y_preds, 'Val': y_preds_val}

        histplot(data=Y, stat='probability', label='Train Set', legend=True, 
                 ax=axs[0], alpha=alpha, common_norm=False)
        axs[0].set_xlabel('Tornado Predicted Probability')
        axs[0].set_xlim([0, 1])

        histplot(data=Y, stat='probability', label='Train Set', legend=True, 
                 ax=axs[1], alpha=alpha, common_norm=False, cumulative=True, 
                 element="step", fill=False)
        axs[1].set_xlabel('Tornado Predicted Probability')
        axs[1].set_xlim([0, 1])
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
        axs[0].legend()

        axs[1].hist(y_train, density=True, cumulative=True, label='Train Set', histtype='step') #, norm=colors.LogNorm())
        axs[1].set_xlabel('Probability')
        axs[1].set_ylabel('cumulative density')
        axs[1].set_xlim([0, .6])
        axs[1].legend()

        # VAL SET
        y_val = y_preds_val.ravel()
        axs[2].hist(y_val, density=True, label='Val Set') #, norm=colors.LogNorm())
        axs[2].set_xlabel('Probability')
        axs[2].set_ylabel('density')
        axs[2].set_xlim([0, 1])
        axs[2].legend()

        axs[3].hist(y_val, density=True, cumulative=True, label='Val Set', histtype='step') #, norm=colors.LogNorm())
        axs[3].set_xlabel('Probability')
        axs[3].set_ylabel('cumulative density')
        axs[3].set_xlim([0, .6])
        axs[3].legend()

    plt.suptitle("Tornado Prediction Probabilities")

    if save:
        print("Saving prediction histograms")
        print(fname)
        plt.savefig(fname, dpi=dpi)

    return fig, axs

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
    parser.add_argument('--out_dir', type=str, required=True,
                         help='Output directory for results, models, hyperparameters, etc.')
    parser.add_argument('--out_dir_tuning', type=str, #required=True,
                         help='(optional) Output directory for training and tuning checkpoints. Defaults to --out_dir if not specified')
    

    # Tuner hyperparameter search arguments
    parser.add_argument('--objective', type=str, default='val_loss', #required=True,
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
    parser.add_argument('--lrate', type=float, default=1e-3, #required=True,
                         help='Learning rate')

    # Callbacks
    # EarlyStopping
    parser.add_argument('--patience', type=int, default=10, #required=True,
                         help='Number of epochs with no improvement after which training will be stopped. See patience in EarlyStopping')
    parser.add_argument('--min_delta', type=float, default=1e-4, #required=True,
                         help='Absolute change of less than min_delta will count as no improvement. See min_delta in EarlyStopping')

    # TODO: choices? tuned hyperparam
    parser.add_argument('--dataset', type=str, default='tor', #required=True,
                        choices=['tor', 'nontor_tor'],
                        help='dataset to use')
    
    parser.add_argument('--number_of_summary_trials', type=int, default=2,
                         help='The number of best hyperparameters to save.')
    parser.add_argument('--gpu', action='store_true',
                         help='Turn on gpu')
    parser.add_argument('--dry_run', action='store_true',
                         help='For testing. Execute without running or saving data and verify output paths')
    parser.add_argument('--nogo', action='store_true',
                         help='For testing. Do NOT execute any experiements')
    parser.add_argument('--save', type=int, default=0,
                         help='Specify data to save. 0 indicates save nothing. >=1 to save best hyperparameters and results in text format. >=2 save figures')
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
        if arg in ['in_dir', 'in_dir_val', 'out_dir', 'out_dir_tuning',
                   'project_name_prefix', 'overwrite', 'dry_run', 'nogo', 'save']:
            continue
        if isinstance(val, bool):
            args_str += f'{arg}={val:1d}_'
        elif isinstance(val, int):
            if arg in ['max_retries_per_trial', 'max_consecutive_failed_trials', 'executions_per_trial', 
            'number_of_summary_trials', 'n_labels', 'factor']:
                args_str += f'{arg}={val:02d}_'
            elif arg == 'patience':
                args_str += f'{arg}={val:03d}_'
            else:
                args_str += f'{arg}={val:04d}_'
        elif isinstance(val, float):
            args_str += f'{arg}={val:06f}_'
        elif isinstance(val, list) or isinstance(val, tuple):
            valstrs = [f'{i:03d}' for i in val]
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

    # Grab select GPU(s)
    if args.gpu: py3nvml.grab_gpus(num_gpus=1, gpu_select=[0])
    physical_devices = tf.config.get_visible_devices('GPU')
    n_physical_devices = len(physical_devices)
    # make sure all devices you are using have the same memory growth flag
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, False)
    # Verify number of GPUs
    print(f'We have {n_physical_devices} GPUs\n')

    if args.nogo:
        print('NOGO.')
        exit()

    tuner, hypermodel = create_tuner(args, DB=args.dry_run, 
                                     strategy=tf.distribute.MirroredStrategy())

    ds_train, ds_val = prep_data(args, n_labels=hypermodel.n_labels)

    PROJ_NAME_PREFIX = args.project_name_prefix
    PROJ_NAME = f'{PROJ_NAME_PREFIX}_{args.tuner}'

    # If a tuner is specified, run the hyperparameter search
    if not args.tuner is None:
        execute_search(args, tuner, ds_train, X_val=ds_val, 
                       callbacks=None, DB=args.dry_run)

        # Report results
        print('\n=====================================================')
        print('\n=====================================================')
        N_SUMMARY_TRIALS = args.number_of_summary_trials #5 #MAX_TRIALS
        tuner.results_summary(N_SUMMARY_TRIALS)

        # Retrieve best hyperparams
        best_hps_obj = tuner.get_best_hyperparameters(num_trials=N_SUMMARY_TRIALS)
        best_hps = [hp.values for hp in best_hps_obj]
        #print("best_hp", best_hps[0].values)

        # Save best hyperparams
        df = pd.DataFrame(best_hps)
        df['args'] = [argstr] * N_SUMMARY_TRIALS
        dirpath = os.path.join(args.out_dir, PROJ_NAME)
        hp_fnpath = os.path.join(dirpath, f"hps_{cdatetime}.csv")
        #hp_fnpath = f"{args.out_dir}/{PROJ_NAME}/hps_{argstr}.csv"
        print(f"\nSaving top {N_SUMMARY_TRIALS:02d} hyperparameter")
        print(hp_fnpath)
        # Display entire dataframe
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        print(df)
        if args.save > 0:
            print("Saving", hp_fnpath)
            df.to_csv(hp_fnpath)

        # Train with best hyperparameters
        BATCH_SIZE = args.batch_size
        print("\n-------------------------")
        print("Training Best Model")
        model = tuner.hypermodel.build(best_hps_obj[0])
        es = EarlyStopping(monitor=args.objective, patience=args.patience,  
                            min_delta=args.min_delta, restore_best_weights=True)
        H = model.fit(ds_train, validation_data=ds_val, 
                      batch_size=BATCH_SIZE, epochs=args.epochs, 
                      callbacks=[es]) #, verbose=1)
        fname = os.path.join(dirpath, f"hp_model00_{cdatetime}_learning_plot.png")
        plot_learning_loss(H, fname, save=(args.save >= 2))
        #print(H.history.keys())
        #['loss', 'max_csi', 'auc_2', 'auc_3', 'binary_accuracy', 'val_loss', 'val_max_csi', 'val_auc_2', 'val_auc_3', 'val_binary_accuracy']

        if args.save >= 2:
            diagram_fnpath = os.path.join(dirpath, f"hp_model00_{cdatetime}_architecture.png")
            print("Saving", diagram_fnpath)
            plot_model(model, to_file=diagram_fnpath, show_dtype=True,  
                    show_shapes=True, expand_nested=False)

        #model_fnpath = os.path.join(dirpath, f"hp_model00_{cdatetime}.h5")
        #hypermodel.save_model(model_fnpath, weights=True, #argstr
        #                      model=model, save_traces=True)

        # Predict with trained model
        print("\nPREDICTION")
        xtrain_preds = model.predict(ds_train)
        xval_preds = model.predict(ds_val)
        #xtest_recon = best_model.predict(X_test, batch_size=BATCH_SIZE)
        #print("FVAF::", fvaf(xtrain_recon, X_train), fvaf(xval_recon, X_val), fvaf(xtest_recon, X_test))
        fname = os.path.join(dirpath, f"hp_model00_{cdatetime}_preds_distr.png")
        plot_predictions(xtrain_preds.ravel(), xval_preds.ravel(), fname, save=(args.save >= 2))

        # Evaluate trained model
        print("\nEVALUATION")
        train_eval = model.evaluate(ds_train)
        val_eval = model.evaluate(ds_val)
        #xtest_recon = model.evaluate(ds_test)
        metrics = H.history.keys()
        evals = [ {k: v for k, v in zip(metrics, train_eval)} ]
        evals.append( {k: v for k, v in zip(metrics, val_eval)} )
        df_eval = pd.DataFrame(evals, index=['train', 'val'])
        print(df_eval)
        fname = os.path.join(dirpath, f"hp_model00_{cdatetime}_eval.csv")
        if args.save > 0: 
            print("Saving", fname)
            df_eval.to_csv(fname)

        # TODO: CSI plot
        # TODO: reliablitiy curve
        # TODO: performance_plot
        # TODO: ROC
        # TODO: ROC - PR


        '''
        # Retrieve best model
        print("\n-------------------------")
        best_model = tuner.get_best_models(num_models=1)[0]
        in_shape = ds_train.element_spec[0].shape
        best_model.build(input_shape=in_shape)
        Hb = best_model.fit(ds_train, validation_data=ds_val, callbacks=[es],
                            batch_size=args.batch_size, epochs=args.epochs)

        # Save best model from hyperparam search
        model_fnpath = os.path.join(dirpath, f"model00_{cdatetime}.h5")
        print(f"\nSaving top model")
        print(model_fnpath)
        best_model.summary()
        hypermodel.save_model(model_fnpath, weights=True, #argstr
                              model=best_model, save_traces=True)

        # Save diagram of model architecture
        diagram_fnpath = os.path.join(dirpath, f"model00_{cdatetime}.png")
        plot_model(best_model, to_file=diagram_fnpath,  
                    show_dtype=True, show_shapes=True, expand_nested=False)
        # REDUNDANT Save expanded diagram
        #diagram_fnpath = os.path.join(dirpath, f"model00_{cdatetime}_expanded.png")
        #plot_model(best_model, to_file=diagram_fnpath,  
        #            show_dtype=True, show_shapes=True, expand_nested=True)
        '''
    # Load the latest model
    else:
        #TODO
        cp_dir = args.out_dir_tuning if not args.out_dir_tuning is None  else args.out_dir
        cp_path = os.path.join(cp_dir, PROJ_NAME)
        #latest = tf.train.latest_checkpoint(cp_path) 
        #latest = tf.train.latest_checkpoint(f'{args.out_dir}/{PROJ_NAME}') 
        #f'{args.out_dir}/checkpoints/{PROJ_NAME}'
        #latest = tf.keras.models.load_model(cp_path, compile=False)
        #latest_hps = df.read_csv(hp_fnpath)
        #best_hp = df.iloc[0]


    # TODO: Load test set

    '''
    # Predict with latest model
    print("-------------------------")
    print("PREDICTION")
    xtrain_recon = best_model.predict(ds_train, batch_size=BATCH_SIZE)
    xval_recon = best_model.predict(ds_val, batch_size=BATCH_SIZE)
    xtest_recon = best_model.predict(ds_test, batch_size=BATCH_SIZE)
    #print("FVAF::", fvaf(xtrain_recon, ds_train), fvaf(xval_recon, ds_val), fvaf(xtest_recon, ds_test))

    # Evaluate the best model
    print("-------------------------")
    print("EVALUATION")
    res_train = best_model.evaluate(X_train, X_train)
    res_val = best_model.evaluate(X_val, X_val)
    res_test = best_model.evaluate(X_test, X_test)
    print(res_train, res_val, res_test)
    '''

    print('DONE.\n')
