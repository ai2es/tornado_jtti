"""
auhtor: Monique Shotande (monique.shotande a ou.edu)

UNet Hyperparameter Search using keras tuners. 
A keras_tuner.HyperModel subclass is defined as UNetHyperModel
that defines how to build various versions of the UNet models
"""

import os, io, sys, random, shutil
import time, datetime
from absl import app
from absl import flags
import argparse
import numpy as np
print("np version", np.__version__)
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
import tf.keras.backend as K
from tf.keras.utils import plot_model
from tf.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint 
from tf.keras.optimizers import Adam #, SGD, RMSprop, Adagrad
from tf.keras.metrics import AUC

from keras_tuner import HyperModel
from keras_tuner.tuners import RandomSearch, Hyperband, BayesianOptimization
#from tensorboard.plugins.hparams import api as hp

from tensorboard.plugins.hparams import api
from tensorboard.plugins.hparams import api_pb2
from tensorboard.plugins.hparams import summary as hp_summary

#GRAB GPU0
import py3nvml
py3nvml.grab_gpus(num_gpus=1, gpu_select=[0])

sys.path.append("/home/lydiaks2/tornado_project/")
from custom_losses import make_fractions_skill_score
from custom_metrics import MaxCriticalSuccessIndex
sys.path.append("/home/momoshog/Tornado")
from keras_unet_collection import models



class UNetHyperModel(HyperModel):
    """
    
    @param input_shape: tuple for the shape of the input data
    @param n_labels: number of class labels. If None, tune as a hyperparameter in 
                that can either be 1 or 2.
    @param name: prefix of the layers and model. Use keras.models.Model.summary to 
                identify the exact name of each layer. Empty by default
    @param tune_optimizer: Whether to tune the optimizer. False by default and uses
                Adam
    """
    def __init__(self, input_shape, n_labels=None, name='', tune_optimizer=False):
        self.input_shape = input_shape
        self.n_labels = n_labels if n_labels > 0 else 1
        self.name = name
        self.tune_optimizer = tune_optimizer

    def save_model(self, filepath, weights=True, model=None, save_traces=True):
        '''
        Save this or some other model
        @params filepath: path to save the model
        @params weights: Save just the model weights when True, otherwise save
                        the whole model.
        @params model: if None, save this model, otherwise save the provided model
        @params save_traces: 

        @return: True is the save occured, False otherwise
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
        # Set our random seed
        rng = None if seed is None else random.Random(seed)

        in_shape = self.input_shape

        # TODO: other options?
        latent_dim = hp.Int("latent_dim", min_value=28, step=2, max_value=256)
        #num = hp.Int("n_conv_down", min_value=3, step=1, max_value=10) # number of layers
        nfilters_per_layer6 = np.around(np.linspace(8, latent_dim, num=6)).astype(int).tolist()
        #nfilters_per_layer = list(range(8, latent_dim, 2))
        nfilters_per_layer2 = [2**i for i in range(3, int(np.log(latent_dim)))]

        # List defining number of conv filters per down/up-sampling block
        filter_num = hp.Choice("nfilters_per_layer", values=[nfilters_per_layer2, nfilters_per_layer6])

        kernel_size = hp.Choice("kernel_size", [3, 5, 7])
        
        # Number of convolutional layers per downsampling level
        stack_num_down = hp.Int("n_conv_down", min_value=1, step=1, max_value=5)
        # Configuration of downsampling (encoding) blocks
        pool = hp.Choice("pool_down", values=[False, 'ave', 'max'])
        # TODO: look up GELU and Snake
        activation = hp.Choice("in_activation", values=['PReLU', 'ELU', 'GELU', 'Snake']) #'ReLU', 'LeakyReLU'

        # Number of conv layers (after concatenation) per upsampling level
        stack_num_up = hp.Int("n_conv_up", min_value=1, step=1, max_value=5)
        # Configuration of upsampling (decoding) blocks
        unpool = hp.Choice("pool_up", values=[False, 'bilinear', 'nearest'])
        # Select appropriate output activation based on loss and number of output nodes
        #has_l2 = hp.Boolean("has_l2")
        n_labels = self.n_labels
        if n_labels is None:
            with hp.conditional_scope("n_labels", [None]): 
                n_labels = hp.Int("n_labels", min_value=1, step=2, max_value=2)
        loss = 'binary_focal_crossentropy'
        metrics = [MaxCriticalSuccessIndex(), AUC(curve='PR')] #"categorical_accuracy"
        if n_labels == 1: 
            output_activation = hp.Choice("out_activation", values=['Linear', 'Sigmoid', 'Snake'])
            metrics.append("binary_accuracy")
        else:
            with hp.conditional_scope("n_labels", [2]): 
                output_activation = hp.Choice("out_activation", values=['Linear', 'Softmax', 'Snake']) #'Sigmoid'
                loss = hp.Choice("loss", ["binary_focal_crossentropy", "categorical_crossentropy", "fractions_skill_score"])
                #tf.keras.losses.binary_focal_crossentropy
                #tf.keras.losses.categorical_crossentropy
                #make_fractions_skill_score(3, 2, c=1.0, cutoff=0.5, want_hard_discretization=False)
                #metrics = ["binary_accuracy", MaxCriticalSuccessIndex()]
                # TODO
                if loss == "binary_focal_crossentropy":
                    metrics.append(["binary_accuracy"])
                else:
                     metrics.append(["categorical_crossentropy", "categorical_accuracy"])
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
        optimizer = Adam(learning_rate=lr) if not self.tune_optimizer else []
        # TODO ?
        # hp.Choice("optimizer", values=[Adagrad(learning_rate=learning_rate),
        #                                    SGD(learning_rate=learning_rate),
        #                                    RMSprop(learning_rate=learning_rate)]

        # Build and return the model
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        return model

def create_parser():
    '''
    Create a command line parser for the XOR experiment
    '''
    parser = argparse.ArgumentParser(description='Tornado Model Hyperparameter Search',
                                     epilog='MoSho')
    ## TODO
    parser.add_argument('--in_dir', type=str, required=True,
                         help='Input directory where the data are stored')
    parser.add_argument('--out_dir', type=str, required=True,
                         help='Output directory for models, hyperparameters, etc.')
    
    parser.add_argument('-t', '--tuner', default='none', choices=['none', 'random', 'hyperband', 'bayesian', 'custom'],
                         help='Include flag to run the hyperparameter tuner. Otherwise load the top five previous models')

    # Tuner hyperparameter search arguments
    parser.add_argument('--max_trials', type=int, default=5, #required=True,
                         help='Number of trials (i.e. hyperparameter configurations) to try. See keras tuner for more information')
    parser.add_argument('--overwrite', action='store_true', #required=True,
                         help='Include to overwrite the project. Otherwise, reload existing project of the same name if found. See keras tuner for more information')
    parser.add_argument('--executions_per_trial', type=int, default=1, #required=True,
                         help='Number of executions (training a model from scratch, starting from a new initialization) to run per trial (model configuration). See keras tuner for more information')

    # Tuner BayesianOptimization arguments
    parser.add_argument('--num_initial_points', type=int, default=5, #required=True,
                         help='Number of initialization points for BayesianOptimization')

    # Architecture arguments num_classes
    parser.add_argument('--num_classes', type=int, default=1, #required=True,
                         help='Number of classes (i.e. output nodes) for classification')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=5, #required=True,
                         help='Number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=128, #required=True,
                         help='Number of examples in each training batch')
    parser.add_argument('--lrate', type=float, default=1e-3, #required=True,
                         help='Learning rate')

    parser.add_argument('--exp', type=int, default=0, #required=True,
                         help='Experiment index')
    parser.add_argument('--gpu', action='store_true',
                         help='turn on gpu')
    parser.add_argument('--dry_run', action='store_true',
                         help='For testing. Execute without running or saving data and verify output paths')
    return parser

def args2string(args):
    '''
    Translate the current set of arguments into a string
    @param args: Command line arguments

    @return: string with the command line arguments
    '''
    ctime = time.time()
    ctime = datetime.datetime.fromtimestamp(ctime).strftime("%Y_%b_%d_%H_%M_%S")

    #for arg in args: args_str += f'{}={}'

    return "max_trials=%02d_hidden=%02d" % (args.max_trials, args.num_initial_points)


parser = create_parser()
args = parser.parse_args()

hypermodel = UNetHyperModel(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)
#weights = dummy_loader(model_old_path)
#model_new = swin_transformer_model(...)
#model_new.set_weights(weights)

# Tuners RandomSearch, Hyperband, BayeOpt
MAX_TRIALS = args.max_trials 
BATCH_SIZE = args.batch_size 
NEPOCHS = args.epochs 

log_dir = ''

tuner_args = {
    'objective': 'val_accuracy',
    'max_retries_per_trial': 3,
    'max_consecutive_failed_trials': 3,
    'executions_per_trial': args.executions_per_trial, #3
    'logger': None, #TODO Optional instance of kerastuner.Logger class for streaming logs for monitoring.
    'tuner_id': None, #TODO Optional string, used as the ID of this Tuner.
    'overwrite': args.overwrite, #TODO: If False, reloads existing project of same name if found. Otherwise, overwrites the project.
    'directory': args.out_dir #TODO relative path to the working directory
}
    #'seed': None,
    #'hyperparameters': None,
    #'tune_new_entries': True,
    #'allow_new_entries': True,

# Select tuner
tuner = None
if args.tuner == 'random':
    tuner = RandomSearch(
        hypermodel,
        max_trials=MAX_TRIALS,
        project_name='tornado_unet', #TODO 
        **tuner_args
    )
elif args.tuner == 'hyperband':
    Hyperband(
        hypermodel,
        max_epochs=100, #TODO command line args. nteger, the maximum number of epochs to train one model. It is recommended to set this to a value slightly higher than the expected epochs to convergence 
        factor=3, #TODO command line args. Integer, the reduction factor for the number of epochs and number of models for each bracket
        hyperband_iterations=1, #TODO command line args. Integer, at least 1, the number of times to iterate over the full Hyperband algorithm. One iteration will run approximately max_epochs * (math.log(max_epochs, factor) ** 2) cumulative epochs across all trials. It is recommended to set this to as high a value as is within your resource budget. 
        **tuner_args
    )
elif args.tuner == 'bayesian':
    tuner = BayesianOptimization(
        hypermodel,
        max_trials=MAX_TRIALS, # # of hyperparameter combinations tested by tuner
        num_initial_points=2, #TODO command line args
        alpha=0.0001, #TODO command line args. value added to the diagonal of the kernel matrix during fitting. It represents the expected amount of noise in the observed performances
        beta=2.6, #TODO command line args. balancing factor of exploration and exploitation. The larger it is, the more explorative
        project_name='tornado_unet', #TODO prefix for files saved by this Tuner.
        **tuner_args
    )
#elif args.tuner == 'custom':

tuner.search_space_summary()

# If a tuner is specified, run the hyperparameter search
if not args.tuner == 'none':
    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    tb = TensorBoard(f"tuners/{args.tuner}") # tensorboard --logdir=bayes_opt
    #cp = ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_weights_only=True, save_freq=5*BATCH_SIZE)
    callbacks = [es, tb]

    # Perform the hyperparameter search
    print("[INFO] performing hyperparameter search...")
    tuner.search(X_train, X_train,
                validation_data=(X_val, X_val), #validation_split=.1
                batch_size=BATCH_SIZE, epochs=NEPOCHS, 
                callbacks=callbacks) 
# Load the latest model
else:
    latest = tf.train.latest_checkpoint(args.out_dir)

# Report results
N_SUMMARY_TRIALS = 5 #MAX_TRIALS
tuner.results_summary(N_SUMMARY_TRIALS)
best_hp = tuner.get_best_hyperparameters(num_trials=N_SUMMARY_TRIALS)
print(best_hp[0].values)

# Retrieve the best model.
best_model = tuner.get_best_models(num_models=1)[0]
best_model.summary()

'''plot_model(best_model, to_file=f"tuners/best_model__{args.tuner}.png",  
           show_dtype=True, show_shapes=True, expand_nested=False)
plot_model(best_model, to_file=f"tuners/best_model__{args.tuner}_expanded.png",  
           show_dtype=True, show_shapes=True, expand_nested=True)
'''

# Predict with best model
'''print("-------------------------")
print("PREDICTION")
xtrain_recon = best_model.predict(X_train, batch_size=BATCH_SIZE)
xval_recon = best_model.predict(X_val, batch_size=BATCH_SIZE)
xtest_recon = best_model.predict(X_test, batch_size=BATCH_SIZE)
print("FVAF::", fvaf(xtrain_recon, X_train), fvaf(xval_recon, X_val), fvaf(xtest_recon, X_test))

# Evaluate the best model
print("-------------------------")
print("EVALUATION")
res_train = best_model.evaluate(X_train, X_train)
res_val = best_model.evaluate(X_val, X_val)
res_test = best_model.evaluate(X_test, X_test)
print(res_train, res_val, res_test)
'''

'''
print("[INFO] training the best model...")
model = tuner.hypermodel.build(best_hp)
H = model.fit(X_train, X_train,
              validation_data=(X_val, X_val), batch_size=BS,
              epochs=NEPOCHS, callbacks=[es], verbose=1)
plot_learning_loss(H)
print(H.history.keys())

# evaluate the network
print("[INFO] evaluating network...")
print(best_hp.values)
xtrain_recon = model.predict(X_train) #, batch_size=BS)
xval_recon = model.predict(X_val) #, batch_size=BS)
xtest_recon = model.predict(X_test) #, batch_size=BS)
'''

