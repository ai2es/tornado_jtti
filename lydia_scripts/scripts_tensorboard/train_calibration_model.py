"""
author: Monique Shotande

Train a regression model to calibration U-Net model predictions

See create_argparser method for command line arguments

1. Loads or builds_then_fits hypermodel
2. Generate predictions for rain, val, and test sets
3. Fit calibration model on train set

"""

import re, os, sys, glob, argparse
from datetime import datetime, date, time
from dateutil.parser import parse as parse_date
import xarray as xr
print("xr version", xr.__version__)
import numpy as np
print("np version", np.__version__)
import pandas as pd
print("pd version", pd.__version__)
from netCDF4 import Dataset, date2num, num2date
from scipy import spatial
import wrf #wrf-python=1.3.2.5==py38h0e9072a_0
print("wrf-python version", wrf.__version__)
import metpy
import metpy.calc

import matplotlib.pyplot as plt
plt.rcParams.update({"text.usetex": True})

from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression, isotonic_regression
import pickle

import tensorflow as tf
print("tensorflow version", tf.__version__)
from tensorflow import keras
print("keras version", keras.__version__)
from keras_tuner import HyperParameters

# Working directory expected to be tornado_jtti/
sys.path.append("lydia_scripts")
from custom_losses import make_fractions_skill_score
from custom_metrics import MaxCriticalSuccessIndex
from scripts_data_pipeline.wofs_to_gridrad_idw import calculate_output_lats_lons
sys.path.append("../keras-unet-collection")
from keras_unet_collection.activations import * #import GELU
from scripts_tensorboard.unet_hypermodel import UNetHyperModel, prep_data, plot_reliabilty_curve, plot_predictions, plot_preds_hists
from tensorflow.keras.callbacks import EarlyStopping 

from evaluate_models import get_gridrad_storm_filenames, pair_gridrad_to_storms, fit_transform_predictions, get_hp_from_df, build_fit_hypermodel


def create_argsparser():
    ''' 
    Create command line arguments parser

    @param args_list: list of strings with command line arguments to override
            any received arguments. default value is None and args are not 
            overridden. not used in production. mostly used for unit testing.
            Element at index 0 should either be the empty string or the name of
            the python script

    @return: the argument parser object
    '''
    parser = argparse.ArgumentParser(description='Tornado GridRad Evaluation', epilog='AI2ES')

    parser.add_argument('--loc_model', type=str, required=True,
        help='Trained model directory or file path (i.e. file descriptor)')
    
    parser.add_argument('--in_dir', type=str, required=True, 
        help='Location of the Gridrad radar file(s)')
    parser.add_argument('--in_dir_labels', type=str, #required=True, 
        help='Location of the Gridrad storm label file(s). Required if loading unpatched GridRad data for --in_dir. Optional if loading patched tf.Dataset file')
    parser.add_argument('--in_dir_val', type=str, #required=True, 
        help='Location of the Gridrad val set file(s)')
    parser.add_argument('--in_dir_test', type=str, #required=True, 
        help='Location of the Gridrad test set file(s)')

    parser.add_argument('--is_tfdataset', action='store_true', #required=True, 
        help='Whether GridRad data provided is a tf Dataset or an xarray dataset')
    
    parser.add_argument('--dir_preds', type=str, required=True, 
        help='Directory to store the predictions. Prediction files are saved individually for each GridRad files. The prediction files are saved of the form: <GRIDRAD_FILENAME>_predictions.nc')
    parser.add_argument('--dir_figs', type=str,  
        help='Top level directory to save any corresponding figures.')
    
    parser.add_argument('--lscratch', type=str, default=None, #required=True,
                        help='(optional) Path to lscratch for caching data. None by default, meaning do NOT use caching. If the empty string is provided, the memory is used.')

    parser.add_argument('-p', '--patch_shape', type=tuple, default=(32,), #required=True, 
        help='Shape of patches. Can be empty (), 2D (xy,), 2D (x, y), or 3D (x, y, h) tuple. If empty tuple, patching is not performed. If tuple length is 1, the x and y dimension are the same. Ex: (32) or (32, 32).')
    
    # Training arguments
    parser.add_argument('--x_shape', type=int, nargs=3, default=(32, 32, 12), #required=True,
                        help='The size of the input patches')
    parser.add_argument('--y_shape', type=int, nargs=3, default=(32, 32, 1), #required=True,
                        help='The size of the output patches')
    parser.add_argument('--n_labels', type=int, default=1, #required=True,
                        help='Number of class labels (i.e. output nodes) for classification')
    parser.add_argument('--epochs', type=int, default=5, #required=True,
                        help='Number of epochs to train the model')
    parser.add_argument('--lrate', type=float, default=1e-3, #required=True, 
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128, #=128,required=True,
                        help='Number of examples in each training batch')
    parser.add_argument('--class_weight', type=float, nargs='+', default=None, #type=list,
                        help='Space delimited list of floats for the class weights. Set equal to -1 to use the inverse natural distribution in the training set for the weighting. For instance, if the ratio of nontor to tor is 9:1, then the class weight for nontor will be set to .1 and the weight for tor will be set to .9. Ex use: --class_weight .1 .9; Ex use: --class_weight=-1')
    parser.add_argument('--resample', type=float, nargs='+', default=None, #type=list,
                        help='Space delimited list of floats for the weights for tf.Dataset.sample_from_datasets(). Ex use: --resample .9 .1')
    
    parser.add_argument('--build_fit_hypermodel', action='store_true', #type=list,
                        help='Flag used to build and fit the hypermodel from the hyperparameters instead of loading a saved model')
    
    parser.add_argument('--fname_prefix', type=str, default='', 
                        help='Prefix used for filenames. Should match naming for the ML model')
    
    parser.add_argument('--method_reg', type=str, default='log', 
                        help='Type of regression to use for calibrating the predictions. either "log" for logistic regression or "iso" for isotonic regression')

    parser.add_argument('--hp_path', type=str, #required=True,
                        help='Path to the csv containing the top hyperparameters')
    parser.add_argument('--hp_idx', type=int, default=0,
                        help='Index indicating the row to use within the csv of the top hyperparameters')
    
    parser.add_argument('-w', '--write', type=int, default=0,
                        help='Write/save data and/or figures. Set to 0 to save nothing, set to 1 to only save calibration model (.pkl; python pickle file), set to 2 to only save the calibration and histogram figures, set to 3 to the model and all figures')
    parser.add_argument('-d', '--dry_run', action='store_true',
                        help='For testing and debugging. Execute without running models or saving data and display output paths')
    
    return parser

def parse_args(args_list=None):
    '''
    Create and parse the command line args parser

    @param args_list: list of strings with command line arguments to override
            any received arguments. default value is None and args are not 
            overridden. not used in production. mostly used for unit testing.
            Element at index 0 should either be the empty string or the name of
            the python script

    @return: the parsed arguments object
    '''
    parser = create_argsparser()
    args = parser.parse_args(args_list)

    args.x_shape = tuple(args.x_shape)
    args.y_shape = tuple(args.y_shape)
    return args



if __name__ == "__main__":
    args = parse_args()

    # Grab GPUs if possible
    ndevices = 0
    devices = []
    if "CUDA_VISIBLE_DEVICES" in os.environ.keys():
        # Fetch list of allocated logical GPUs; numbered 0, 1, …
        devices = tf.config.get_visible_devices('GPU')
        ndevices = len(devices)

        # Set memory growth for each
        try:
            for device in devices:
                tf.config.experimental.set_memory_growth(device, True)
            print("Memory growth set")
        except Exception as err:
            print(err)
    else:
        try: tf.config.set_visible_devices([], 'GPU')
        except Exception as err: print(err)

    devices_logical = tf.config.list_logical_devices('GPU')
    print(f'Visible {ndevices} devices {devices}. \nLogical devices {len(devices_logical)} {devices_logical}\n')
    print("GPUs (Phys. Devices) Available: ", tf.config.list_physical_devices('GPU'))


    # Load and prepare tf Datasets
    ds_train, ds_val, ds_test, train_val_steps, ds_train_og, ds_val_og = prep_data(
                args, n_labels=args.n_labels, sample_method='sample')

    hps = None
    model = None
    H = None

    if args.build_fit_hypermodel:
        # Construct model
        print(f"Building model")
        hps, model, H = build_fit_hypermodel(args, ds_train, ds_val, 
                                            train_val_steps) #, callbacks=[])
    else:
        # Or load model
        print(f"Loading model {args.loc_model}")
        fss_args = {'mask_size': 2, 'num_dimensions': 2, 'c': 1.0, 
                    'cutoff': 0.5, 'want_hard_discretization': False}
        fss = make_fractions_skill_score(**fss_args)
        model = keras.models.load_model(args.loc_model, 
                                        custom_objects={'fractions_skill_score': fss, 
                                            'MaxCriticalSuccessIndex': MaxCriticalSuccessIndex,
                                            'GELU': GELU})


    # Predict with trained model
    print("\nPREDICTION")
    xtrain_preds = model.predict(ds_train_og, #steps=train_val_steps['steps_per_epoch'], 
                                 verbose=1, workers=3, use_multiprocessing=True)
    xval_preds = model.predict(ds_val_og, #steps=train_val_steps['val_steps'],
                               verbose=1, workers=3, use_multiprocessing=True)
    if ds_test is not None:
        xtest_preds = model.predict(ds_test, verbose=1, workers=3, 
                                    use_multiprocessing=True)

    # Convert tf Dataset to numpy for input to calibration model
    if args.class_weight is None:
        y_train = np.concatenate([y for x, y in ds_train_og]) #ds.map(get_y)
    else:
        y_train = np.concatenate([y for x, y, w in ds_train_og])
    y_val = np.concatenate([y for x, y in ds_val_og])
    if ds_test is not None:
        y_test = np.concatenate([y for x, y in ds_test])

    ntrain = y_train.shape[0]
    nval = y_val.shape[0]
    ntest = y_test.shape[0]

    # Reshape predictions for input to calibration model
    y_train_flat = y_train.reshape(-1, 1) #.reshape(ntrain, -1)
    xtrain_preds_flat = xtrain_preds.reshape(-1, 1) #.reshape(ntrain, -1)

    y_val_flat = y_val.reshape(-1, 1) #.reshape(nval, -1)
    xval_preds_flat = xval_preds.reshape(-1, 1) #.reshape(nval, -1)

    y_test_flat = y_test.reshape(-1, 1) #.reshape(ntest, -1)
    xtest_preds_flat = xtest_preds.reshape(-1, 1) #.reshape(ntest, -1)

    print("shape", y_train_flat.shape, xtrain_preds_flat.shape)

    # Calibrate predictions
    reg_args = {}
    calib, ycalib, mu_acc = fit_transform_predictions(args, y_train_flat, 
                                                      xtrain_preds_flat, 
                                                      method=args.method_reg, 
                                                      **reg_args)
    # Save fit regression model
    fname = os.path.join(args.dir_preds, f'{args.fname_prefix}calibraion_model_{args.method_reg}.pkl')
    if args.dry_run: print(f"Calibration model {fname}")
    if args.write in [1, 3]:
        with open(fname, 'wb') as fid:
            pickle.dump(calib, fid)
        print(f"Saving calibration model {fname}")
    #READ:: #with open(fname, 'rb') as fid:
        #calib = pickle.load(fid)

    ycalib_val = calib.predict(xval_preds_flat)
    ycalib_test = calib.predict(xtest_preds_flat)


    # Reliabilty Curve
    plot_calib = (args.write in [2, 3]) #True
    fname = os.path.join(args.dir_preds, f"{args.fname_prefix}calibration_model_reliability.png")
    if args.dry_run: print(f"Calibration plot {fname}")
    if plot_calib:
        strat = 'uniform'
        kwargs = {'lw': 4}
        fig, ax = plot_reliabilty_curve(y_train.ravel(), xtrain_preds.ravel(), fname, 
                                        label='Train', strategy=strat, **kwargs, 
                                        save=False)
        plot_reliabilty_curve(y_val.ravel(), xval_preds.ravel(), fname, 
                            label='Val', c='orange', **kwargs, 
                            strategy=strat, fig_ax=(fig, ax), save=False) 
        plot_reliabilty_curve(y_test.ravel(), xtest_preds.ravel(), fname, 
                            label='Test', c='green', **kwargs, 
                            strategy=strat, fig_ax=(fig, ax), save=True) #>>SAVE
        print(f"Saving calibration figure {fname}")
        plt.close(fig)
        del fig, ax

        # Reliability Curve - Log Calibration
        fname = os.path.join(args.dir_preds, f"{args.fname_prefix}calibration_model_reliability_{args.method_reg}.png")
        if args.dry_run: print(f"Re-Calibration plot {fname}")
        kwargs = {'lw': 8}
        fig, ax = plot_reliabilty_curve(y_train.ravel(), ycalib, fname, **kwargs,
                                        label='Train', strategy=strat, save=False)
        kwargs = {'lw': 4}
        plot_reliabilty_curve(y_val.ravel(), ycalib_val, fname, **kwargs, 
                            label='Val', c='orange', 
                            strategy=strat, fig_ax=(fig, ax), save=False) 
        kwargs = {'lw': 2}
        plot_reliabilty_curve(y_test.ravel(), ycalib_test, fname, **kwargs, 
                            label='Test', c='green', 
                            strategy=strat, fig_ax=(fig, ax), save=True) #>>SAVE
        print(f"Saving calibration figure {fname}")
    
    # Prediction Distributions
    plot_pred_hists = (args.write in [2, 3]) #True
    fname = os.path.join(args.dir_preds, f"{args.fname_prefix}calibration_model_preds_hists_{args.method_reg}.png")
    if args.dry_run: print(f"Predictions histograms {fname}")
    if plot_pred_hists:
        Y = {'Train True': y_train.ravel()[:-1:20], 'Train Preds': xtrain_preds.ravel()[:-1:20],
            'Train Calib': ycalib[:-1:20],
            'Val True': y_val.ravel()[:-1:20], 'Val Preds': xval_preds.ravel()[:-1:20],
            'Val Calib': ycalib_val[:-1:20],
            'Test True': y_test.ravel()[:-1:20], 'Test Preds': xtest_preds.ravel()[:-1:20],
            'Test Calib': ycalib_test[:-1:20]
            }
        plot_preds_hists(Y, fname, use_seaborn=True, fig_ax=None, 
                        figsize=(10, 8), alpha=.5, save=True, dpi=160)
        #plot_predictions(y_train.ravel(), ycalib, fname, use_seaborn=True, 
        #                 figsize=(10, 8), alpha=.5, save=False, dpi=160) #y_train.ravel(), xtrain_preds.ravel()
        print(f"Saving preds hists {fname}")

    print("DONE.")
