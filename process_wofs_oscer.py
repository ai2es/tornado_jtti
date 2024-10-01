import os, glob, json, time, argparse, traceback, datetime
from multiprocessing.pool import Pool
from real_time_scripts import preds_to_msgpk_oscer
import pandas as pd
import warnings
import logging
import subprocess
warnings.filterwarnings("ignore")


def create_argsparser(args_list=None):
    ''' 
    Create command line arguments parser

    @param args_list: list of strings with command line arguments to override
            any received arguments. default value is None and args are not 
            overridden. not used in production. mostly used for unit testing.
            Element at index 0 should either be the empty string or the name of
            the python script

    @return: the argument parser object
    '''
    if not args_list is None:
        sys.argv = args_list
        
    parser = argparse.ArgumentParser(description='Tornado Prediction end-to-end from raw WoFS data', epilog='AI2ES')

    # WoFS file(s) path 
    parser.add_argument('--loc_wofs', type=str, required=True, 
        help='Location of the WoFS file(s). Can be a path to a single file or a directory to several files')
    
    parser.add_argument('--datetime_format', type=str, required=True, default="%Y-%m-%d_%H:%M:%S",
        help='Date time format string used in the WoFS file name. See python datetime module format codes for more details (https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes). ')
    parser.add_argument('--filename_prefix', type=str, 
        help='Prefix used in the WoFS file name')

    parser.add_argument('--dir_preds', type=str, required=True, 
        help='Directory to store the predictions. Prediction files are saved individually for each WoFS files. The prediction files are saved of the form: <WOFS_FILENAME>_predictions.nc')
    parser.add_argument('--dir_patches', type=str,  
        help='Directory to store the patches of the interpolated WoFS data. The files are saved of the form: <WOFS_FILENAME>_patched_<PATCH_SHAPE>.nc. This field is optional and mostly for testing')
    parser.add_argument('--dir_figs', type=str,  
        help='Top level directory to save any corresponding figures.')
    parser.add_argument('-p', '--patch_shape', type=tuple, default=(32,), #required=True, 
        help='Shape of patches. Can be empty (), 2D (xy,), 2D (x, y), or 3D (x, y, h) tuple. If empty tuple, patching is not performed. If tuple length is 1, the x and y dimension are the same. Ex: (32) or (32, 32).')
    parser.add_argument('--with_nans', action='store_true', 
        help='Set flag such that data points with reflectivity=0 are stored as NaNs. Otherwise store as normal floats. It is recommended to set this flag')
    parser.add_argument('-Z', '--ZH_only', action='store_true',  
        help='Use flag to only extract the reflectivity (COMPOSITE_REFL_10CM and REFL_10CM), updraft (UP_HELI_MAX) and forecast time (Times) data fields, excluding divergence and vorticity fields. Additionally, do not compute divergence and vorticity. Regardless of the value of this flag, only reflectivity is used for training and prediction.')
    parser.add_argument('-f', '--fields', type=str, nargs='+', #type=list,
        help='Space delimited list of additional WoFS fields to store. Regardless of whether these fields are specified, only reflectivity is used for training and prediction. Ex use: --fields U WSPD10MAX W_UP_MAX')
    parser.add_argument('--interp_method', type=int, default=0,
        help='WoFS to GridRad (and vice versa) interpolation method. 0 to use scipy.interpolate.RectBivariateSpline. 1 to use scipy.spatial.cKDTree')

    # Model directories and files
    parser.add_argument('--loc_model', type=str, required=True,
        help='Trained model directory or file path (i.e. file descriptor)') 
    parser.add_argument('--loc_model_calib', type=str, 
        help='Path to pickle file with a trained model for calibrating the predictions')
    parser.add_argument('--file_trainset_stats', type=str, required=True,
        help='Path to training set statistics file (i.e., training metadata in Lydias code) for normalizing test data. Contains the means and std computed from the training data for at least the reflectivity (i.e., ZH)')
    # If loading model weights and using hyperparameters from_weights
    hyperparmas_sparsers = parser.add_subparsers(title='model_loading', dest='load_options', 
        help='optional, additional model loading options')
    hp_parsers = hyperparmas_sparsers.add_parser('load_weights_hps', #aliases=['hyper'], 
        help='Specifiy details to load model weights and UNetHyperModel hyperparameters')
    #hp_parsers.add_argument('--from_weights', action='store_true',
    #    help='Boolean whether to load the model as weights')
    hp_parsers.add_argument('--hp_path', type=str, required=True,
        help='Path to the csv containing the top hyperparameters')
    hp_parsers.add_argument('--hp_idx', type=int, default=0,
        help='Index indicating the row to use within the csv of the top hyperparameters')
    
    # Preds to msgpk ___________________________________________________
    parser.add_argument('-v', '--variables', nargs='+', type=str,
        help='List of string variables to save out from predictions. Ex use: --variables ML_PREDICTED_TOR REFL_10CM')
    parser.add_argument('-t', '--thresholds', nargs='+', type=float,
        help='Space delimited list of float thresholds. Ex use: --thresholds 0.08 0.07')
    
    parser.add_argument('-w', '--write', type=int, default=0,
        help='Write/save data and/or figures. Set to 0 to save nothing, set to 1 to only save WoFS predictions file (.nc), set to 2 to only save all .nc data files ([patched ]data on gridrad grid), set to 3 to only save figures, and set to 4 to save all data files and all figures')
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
    parser = create_argsparser(args_list=args_list)
    args = parser.parse_args()
    return args

def process_one_file(wofs_filepath, args):
    
    from lydia_scripts import wofs_raw_predictions_oscer
    vm_filepath = wofs_raw_predictions_oscer.wofs_to_preds(wofs_filepath, args)
    
    return vm_filepath

if __name__ == '__main__':
    
    args = parse_args()

    fcst_time_files = sorted(glob.glob(args.loc_wofs+"/ENS_MEM_1/**.nc"))
    fcst_time_files = [f.rsplit("/")[-1] for f in fcst_time_files]
    ncpus = int(os.getenv("SLURM_CPUS_PER_TASK"))
    
    for fcst_time_file in fcst_time_files:
        wofs_fps = sorted(glob.glob(args.loc_wofs+f"/ENS_MEM_**/{fcst_time_file}"))
        with Pool(ncpus) as p:
            try:
                result = p.starmap(process_one_file,
                                   [(wofs_fp, args) for wofs_fp in wofs_fps],
                                   )
                # result = process_one_file(wofs_fps[0], args)
                # result = [f.replace('wofs-preds-2023-hgt', 'wofs-preds-2023-update') for f in wofs_fps]
                print(result)
                preds_to_msgpk_oscer.preds_to_msgpk(result, args)
    
            except Exception as e:
                print(traceback.format_exc())
