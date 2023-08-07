"""
author: Monique Shotande

Main for visual analysis of GridRad data

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
#import netCDF4
#print("netCDF4 version", netCDF4.__version__)
from netCDF4 import Dataset, date2num, num2date
#import scipy
from scipy import spatial
#print("scipy version", scipy.__version__)
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
#from keras_unet_collection import models
from keras_unet_collection.activations import *
from scripts_tensorboard.unet_hypermodel import UNetHyperModel, prep_data, plot_reliabilty_curve
from tensorflow.keras.callbacks import EarlyStopping 


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
    
    parser.add_argument('--method_reg', type=str, default='log', 
                        help='Type of regression to use for calibrating the predictions. either "log" for logistic regression or "iso" for isotonic regression')

    parser.add_argument('--hp_path', type=str, #required=True,
                        help='Path to the csv containing the top hyperparameters')
    parser.add_argument('--hp_idx', type=int, default=0,
                        help='Index indicating the row to use within the csv of the top hyperparameters')
    
    parser.add_argument('-w', '--write', type=int, default=0,
                        help='Write/save data and/or figures. Set to 0 to save nothing, set to 1 to only save predictions file (.nc), set to 2 to only save all .nc data files ([patched ]data), set to 3 to only save figures, and set to 4 to save all data files and all figures')
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


"""
Manage and structure GridRad data
"""
def load_gridrad(args):
    '''
    Load storm mask and radar files
    '''
    radar_filepattern = os.path.join(args.in_dir, '*/*/*') #f'{args.in_dir_radar}/*/*/*')
    all_gridrad_files = glob.glob(radar_filepattern)
    all_gridrad_files.sort()

    storm_filepattern = os.path.join(args.in_dir_labels, '*/*/5_min_window/*')
    all_storm_mask_files = glob.glob(storm_filepattern)
    all_storm_mask_files.sort()

    return all_gridrad_files, all_storm_mask_files

def pair_gridrad_to_storms(args, gridrad_filenames, labels_filenames, DB=False):
    '''
    Match up the gridrad files with their corresponding storm labels files
    :param gridrad_filenames: list of strings of  gridrad file path names
    :param labels_filenames: list of strings of storm label file path names

    :return: list of 2-tuples, where each tuple contains the strings for the 
            gridrad and storm label file path names, respectively
    '''
    gridrad_stormlabel_files = []
    for storm_mask_fnpath in labels_filenames:
        #'storm_mask_2013-01-29-220000.nc'
        # Break up the directory structure
        path_parts = storm_mask_fnpath.split('/')
        # Remove empty strings from list of path parts
        path_parts = list(filter(lambda x: x != '', path_parts))

        storm_mask_basename, ext = os.path.splitext(path_parts[-1])
        # Remove filename prefix to easily extract datetime
        str_dt = storm_mask_basename.replace('storm_mask_', '')
        # File datetime (note: not necessarily same as that of the directory)
        file_datetime = datetime.strptime(str_dt, '%Y-%m-%d-%H%M%S') # datetime object

        YYYY = path_parts[-4]
        YYYYMMDD_dir = path_parts[-3]

        nexrad_fn = file_datetime.strftime('nexrad_3d_v4_2_%Y%m%dT%H%M%SZ.nc')
        gridrad_fn = f'{YYYY}/{YYYYMMDD_dir}/{nexrad_fn}'
        gridrad_fnpath = os.path.join(args.in_dir, gridrad_fn)

        gridrad_stormlabel_files.append((gridrad_fnpath, storm_mask_fnpath))
        if DB:
            print(gridrad_fnpath, storm_mask_fnpath)

    return gridrad_stormlabel_files

def prep_data_radars_storms(args, all_gridrad_stormlabel_files, patch=True):
    '''
    Make tf Datasets from the radar and storm files

    Parameters
    ----------
    args: command arguments see create_argsparser
    all_gridrad_stormlabel_files: list of tuples of gridrad-stormlabel files
            pairs

    Returns
    ----------
    ds: tf Dataset of radar-tor pairs
    '''
    gridrad_stormlabel_files = np.array(all_gridrad_stormlabel_files)
    ds = tf.data.Dataset.from_tensor_slices(gridrad_stormlabel_files)

    ds = ds.map(prep_ntuple, num_parallel_calls=tf.data.AUTOTUNE)
    ds_unpatched = ds.map(prep_pair, num_parallel_calls=tf.data.AUTOTUNE)

    if patch:
        pass

    ds = ds.batch(args.batch)

    return ds

def prep_ntuple(gr_fnpath, sl_fname):
    '''
    Helper method to create tf Datasets from gridrad-stormlabel files pairs

    Parameters
    ----------
    gr_fnpath: string for gridrad file name path
    sl_fname: string for the correspond storm labels file name path
    
    Returns
    ----------
    X: 3D array for radar
    Y: 2D array for tornadic labels
    S: 2D array for significant tornado labels 
    T: 2D array for raw storm labels 
    istor: bool whether the storm has at least one tornadic pixel
    isclear: bool whether the storm 'clear air' (i.e. all pixels have 
            reflectivity below 30
    '''
    #print(f"Opening {gr_fnpath}")
    #print(f"Opening {sl_fname}")
    gridrad_ds = xr.load_dataset(gr_fnpath)
    storm_mask_ds = xr.load_dataset(sl_fname)

    Lat = gridrad_ds.Latitude.values
    Lon = gridrad_ds.Longitude.values
    Alt = gridrad_ds.Altitude.values

    time = gridrad_ds.time.values

    X = np.squeeze(gridrad_ds.ZH.values)
    X = np.transpose(X, (1, 2, 0)) # move altitude to be last dimension
    #X = np.moveaxis(X, 0, -1) # move altitude to be last dimension

    # Raw labels (0: notor, 1: EF0-1, 2: EF2-5)
    T = np.squeeze(storm_mask_ds.storm_mask.values)
    # All tornadoes mask
    Y = np.where(T > 0, 1, 0)
    # Significant tornadoes mask
    S = np.where(T > 1, 1, 0)
    # Is tornadic (at least one tornadic pixel)
    istor = np.any(T > 0)
    # Is clear air (all pixels less than 30 dBZ)
    isclear = np.all(X < 30)

    #n_pixels_EF0_EF1 = ([], np.count_nonzero(storm_mask[xi:xi+size, yi:yi+size] == 1)),
    #n_pixels_EF2_EF5 = ([], np.count_nonzero(storm_mask[xi:xi+size, yi:yi+size] == 2)),
    #n_convective_pixels = ([], np.count_nonzero(radar.ZH.isel(Latitude=slice(xi, xi+size), Longitude=slice(yi, yi+size), time=0).fillna(0).values.swapaxes(0,2).swapaxes(1,0).max(axis=(2)) >= 30

    return X, Y, S, T, istor, isclear, Lat, Lon, Alt

def prep_pair(X, Y, S, T, istor, isclear, Lat, Lon, Alt):
    '''
    Helper method to create tf Datasets from dataset with the loaded 
    gridrad-stormlabel n-tuples

    Parameters
    ----------
    X: 3D array for radar
    Y: 2D array for tornadic labels
    S: 2D array for significant tornado labels 
    T: 2D array for raw storm labels 
    istor: bool whether the storm has at least one tornadic pixel
    isclear: bool whether the storm 'clear air' (i.e. all pixels have 
            reflectivity below 30
    
    Returns
    ----------
    X: 3D array for radar
    Y: 2D array for tornadic labels
    '''
    return X, Y


"""
Construct patches
"""
def make_patches_random(radar, storm_mask, patch_shape, n=100, fwindow=5, 
                 lat_options=None, lon_options=None, Zthres=30):
    '''
    Create random patches. Based on code from Lydia
    :param radar: xarray dataset (GridRad or potentially WoFS) containing reflectivity, 
            divergence, and reflectivity
    :param storm_mask: xarray dataset containing:
            storm_mask: matrix indicating which pixels are tornadic
            time: scalar indicating the time
            Longitude: matrix 
            Latitude: matrix
    :param patch_shape: tuple of length 1 or 2. 
            0: size of x dimension (and y dimension when length is 1)
            1: size of y dimension
    :param n: int number of of patches to create. 
            TODO: if None use all sets of nonoveerlapping patches
    :param fwindow: int for the forecast window in minutes. default 5
    :param lat_options: 1D array or list of the latitude locations containing tornadic pixels
    :param lon_options: 1D array or list of the longitude locations containing tornadic pixels
    :param Zthres: float minimum reflectivity indicating a convective pixel. default 30
    '''
    #T, height, width = storm_mask_ds.storm_mask.shape #(1, 480, 528)
    T, levels, height, width = radar.ZH.shape #(1, 29, 480, 528)
    mask = storm_mask.storm_mask[0].values
    time = storm_mask.time.values
    lats = storm_mask.Latitude.values
    lons = storm_mask.Longitude.values
    
    # Patch size
    xsize = patch_shape[0]
    ysize = xsize
    if len(args.patch_shape) == 2:
        ysize = args.patch_shape[1]

    # Make n patches from this time
    patches = []
    xs = []
    ys = []
    
    # Randomly pre-generate lists of random coordinates
    if lat_options is None and lon_options is None:
        # Select from anywhere within the image
        xs = np.random.randint(0, height - ysize, size=n) #[low, high)
        ys = np.random.randint(0, width - xsize, size=n)
    else:
        # Select random for regions with tornadic pixels
        sel = np.random.randint(0, lat_options.size, size=n)
        xs = lat_options[sel]
        ys = lon_options[sel]
        
    for k in range(n):
        # Select a random patch from the available gridpoints
        xi = xs[k]
        yi = ys[k]
        print(f"my ({xi}, {yi})")
        
        xj = xi + xsize
        yj = yi + ysize
        
        xsel = slice(xi, xj)
        ysel = slice(yi, yj)
        #print(k, xsel, ysel)

        _patch = xr.Dataset(data_vars=dict(
                           ZH=(["x", "y", "z"], radar.ZH.isel(Latitude=xsel, Longitude=ysel, 
                                                              time=0).values.swapaxes(0,2).swapaxes(1,0)), 
                           DIV=(["x", "y", "z"], radar.DIV.isel(Latitude=xsel, Longitude=ysel, 
                                                                time=0).values.swapaxes(0,2).swapaxes(1,0)),
                           VOR=(["x", "y", "z"], radar.VOR.isel(Latitude=xsel, Longitude=ysel, 
                                                                time=0).values.swapaxes(0,2).swapaxes(1,0)),
                           labels=(["x", "y"], mask[xi:xj, yi:yj]),
                           n_tornadic_pixels=([], np.count_nonzero(mask[xi:xj, yi:yj])),
                           n_pixels_EF0_EF1 = ([], np.count_nonzero(mask[xi:xj, yi:yj] == 1)),
                           n_pixels_EF2_EF5 = ([], np.count_nonzero(mask[xi:xj, yi:yj] == 2)),
                           n_convective_pixels = ([], np.count_nonzero(radar.ZH.isel(Latitude=xsel, 
                                                                                     Longitude=ysel, 
                                                                                     time=0).fillna(0).values.swapaxes(0,2).swapaxes(1,0).max(axis=(2)) >= Zthres)),
                           lat=([], lats[xi]),
                           lon=([], lons[yi]),
                           time=([], time),
                           forecast_window=([], fwindow)),
                           top_left_lat_lon=([], (xi, yi)), 
                    coords=dict(x=(["x"], np.arange(xsize, dtype=np.int16)), y=(["y"], np.arange(ysize, dtype=np.int16)), 
                                z=(["z"], np.arange(levels, dtype=np.int16))))
        _patch = _patch.fillna(0)
        
        _patch['n_convective_pixels'] = ([], np.count_nonzero(_patch.ZH.max(axis=(2), skipna=True) >= Zthres))
        patches.append(_patch)
    
    output = xr.concat(patches, 'patch')
    
    return output

def make_patches(args, radar, window):
    '''
    Break the data into all regular patches. Create a Dataset of concatenated 
    along the patches

    Parameters
    ----------
    args: parsed command line args
            Relevant args: see create_argsparser for more details
                ZH_only: only extract ZH
                patch_shape: shape of the patches
    radar: WoFS data
    window: duration time in minutes of the forecast window (i.e., the 
                    time between the initialization of the simulation and the 
                    time of the current forecast)

    Return
    ----------
    ds_patches: xarray Dataset of patches
    '''
    # List of patches
    patches = []

    Z_only = args.ZH_only
    
    lat_len = radar.Latitude.shape[0]
    lon_len = radar.Longitude.shape[0]

    xsize = None
    ysize = None
    csize = None
    ndims = len(args.patch_shape)
    if ndims > 0:
        xsize = args.patch_shape[0]
    if ndims == 1:
        ysize = xsize
    elif ndims >= 2:
        ysize = args.patch_shape[1]
    if ndims == 3:
        # Number of channels
        csize = args.patch_shape[2]
    #else: raise ValueError(f"[ARGUMENTS] patch_shape number of dimensions should be 0, 1, 2, or 3 but was {ndims}")
    
    lat_range = range(0, lat_len, xsize - 4)
    lon_range = range(0, lon_len, ysize - 4)
    
    naltitudes = radar.Altitude.values.shape[0]

    # Iterate over Latitude and Longitude for every size-th pixel. Performed every (xi, yi) = multiples of size, and will give us a normal array
    pi = 0 # patch index
    for xi in lat_range:
        for yi in lon_range:  
        
            # Account for case that the patch goes outside of the domain
            # If part of patch is outside, move over, so the patch edge lines up with the domain edge
            if xi >= lat_len - xsize:
                xi = lat_len - xsize - 1
            if yi >= lon_len - ysize:
                yi = lon_len - ysize - 1

            xj = xi + xsize
            yj = yi + ysize
            xsel = slice(xi, xj)
            ysel = slice(yi, yj)

            # Create the patch
            data_vars = dict(
                            ZH=(["x", "y", "z"], radar.ZH.isel(Latitude=xsel, Longitude=ysel, 
                                                              time=0).values.swapaxes(0,2).swapaxes(1,0)), 
                            UH=(["x", "y"], radar.UH.isel(Latitude=xsel, Longitude=ysel, 
                                                         time=0, Altitude=0).values),
                            stitched_x=(["x"], range(xi, xj)),
                            stitched_y=(["x"], range(yi, yj)),
                            n_convective_pixels = ([], np.count_nonzero(radar.ZH.isel(Latitude=xsel,
                                                                                      Longitude=ysel,
                                                                                      time=0).fillna(0).values.swapaxes(0,2).swapaxes(1,0).max(axis=(2)) >= 30)),
                            n_uh_pixels = ([], np.count_nonzero(radar.UH.isel(Latitude=xsel, 
                                                                              Longitude=ysel,
                                                                              time=0).fillna(0).values > 0)),
                            lat=([], radar.Latitude.values[xi]),
                            lon=([], radar.Longitude.values[yi]),
                            time=([], radar.time.values[0]),
                            forecast_window=([], window))
            if not Z_only:
                div = dict(DIV=(["x", "y", "z"], radar.DIV.isel(Latitude=xsel, Longitude=ysel, 
                                                                time=0).values.swapaxes(0,2).swapaxes(1,0)))
                vor = dict(VOR=(["x", "y", "z"], radar.VOR.isel(Latitude=xsel, Longitude=ysel, 
                                                                time=0).values.swapaxes(0,2).swapaxes(1,0)))
                data_vars.update(div)
                data_vars.update(vor)

            to_add = xr.Dataset(data_vars=data_vars,
                                coords=dict(patch=(['patch'], [np.int32(pi)]),
                                            x=(["x"], np.arange(xsize, dtype=np.int32)), 
                                            y=(["y"], np.arange(ysize, dtype=np.int32)), 
                                            z=(["z"], np.arange(1, naltitudes + 1, dtype=np.int32))))

            to_add = to_add.fillna(0)

            patches.append(to_add)
            pi += 1

    # Combine all patches into one dataset
    ds_patches = xr.concat(patches, 'patch')
    
    return ds_patches

def make_patches_ds(X, Y, patch_shape=(32,), noverlap=(4,)):
    '''TODO
    Break the data into all regular patches. Create a Dataset of concatenated 
    along the patches

    Parameters
    ----------
    X: 
    Y:
    patch_shape: int or tuple. number of pixels overlaping between patches. if 
            tuple, entry 0 is the number of overlapping pixels in the x dimension
            and entry 1 is the number of overlapping pixels in the y dimension.
            if an int use the same number for both directions
    noverlap: tuple. number of pixels overlaping between patches. if 
            tuple length is 2, entry 0 is the number of overlapping pixels in the x dimension
            and entry 1 is the number of overlapping pixels in the y dimension.
            if tuple length 1, use the same number for both dimensions

    Return
    ----------
    ds_patches: xarray Dataset of patches
    '''
    # List of patches
    patches = []

    xsize = patch_shape[0]
    ysize = None
    csize = None
    ndims = len(patch_shape)
    if ndims == 1:
        ysize = xsize
    elif ndims >= 2:
        ysize = patch_shape[1]
    else: 
        raise ValueError(f"in make_patches_ds, argument patch_shape number of dimensions should be 1 or 2 but was {ndims}")
    
    lat_len = X.shape[0]
    lon_len = X.shape[1]

    xoverlap = noverlap[0]
    yoverlap = xoverlap
    if len(noverlap) == 2:
        yoverlap = noverlap[1]
    lat_range = range(0, lat_len, xsize - xoverlap)
    lon_range = range(0, lon_len, ysize - yoverlap)
    
    naltitudes = X.shape[-1]

    # Iterate over Latitude and Longitude for every size-th pixel. Performed every (xi, yi) = multiples of size, and will give us a normal array
    pi = 0 # patch index
    for xi in lat_range:
        for yi in lon_range:  
        
            # Account for case that the patch goes outside of the domain
            # If part of patch is outside, move over, so the patch edge lines up with the domain edge
            if xi >= lat_len - xsize:
                xi = lat_len - xsize - 1
            if yi >= lon_len - ysize:
                yi = lon_len - ysize - 1

            xj = xi + xsize + 1
            yj = yi + ysize + 1
            xsel = slice(xi, xj)
            ysel = slice(yi, yj)

            _X = X[xsel, ysel, :]

            # Create the patch
            data_vars = dict(
                            ZH=(["x", "y", "z"], X),
                                #radar.ZH.isel(Latitude=xsel, Longitude=ysel, 
                                #time=0).values.swapaxes(0,2).swapaxes(1,0)), 
                            stitched_x=(["x"], range(xi, xj)),
                            stitched_y=(["x"], range(yi, yj)),
                            n_convective_pixels = ([], np.count_nonzero(radar.ZH.isel(Latitude=xsel,
                                                                                      Longitude=ysel,
                                                                                      time=0).fillna(0).values.swapaxes(0,2).swapaxes(1,0).max(axis=(2)) >= 30)),
                            n_uh_pixels = ([], np.count_nonzero(radar.UH.isel(Latitude=xsel, 
                                                                              Longitude=ysel,
                                                                              time=0).fillna(0).values > 0)),
                            lat=([], radar.Latitude.values[xi]),
                            lon=([], radar.Longitude.values[yi]),
                            time=([], radar.time.values[0]))

            to_add = xr.Dataset(data_vars=data_vars,
                                coords=dict(patch=(['patch'], [np.int32(pi)]),
                                            x=(["x"], np.arange(xsize, dtype=np.int32)), 
                                            y=(["y"], np.arange(ysize, dtype=np.int32)), 
                                            z=(["z"], np.arange(1, naltitudes + 1, dtype=np.int32))))

            to_add = to_add.fillna(0)

            patches.append(to_add)
            pi += 1

    # Combine all patches into one dataset
    ds_patches = xr.concat(patches, 'patch')
    
    return ds_patches

def fit_transform_predictions(args, ytrue, ypred, method='log', **kwargs):
    ''' TODO TEST
    Use isotonic or logistic regression to transform the predictions into a more
    forecaster friendly range

    @param args:
    @param ytrue: 
    @param ypred: 
    @param method: string indicating the regression approach to use. could also
            be Sci-Kit Learn Regresssion object.
            string options:
                'log': use sklearn.linear_model.LogisticRegression
                'iso': use sklearn.isotonic.IsotonicRegression
    @param **kwargs: arguments for the regression method is either 'log' or 'iso'

    Returns
    ---------
    reg: fit Sci-kit Learn regression model
    yhat:
    mu_acc: mean accuracy
    '''
    reg = None
    if method == 'log':
        reg = LogisticRegression(**kwargs).fit(ypred, ytrue)
    elif method == 'iso':
        reg = IsotonicRegression(**kwargs).fit(ypred, ytrue)
        #yhat = isotonic_regression(ypred, y_min=0, y_max=1, increasing=True)
    elif isinstance(method, IsotonicRegression) or isinstance(method, LogisticRegression):
        reg = method.fit(ypred, ytrue)
    else:
        raise ValueError(f'in transform_predictions. method was {method}. method should either be "log", "iso", or instance of a sklearn Regression Object.')
    
    mu_acc = reg.score(ypred, ytrue)
    yhat = reg.predict(ypred)
    return reg, yhat, mu_acc


"""
Construct Hypermodel
"""
def get_hp_from_df(df_hps, row=0):
    '''
    :param df_hps: pandas DataFrame with the hyperparameters
    '''
    best_hps = df_hps.drop(columns=['Unnamed: 0', 'args']) #['Unnamed: 0', 'args']
    #hps_dict = best_hps.to_dict(orient='dict')
    hps_dict = best_hps.iloc[row].to_dict()

    hp = HyperParameters()
    #hp.values = {}
    for k, v in hps_dict.items(): 
        hp.Fixed(k, value=v)

    return hp

def build_fit_hypermodel(args, train_data, val_data, train_val_steps,
                         callbacks=[]):
    '''
    Build and train the hypermodel
    :param args
    :param train_data
    :param val_data
    :param train_val_steps
    :param callbacks

    :return:
        hps
        model
        history
    '''
    # Load hyperparameters and create HP object
    df_hps = pd.read_csv(args.hp_path)
    hps = get_hp_from_df(df_hps, row=args.hp_idx)

    # Re-build model from hyperparameters
    hmodel = UNetHyperModel(input_shape=args.x_shape, n_labels=args.n_labels) #input_shape=(32, 32, 12), n_labels=1
    model = hmodel.build(hps) 

    if callbacks == []:
        es = EarlyStopping(monitor='val_loss', patience=12,  
                        min_delta=1e-5, restore_best_weights=True)
        #es = EarlyStopping(monitor=args.objective, patience=args.patience,  
        #                    min_delta=args.min_delta, restore_best_weights=True)
        callbacks.append(es)

    # Re-train model 
    H = model.fit(train_data, validation_data=val_data, 
                  steps_per_epoch=train_val_steps['steps_per_epoch'],
                  validation_steps=train_val_steps['val_steps'], 
                  epochs=args.epochs, callbacks=callbacks) 

    return hps, model, H


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
        #config.gpu_options.allow_growth = True
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
    #os.chmod(args.lscratch, mode=0o777)
    ds_train, ds_val, ds_test, train_val_steps, ds_train_og, ds_val_og = prep_data(
                args, n_labels=args.n_labels, sample_method='sample')

    # Construct model
    hps, model, H = build_fit_hypermodel(args, ds_train, ds_val, 
                                         train_val_steps) #, callbacks=[])
    # Or load  the model
    model = keras.models.load(args.loc_model)
    
    #ds_patches = make_patches(args, wofs_regridded, forecast_window)
    #preds = predict(args, wofs_gridrad, train_stats, from_weights=from_weights, DB=DB)
    #predictions =  stitch_patches(args, wofs_gridrad, stats, 
                                    #gridrad_spacing=gridrad_spacing, 
                                    #seconds_since=seconds_since, DB=DB)
    

    '''
    # Evaluate trained model
    print("\nEVALUATION")
    train_eval = model.evaluate(ds_train_og, workers=3, use_multiprocessing=True)
    val_eval = model.evaluate(ds_val_og, workers=3, use_multiprocessing=True)
    if ds_test is not None:
        test_eval = model.evaluate(ds_test, workers=3, use_multiprocessing=True)
        '''

    # Predict with trained model
    print("\nPREDICTION")
    xtrain_preds = model.predict(ds_train_og, #steps=train_val_steps['steps_per_epoch'], 
                                 verbose=1, workers=3, use_multiprocessing=True)
    xval_preds = model.predict(ds_val_og, #steps=train_val_steps['val_steps'],
                               verbose=1, workers=3, use_multiprocessing=True)
    if ds_test is not None:
        xtest_preds = model.predict(ds_test, verbose=1, workers=3, 
                                    use_multiprocessing=True)

    # Calibrate Predictions
    if args.class_weight is None:
        y_train = np.concatenate([y for x, y in ds_train_og]) #ds.map(get_y)
    else:
        y_train = np.concatenate([y for x, y, w in ds_train_og])
    y_val = np.concatenate([y for x, y in ds_val_og])
    if ds_test is not None:
        y_test = np.concatenate([y for x, y in ds_test])

    reg_args = {}
    calib, ycalib, mu_acc = fit_transform_predictions(args, y_train, xtrain_preds, 
                                   method=args.method_reg, **reg_args)
    # Save fit regression model
    fname = os.path.join(args.dir_preds, 'calibraion_model_log.pkl')
    #>>pickle.dump(calib, open(fname, "wb"))
    print(f"Saving calibration model {fname}")

    ycalib_val = calib.predict(xval_preds)

    fname = os.path.join(args.dir_preds, f"calibraion_model_log_reliability_train_val.png")
    strat = 'uniform'
    fig, ax = plot_reliabilty_curve(y_train.ravel(), ycalib.ravel(), fname, 
                                    label='Train', strategy=strat, save=False)
    plot_reliabilty_curve(y_val.ravel(), ycalib_val.ravel(), fname, 
                          label='Val', c='orange', 
                          strategy=strat, fig_ax=(fig, ax), save=False) #>>SAVE
    print(f"Saving calibration figure {fname}")
    


    '''
    # Load unpatched data
    all_gridrad_files, all_storm_mask_files = load_gridrad(args)
    # Match up radar data and storm labels
    all_gridrad_stormlabel_files = pair_gridrad_to_storms(args, all_gridrad_files,
                                                          all_storm_mask_files)
    gridrad_stormlabel_files = np.array(all_gridrad_stormlabel_files)
    ds = tf.data.Dataset.from_tensor_slices(gridrad_stormlabel_files)
    '''


