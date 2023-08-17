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
from keras_unet_collection.activations import * #import GELU
from scripts_tensorboard.unet_hypermodel import UNetHyperModel, prep_data, plot_reliabilty_curve, plot_predictions, plot_preds_hists
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
def get_gridrad_storm_filenames(args):
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
    Match up the gridrad files with their corresponding storm labels files based
    on the datetimes
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

def create_dataset(args, all_gridrad_stormlabel_files, pair=False, patch=False):
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
    X = gridrad_stormlabel_files[:, 0]
    Y = gridrad_stormlabel_files[:, 1]
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    print("make slice", ds)

    # Loads the files
    ds = ds.map(prep_ntuple, num_parallel_calls=tf.data.AUTOTUNE)

    # Extract just the reflectivity and storm label pair. Removes all other fields
    if pair:
        ds = ds.map(prep_pair, num_parallel_calls=tf.data.AUTOTUNE)

    #if patch:
    #make_patches_ds(X, Y, S, T, istor, isclear, Lat, Lon, Alt, time, 
    #                patch_shape=args.patch_shape, noverlap=(4,), clear_thres=30)
    
    #ds = ds.map(lambda X, Y, S, T, istor, isclear, Lat, Lon, Alt, time: make_patches_ds(X, Y, S, T, istor, isclear, Lat, Lon, Alt, time, patch_shape=args.patch_shape, noverlap=(0,)))
    ds = ds.map(lambda XY: make_patches_ds(*XY, patch_shape=args.patch_shape, noverlap=(0,)), num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(args.batch)
    print(ds)

    return ds

def prep_ntuple(gr_fnpath, sl_fname, clear_thres=30):
    '''
    Helper method to create tf Datasets from gridrad-stormlabel files pairs.
    Loads the files.

    Parameters
    ----------
    gr_fnpath: string for gridrad file name path
    sl_fname: string for the correspond storm labels file name path
    clear_thres: int threshold for clear air. Reflectivity below clear_thres
            is considered clear air. Default 30
    
    Returns
    ----------
    X: 3D array for radar
    Y: 2D array for tornadic labels
    S: 2D array for significant tornado labels 
    T: 2D array for raw storm labels 
    istor: bool whether the storm has at least one tornadic pixel
    isclear: bool whether the storm 'clear air' (i.e. all pixels have 
            reflectivity below 30
    Lon: array for longtitude
    Lat: array for latittude
    Alt: array for altitude
    '''
    if not os.path.exists(gr_fnpath):
        print(f"Does not exist {gr_fnpath}")
    if not os.path.exists(sl_fname):
        print(f"Does not exist {sl_fname}")

    gridrad_ds = xr.load_dataset(gr_fnpath) #, engine='netcdf4')
    storm_mask_ds = xr.load_dataset(sl_fname) #, engine='netcdf4')

    Lat = gridrad_ds.Latitude.values
    Lon = gridrad_ds.Longitude.values
    Alt = gridrad_ds.Altitude.values

    time = gridrad_ds.time.values[0]

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
    isclear = np.all(X < clear_thres)

    #n_pixels_EF0_EF1 = ([], np.count_nonzero(storm_mask[xi:xi+size, yi:yi+size] == 1)),
    #n_pixels_EF2_EF5 = ([], np.count_nonzero(storm_mask[xi:xi+size, yi:yi+size] == 2)),
    #n_convective_pixels = ([], np.count_nonzero(radar.ZH.isel(Latitude=slice(xi, xi+size), Longitude=slice(yi, yi+size), time=0).fillna(0).values.swapaxes(0,2).swapaxes(1,0).max(axis=(2)) >= 30

    return X, Y, S, T, istor, isclear, Lat, Lon, Alt, time

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
                            n_convective_pixels = ([], np.count_nonzero(radar.ZH.isel(Latitude=xsel, Longitude=ysel, time=0).fillna(0).values.swapaxes(0,2).swapaxes(1,0).max(axis=(2)) >= 30)),
                            #n_uh_pixels = ([], np.count_nonzero(radar.UH.isel(Latitude=xsel,  Longitude=ysel, time=0).fillna(0).values > 0)),
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

def make_patches_ds(X, Y, S, T, istor, isclear, Lat, Lon, Alt, time, 
                    patch_shape=(32,), noverlap=(4,), clear_thres=30):
    '''TODO
    Break the data into all regular patches. 

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
    ds_patches: 
    '''
    # List of patches
    patches = []

    xsize = patch_shape[0]  #lat
    ysize = None            #lon
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
    
    X_filld = np.nan_to_num(X, nan=0) #.fillna(0)
    
    #tf.keras.layers.RandomCrop

    # Iterate over Latitude and Longitude for every size-th pixel. Performed every (xi, yi) = multiples of size, and will give us a normal array
    for xi in lat_range:
        for yi in lon_range:  
        
            # Account for case that the patch goes outside of the domain
            # If part of patch is outside, move over, so the patch edge lines up with the domain edge
            if xi >= lat_len - xsize:
                xi = lat_len - xsize - 1
            if yi >= lon_len - ysize:
                yi = lon_len - ysize - 1

            xj = xi + xsize #+ 1
            yj = yi + ysize #+ 1
            xsel = slice(xi, xj)
            ysel = slice(yi, yj)

            #_X = X[xsel, ysel, :]
            #_X_filld = _X.nan_to_num(_X, nan=0) #.fillna(0)

            #_Y = Y[xsel, ysel, :]

            Xpatch = tf.image.crop_to_bounding_box(X_filld, xi, yi, xsize, ysize)
            Ypatch = tf.image.crop_to_bounding_box(Y, xi, yi, xsize, ysize)
            Spatch = tf.image.crop_to_bounding_box(S, xi, yi, xsize, ysize)
            Tpatch = tf.image.crop_to_bounding_box(T, xi, yi, xsize, ysize)
            print("X,Y shape", Xpatch.shape, Ypatch.shape)
            #patch_filld = patch.nan_to_num(patch, nan=0) #.fillna(0)
            nconvective_pixels = np.count_nonzero(Xpatch.max(axis=2) >= clear_thres)
            # lat=([], Lat[xi]), #radar.Latitude.values[xi]
            # lon=([], Lon[yi]), #radar.Longitude.values[yi]
            # time=([], time)) #radar.time.values[0]

            #X, Y, S, T, istor, isclear, Lat, Lon, Alt, time
            patches.append((Xpatch, Ypatch, Spatch, Tpatch, istor, isclear, 
                            Lat[xi], Lon[yi], Alt, time, xi, yi))

    # Combine all patches into one dataset
    ds_patches = xr.concat(patches, 'patch')
    print("ds_patch dims", ds_patches.dims)
    print("ds_patch coords", ds_patches.coords)
    
    return ds_patches

def make_patches_ds_xr(X, Y, Lat, Lon, Alt, time, patch_shape=(32,), noverlap=(4,)):
    ''' TODO
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
    
    #tf.image.crop_to_bounding_box

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
            _X_filld = _X.fillna(0)

            # Create the patch
            data_vars = dict(
                            ZH=(["x", "y", "z"], _X),
                                #radar.ZH.isel(Latitude=xsel, Longitude=ysel, 
                                #time=0).values.swapaxes(0,2).swapaxes(1,0)), 
                            stitched_x=(["x"], range(xi, xj)),
                            stitched_y=(["x"], range(yi, yj)),
                            n_convective_pixels = ([], np.count_nonzero(_X_filld.max(axis=2) >= 30)),
                                #radar.ZH.isel(Latitude=xsel, Longitude=ysel, time=0).fillna(0).values.swapaxes(0,2).swapaxes(1,0).max(axis=(2)) >= 30)),
                            lat=([], Lat[xi]), #radar.Latitude.values[xi]
                            lon=([], Lon[yi]), #radar.Longitude.values[yi]
                            time=([], time)) #radar.time.values[0]

            to_add = xr.Dataset(data_vars=data_vars,
                                coords=dict(patch=(['patch'], [np.int32(pi)]),
                                            x=(["x"], np.arange(xsize, dtype=np.int32)), 
                                            y=(["y"], np.arange(ysize, dtype=np.int32)), 
                                            z=(["z"], Alt))) #np.arange(1, naltitudes + 1, dtype=np.int32)

            to_add = to_add.fillna(0)

            patches.append(to_add)
            pi += 1

    # Combine all patches into one dataset
    ds_patches = xr.concat(patches, 'patch')
    
    return ds_patches

def stitch_patches(X, Y, S, T, istor, isclear, Lat, Lon, Alt, time, noverlap=(4,)):
    """ 
    Reconstruct the grid by stitching the patches back together

    @param args: command line args. see create_argsparser()
            Relevant arguments
                patch_shape: shape of the patches
    @param wofs: WoFS data from a single time point in the gridrad grid
    @param stats: xarray dataset with training mean and std of the reflectivity 
            and other fields
    @param gridrad_spacing: Gridrad files have grid spacings of 1/48th degrees lat/lon
            1 / (gridrad_spacing) degrees
    @param seconds_since: string seconds since date statement to use for creating
            NETCDF4 datetime integers. string of the form since describing the 
            time units. can be days, hours, minutes, etc. see netcdf4.date2num() 
            documentation for more information
    @param DB: int debug flag to print out additional debug information

    @return: xarray Dataset of the stitched WoFS patches
    """

    # Compute total number of grid points in latitude and longitude
    total_in_lat = X.shape[0] #int(wofs.stitched_x.max().values + 1)
    total_in_lon = X.shape[1] #int(wofs.stitched_y.max().values + 1)

    # Get minimum value of latitude and longitude
    lat_min = wofs.lat.min().values
    lon_min = wofs.lon.min().values

    # The grid is regular in lat/lon, therefore reconstruct the lat/lon grid for the stitched data
    lats = np.linspace(lat_min, lat_min+(total_in_lat-1) / gridrad_spacing, total_in_lat)
    lons = np.linspace(lon_min, lon_min+(total_in_lon-1) / gridrad_spacing, total_in_lon)

    # Define empty arrays that will hold the stitched data
    tor_preds = np.zeros((total_in_lat, total_in_lon))
    uh_array = np.zeros((total_in_lat, total_in_lon))
    zh_low_level = np.zeros((total_in_lat, total_in_lon))
    overlap_array = np.zeros((total_in_lat, total_in_lon))

    xsize = X.shape[0]
    ysize = X.shape[1]

    # Loop through all the patches
    zeros = np.ones((xsize, ysize))
    npatches = wofs.patch.values.size
    #for p in range(wofs.patch.values.shape[0]):
    for p in range(npatches):
        # Find locations of the corner of this patch in the stitched grid
        min_x = int(wofs.isel(patch=p).stitched_x.min().values)
        min_y = int(wofs.isel(patch=p).stitched_y.min().values)
        max_x = int(wofs.isel(patch=p).stitched_x.max().values + 1)
        max_y = int(wofs.isel(patch=p).stitched_y.max().values + 1)
        
        # Reconstruct stitched grid by taking the average from all the patches at each grid point
        # At this step, adding the data from all patches
        tor_preds[min_x:max_x, min_y:max_y] += wofs.isel(patch=p).predicted_tor.values      
        uh_array[min_x:max_x, min_y:max_y] += wofs.isel(patch=p).UH.values
        zh_low_level[min_x:max_x, min_y:max_y] += wofs.isel(patch=p, z=0).ZH.values * ZH_std + ZH_mu
        overlap_array[min_x:max_x, min_y:max_y] += zeros
    
    # Compute average of each patch by dividing by the total number of patches that contained each pixel
    tor_preds = tor_preds / overlap_array
    uh_array = uh_array / overlap_array
    zh = zh_low_level
    zh_low_level = zh_low_level / overlap_array

    # Obtain the time of the forecast
    datetime_int = num2date(wofs.time.values[0], seconds_since) #
    forecast_time = np.datetime64(datetime_int)

    # Put stitched grid into Dataset 
    wofs_stiched = xr.Dataset(data_vars=dict(UH=(["time", "lat", "lon"],
                                                    uh_array.reshape(1, total_in_lat, total_in_lon)),
                                            ZH_1km=(["time", "lat", "lon"], 
                                                    zh_low_level.reshape(1, total_in_lat, total_in_lon)),
                                            ZH=(["time", "lat", "lon"], 
                                                    zh.reshape(1, total_in_lat, total_in_lon)),
                                            predicted_tor=(["time", "lat", "lon"], 
                                                    tor_preds.reshape(1, total_in_lat, total_in_lon))),
                            coords=dict(time=[forecast_time], lon=lons, lat=lats),
                            attrs=wofs.attrs
                            )
    wofs_stiched.ZH.attrs['units'] = 'dBZ'
    wofs_stiched.ZH_1km.attrs['units'] = 'dBZ'
    wofs_stiched.UH.attrs['units'] = 'm^2/s^2'

    return wofs_stiched


def fit_transform_predictions(args, ytrue, ypred, method='log', **kwargs):
    ''' 
    Use isotonic or logistic regression to transform the predictions into a more
    forecaster friendly range

    @param args: command line arguments. see create_argsparser()
    @param ytrue: array of ground truth outputs
    @param ypred: array of model predictions
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
    mu_acc: mean accuracy for logistic regression; coefficient of determination
            for isotoni regression
    '''
    reg = None
    if method == 'log':
        reg = LogisticRegression(**kwargs).fit(ypred, ytrue)
    elif method == 'iso':
        reg = IsotonicRegression(y_min=0, y_max=1, **kwargs).fit(ypred.ravel(), ytrue.ravel())
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

    """>>>
    # Load and prepare tf Datasets
    ds_train, ds_val, ds_test, train_val_steps, ds_train_og, ds_val_og = prep_data(
                args, n_labels=args.n_labels, sample_method='sample')

    '''
    # Construct model
    hps, model, H = build_fit_hypermodel(args, ds_train, ds_val, 
                                         train_val_steps) #, callbacks=[])
    '''
    """

    # Or load  the model
    fss_args = {'mask_size': 2, 'num_dimensions': 2, 'c':1.0, 
                'cutoff': 0.5, 'want_hard_discretization': False}
    fss = make_fractions_skill_score(**fss_args)
    model = keras.models.load_model(args.loc_model, 
                                    custom_objects={'fractions_skill_score': fss, 
                                        'MaxCriticalSuccessIndex': MaxCriticalSuccessIndex,
                                        'GELU': GELU})

    """
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

    ntrain = y_train.shape[0]
    nval = y_val.shape[0]
    ntest = y_test.shape[0]

    y_train_flat = y_train.reshape(-1, 1) #.reshape(ntrain, -1)
    xtrain_preds_flat = xtrain_preds.reshape(-1, 1) #.reshape(ntrain, -1)

    y_val_flat = y_val.reshape(-1, 1) #.reshape(nval, -1)
    xval_preds_flat = xval_preds.reshape(-1, 1) #.reshape(nval, -1)

    y_test_flat = y_test.reshape(-1, 1) #.reshape(ntest, -1)
    xtest_preds_flat = xtest_preds.reshape(-1, 1) #.reshape(ntest, -1)

    print("shape", y_train_flat.shape, xtrain_preds_flat.shape)

    reg_args = {}
    calib, ycalib, mu_acc = fit_transform_predictions(args, y_train_flat, 
                                                      xtrain_preds_flat, 
                                                      method=args.method_reg, 
                                                      **reg_args)
    # Save fit regression model
    fname = os.path.join(args.dir_preds, f'calibraion_model_{args.method_reg}.pkl')
    #pickle.dump(calib, open(fname, "wb"))
    with open(fname, 'wb') as fid:
        pickle.dump(calib, fid)
    #with open(fname, 'rb') as fid:
        #calib = pickle.load(fid)
    print(f"Saving calibration model {fname}")

    ycalib_val = calib.predict(xval_preds_flat)
    ycalib_test = calib.predict(xtest_preds_flat)
    <<<<<"""

    # Reliabilty Curve
    plot_calib = False
    if plot_calib:
        fname = os.path.join(args.dir_preds, f"calibration_model_reliability.png")
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
        fname = os.path.join(args.dir_preds, f"calibration_model_reliability_{args.method_reg}.png")
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
    plot_pred_hists = False
    if plot_pred_hists:
        Y = {'Train True': y_train.ravel()[:-1:20], 'Train Preds': xtrain_preds.ravel()[:-1:20],
            'Train Calib': ycalib[:-1:20],
            'Val True': y_val.ravel()[:-1:20], 'Val Preds': xval_preds.ravel()[:-1:20],
            'Val Calib': ycalib_val[:-1:20],
            'Test True': y_test.ravel()[:-1:20], 'Test Preds': xtest_preds.ravel()[:-1:20],
            'Test Calib': ycalib_test[:-1:20]
            }
        fname = os.path.join(args.dir_preds, f"calibration_model_preds_hists_{args.method_reg}.png")
        plot_preds_hists(Y, fname, use_seaborn=True, fig_ax=None, 
                        figsize=(10, 8), alpha=.5, save=True, dpi=160)
        #plot_predictions(y_train.ravel(), ycalib, fname, use_seaborn=True, 
        #                 figsize=(10, 8), alpha=.5, save=False, dpi=160) #y_train.ravel(), xtrain_preds.ravel()
        print(f"Saving preds hists {fname}")


    # Load unpatched data
    all_gridrad_files, all_storm_mask_files = get_gridrad_storm_filenames(args)

    # Match up radar data and storm label files by datetime
    all_gridrad_stormlabel_files = pair_gridrad_to_storms(args, all_gridrad_files,
                                                          all_storm_mask_files)
    
    # Create tf Datasest
    ds = create_dataset(args, all_gridrad_stormlabel_files[:5], 
                        pair=False, patch=False)
    
    '''
    ypreds_train = model.predict(ds, #steps=train_val_steps['steps_per_epoch'], 
                                 verbose=1, workers=4, use_multiprocessing=True)
    ypreds_val = model.predict(ds_val_og, #steps=train_val_steps['val_steps'],
                               verbose=1, workers=4, use_multiprocessing=True)
    if ds_test is not None:
        ypreds_val = model.predict(ds_test, verbose=1, workers=4, 
                                    use_multiprocessing=True)
    '''
        
    # Without creating dataset
    #ds_patches = make_patches(args, wofs_regridded, forecast_window)
    #preds = predict(args, wofs_gridrad, train_stats, from_weights=from_weights, DB=DB)
    #preds_stitch = stitch_patches(args, wofs_gridrad, stats, DB=DB)

    print("DONE.")
