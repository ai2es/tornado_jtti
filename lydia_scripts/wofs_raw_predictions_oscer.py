"""
author: Monique Shotande 
Functions based on those provided by Lydia

End to end script takes as input the raw WoFS (warn on Forecast System) data,
and outputs the predictions in the WoFS grid.

General Procedure:
1. read raw WoFS file(s)
2. interpolate WoFS grid to GridRad grid
3. construct patches (dependent on patch size)
4. load ML model
5. make predictions
6. interpolate back to WoFS grid

Execution Instructions:
    Conda environment requires are in the environment.yml
    file. Execute:
        conda env create --name tf_tornado --file environment.yml
        conda activate tf_tornado
    to create the conda environment necessary to run this script.
    Custom module requirements:
        The custom modules: custom_losses and custom_metrics
        are in the tornado_jtti project directory. Working directory is expected 
        to be tornado_jtti/ to use these custom modules
    Information about the required command line arguments are described in the 
    method get_arguments(). For details on the command line arguments run
        python lydia_scripts/wofs_raw_predictions.py --h
    Example use can be found in lydia_scripts/wofs_raw_predictions.sh:
        python lydia_scripts/wofs_raw_predictions.py --loc_wofs 
                                                     --datetime_format 
                                                     (--filename_prefix)
                                                     --dir_preds
                                                     (--dir_patches)
                                                     (--dir_figs)
                                                     --with_nans 
                                                     (--fields)
                                                     --loc_model
                                                     --file_trainset_stats
                                                     --write=0
                                                     --dry_run
"""
import re, os, sys, errno, glob, argparse
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
from scipy.ndimage import uniform_filter, generic_filter, binary_dilation, grey_dilation
from scipy.interpolate import RectBivariateSpline
#print("scipy version", scipy.__version__)
import wrf #wrf-python=1.3.2.5==py38h0e9072a_0
print("wrf-python version", wrf.__version__)
import metpy
import metpy.calc
#import multiprocessing as mp
#import tqdm

import matplotlib.pyplot as plt
plt.rcParams.update({"text.usetex": True})

import pickle
#from sklearn.isotonic import IsotonicRegression
#from sklearn.linear_model import LogisticRegression

import tensorflow as tf
print("tensorflow version", tf.__version__)
from tensorflow import keras
# print("keras version", keras.__version__)
from keras_tuner import HyperParameters

# Working directory expected to be tornado_jtti/
sys.path.append("lydia_scripts")
from custom_losses import make_fractions_skill_score
from custom_metrics import MaxCriticalSuccessIndex
from scripts_data_pipeline.wofs_to_gridrad_idw import calculate_output_lats_lons
sys.path.append("../keras-unet-collection")
#from keras_unet_collection import models
from keras_unet_collection.activations import * #imports GELU
from scripts_tensorboard.unet_hypermodel import UNetHyperModel

print(" ")
import gc
gc.collect()


"""
WoFS to GridRad interpolation methods, refactored from lydia_scripts/scripts_data_pipeline
"""
def load_wofs_file(filepath, filename_prefix=None, wofs_datetime=None, 
                   datetime_format='%Y-%m-%d_%H:%M:%S', 
                   seconds_since='seconds since 2001-01-01', 
                   engine='netcdf4', DB=0, **kwargs):
    ''' 
    Load the raw WoFS file as xarray dataset and netcdf dataset

    @param filepath: string with the WoFS file path
    @param filename_prefix: (optional) prefix in the WoFS file naming convention
    @param wofs_datetime: (optional) forecast time of the WoFS file. Optionally 
            used if forecast time not expected to be within the WoFS Dataset 
            under the data variable Times
    @param datetime_format: datetime format expected for the datetimes
    @param seconds_since: string seconds since date statement to use for creating
            NETCDF4 datetime integers. string of the form since describing the 
            time units. can be days, hours, minutes, etc. see netcdf4.date2num() 
            documentation for more information
    @param engine: Engine to use when reading file. see documentation for xarray 
            Dataset
    @param DB: int debug flag to print out additional debug information
    @param kwargs: any additional desired parameters to xarray.open_dataset(...)
    
    @return: tuple with xarray Dataset and netCDF4 Dataset
    '''
    if DB: print(f"Loading {filepath}")

    # Create Dataset
    wofs = None
    wofs = xr.open_dataset(filepath, engine=engine, decode_coords=True, **kwargs).load()
    wofs.close()
    wofs.attrs['filenamepath'] = filepath

    if wofs_datetime is None:
        wofs_datetime = wofs.Times.data[0].decode('ascii') # decode from binary string to text string 
    if wofs_datetime == '':
        wofs_datetime = wofs.Times.data[1].decode('ascii') # decode from binary string to text string
    _datetime, dt_int, dt_str = get_wofs_datetime(filepath, filename_prefix=filename_prefix, 
                                                wofs_datetime=wofs_datetime, seconds_since=seconds_since, 
                                                datetime_format=datetime_format, DB=DB)
    wofs = wofs.reindex(Time=[dt_int])
    wofs.attrs['FORECAST_DATETIME'] = dt_str
    
    # Create netcdf4 Dataset to use wrf-python 
    wofs_netcdf = Dataset(filepath)
    return wofs, wofs_netcdf

def load_wofs_files(filepaths, filename_prefix, datetime_format, engine='netcdf4',
                   seconds_since='seconds since 2001-01-01', DB=0, **kwargs):
    ''' TODO: maybe. stretch
    (not implemented) Load multiple WoFS files
    '''
    wofs = xr.open_mfdataset(filepaths, concat_dim='Times', combine='nested', 
                             parallel=True, engine=engine, **kwargs) #concat_dim='patch'

    for f, filepath in enumerate(filepaths):
        wofs_datetime = wofs.Times.data[f]
        _datetime, dt_int, dt_str = get_wofs_datetime(filepath, filename_prefix=filename_prefix, 
                                                wofs_datetime=wofs_datetime, seconds_since=seconds_since, 
                                                datetime_format=datetime_format, DB=DB)
        wofs = wofs.reindex(Time=[dt_int])
        wofs_netcdf = Dataset(filepath)

    #return wofs
    #pass

def create_wofs_time(year, month, day, hour, min, sec, 
                     seconds_since='seconds since 2001-01-01', DB=0):
    ''' 
    Create corresponding datatime object and return the date in GridRad time 
    seconds since

    @param year: integer year of the corresponding datetime
    @param month: integer month of the corresponding datetime
    @param day: integer day of the corresponding datetime
    @param hour: integer hour of the corresponding datetime
    @param min: integer min of the corresponding datetime
    @param sec: integer sec of the corresponding datetime
    @param seconds_since: string of the form since describing the time units. 
            can be days, hours, minutes, seconds, milliseconds or microseconds. 
            see netcdf4.date2num() documentation for more information
    @param DB: int debug flag to print out additional debug information

    @return: the WoFS datetime as a np.datetime int
    '''
    dtime = datetime.datetime(year, month, day, hour, min, sec)
    dtime = date2num(dtime, seconds_since)
    if DB: print(f"WoFS time: {dtime}")
    return dtime 

def get_wofs_datetime(fnpath, filename_prefix=None, wofs_datetime=None, 
                      datetime_format='%Y-%m-%d_%H:%M:%S', 
                      seconds_since='seconds since 2001-01-01', DB=0):
    ''' 
    Construct a datetime object for the raw WoFS file name. Raw WoFS file names 
    are of the form: wrfwof_d01_2019-05-18_07:35:00 ('YYYY-MM-DD_hh:mm:ss'). 
    However, this method should be generic enough for most formats, assuming the 
    datetime is part of the filename.

    @param fnpath: filename or filename path of the file. If a file path is 
            provided, the filename is extracted
    @param filename_prefix: (optional) string prefix used for the file name if 
            wofs_datetime is not provided
    @param wofs_datetime: (optional) string forecast datetime of the WoFS data 
            if filename_prefix not provided. string indicates the datetime of 
            the WoFS forecast
    @param datetime_format: (optional) string indicating the expected datetime 
            format for the datetimes. default '%Y-%m-%d_%H:%M:%S'
    @param seconds_since: string of the form since describing the time units. 
            can be days, hours, minutes, seconds, milliseconds or microseconds. 
            see netcdf4.date2num() documentation for more information
    @param DB: int debug flag to print out additional debug information

    @return: a 3-tuple with the datetime object, the datetime int 
            since seconds_since and the formatted datetime string for the WoFS 
            data file
    '''
    fname = os.path.basename(fnpath) 
    fname, file_extension = os.path.splitext(fname)

    if not filename_prefix is None:
        fname = fname.replace(filename_prefix, '')
        wofs_datetime, _ = parse_date(fname, fuzzy_with_tokens=True)
        if DB: print(f"Extracted {wofs_datetime} from file {fnpath}:{fname}")

    datetime_obj = wofs_datetime
    if not isinstance(datetime_obj, datetime):
        # Convert datetime string to datetime object
        datetime_obj = datetime.strptime(wofs_datetime, datetime_format) #datetime_format
        if DB: print(f"\nExtracted datetime object:: {wofs_datetime}={datetime_obj} from file {fnpath}({fname})")

    # Convert datetime object to np.datetime int
    datetime_int = date2num(datetime_obj, seconds_since)
    if DB: print(f"Extracted datetime int in {seconds_since}:: {datetime_int} from file {fnpath}({fname})")

    # Convert datetime object to string with the desired format
    datetime_str = datetime_obj.strftime(datetime_format)
    if DB: print(f"Extracted datetime string:: {datetime_str} from file {fnpath}({fname})\n")

    return datetime_obj, datetime_int, datetime_str

def compute_forecast_window(init_time, forecast_time, DB=0):
    ''' 
    Calculate the forecast window as the time between the WoFS initialization 
    time and the WoFS forecast time.

    @param init_time: string indicating the initialization time of the WoFS 
            simulation.
    @param forecast_time: string indicating the forecast time of the WoFS data
    @param DB: int debug flag to print out additional debug information

    @return: integer duration of the forecast window in minutes
    '''
    # Initialization time
    t0 = time.fromisoformat(init_time) 
    # Forecast time
    t1 = time.fromisoformat(forecast_time) 

    forecast_window = (t1 - t0).total_seconds() / 60
    if DB: print(f"Forecast window: {forecast_window} min\n")  

    return forecast_window

def extract_netcdf_dataset_fields(args, wrfin, gridrad_heights):
    ''' 
    Extract WoFS fields reflectivity, vorticity, divergence from the netcdf4 
    Dataset

    @param args: parsed command line args. see create_argsparser()
                ZH_only: TODO flag whether to only extract reflectivity
                fields: TODO list of fields to extract
    @param wrfin: netCDF Dataset, not xarray, is required to use wrf-python 
            functions
    @param gridrad_heights: list or numpy array of the elevations in the GridRad 
            data in km

    @return: 4-tuple of numpy arrays containing the reflectivity, divergence, 
            vorticity, and updraft helicity. ( Z_agl, div, vort, uh)
    '''
    # Get wofs heights
    height = wrf.getvar(wrfin, "height_agl", units='m')

    # Get reflectivity
    Z = wrf.getvar(wrfin, "REFL_10CM")
    
    # Interpolate wofs data to gridrad heights
    gridrad_heights = gridrad_heights * 1000
    Z_agl = wrf.interplevel(Z, height, gridrad_heights)

    if args.dry_run:
        print("wofs heights", height.data.shape)
        print("wofs heights quantiles", np.quantile(height.data.ravel(), [0, .5, 1]))
        print("wofs.REFL_10CM (Z) shape", Z.shape)
        print("Z_agl shape", Z_agl.shape)
        print("wofs.REFL_10CM.data ravel shape", Z.data.ravel().shape)
        print(" extract:: # nan wofs", np.isnan(Z.data.ravel()).sum())
        print(" extract:: # nan gridrad", np.isnan(Z_agl.data.ravel()).sum())
        print(" extract:: # inf wofs", np.isinf(Z.data.ravel()).sum())
        print(" extract:: # inf gridrad", np.isinf(Z_agl.data.ravel()).sum())

        Z_max = np.nanmax(Z)
        Z_agl_max = np.nanmax(Z_agl)
        #_mx = np.max([Z_max, Z_agl_max])
        print(f" extract netcdf:: Zmax={Z_max} Zaglmax={Z_agl_max}")
        
        wofs_basefname = os.path.basename(args.loc_wofs)
        figsize = (12, 12)
        dpi = 300
        save = (args.write in [3, 4])

        dir_figs = args.dir_preds if args.dir_figs is None  else args.dir_figs
        fname = wofs_basefname + f'__debug_Z00.png'
        fnpath = os.path.join(dir_figs, fname)
        print(fname)
        plot_pcolormesh(args, Z[0], fnpath, 
                        title='Z0 (WoFS Grid)', cb_label='dBZ', cmap="Spectral_r", 
                        vmin=0, vmax=60, dpi=dpi, figsize=figsize, close=True, save=save)
        fname = wofs_basefname + f'__debug_Zs.png'
        fnpath = os.path.join(dir_figs, fname)
        fig, axs = plt.subplots(5, 10, figsize=(25, 12))
        fig.suptitle('WoFS Grid')
        axs = axs.ravel()
        nwh = Z.shape[0] # n wofs heights
        for i, ax in enumerate(axs):
            cb_label = None if i < nwh-1 else 'dBZ'
            plot_pcolormesh(args, Z[i], fnpath, fig_ax=(fig, ax, axs.tolist()),
                            title=f'Z{i}', vmin=0, vmax=60, 
                            cb_label=cb_label, cmap="Spectral_r", dpi=550, 
                            figsize=figsize, close=(i == nwh-1), save=(i == nwh-1))

        fname = wofs_basefname + f'__debug_Zhist.png'
        fnpath = os.path.join(dir_figs, fname)
        print(fname) #Z.stack()
        zvals = np.nan_to_num(Z.data.ravel(), copy=True, nan=-20)
        plot_hist(args, zvals, title='WoFS Grid', xlabel='Reflectivity',
                  ylabels=['density', 'cum density'], fname=fnpath, figsize=(8, 6),
                  dpi=120) #, **hist_args)
        
        fname = wofs_basefname + f'__debug_Z00_gridrad.png'
        fnpath = os.path.join(dir_figs, fname)
        print(fname)
        plot_pcolormesh(args, Z_agl[0], fnpath, 
                        title='Z0 (GridRad Grid)', cb_label='dBZ', cmap="Spectral_r", 
                        vmin=0, vmax=60, dpi=dpi, figsize=figsize, close=True, save=save)
        fname = wofs_basefname + f'__debug_Zs_gridrad.png'
        fnpath = os.path.join(dir_figs, fname)
        fig, axs = plt.subplots(3, 4, figsize=(15, 12))
        axs = axs.ravel()
        for i, ax in enumerate(axs):
            if i > 11: break
            cb_label = None if i < 11 else 'dBZ'
            plot_pcolormesh(args, Z_agl[i], fnpath, fig_ax=(fig, ax, axs.tolist()),
                            title=f'$Z{i}$ (GridRad Grid)', vmin=0, vmax=60, 
                            cb_label=cb_label, cmap="Spectral_r", dpi=400, 
                            figsize=figsize, close=(i == 11), save=(i == 11))
        
        fname = wofs_basefname + f'__debug_Zhist_gridrad.png'
        fnpath = os.path.join(dir_figs, fname)
        print(fname)
        zaglvals = np.nan_to_num(Z_agl.data.ravel(), copy=True, nan=-20)
        plot_hist(args, zaglvals, title='GridRad Grid', xlabel='Reflectivity',
                  ylabels=['density', 'cum density'], fname=fnpath, figsize=(8, 6),
                  dpi=120) #, **hist_args)


    div = None
    vort = None
    if not args.ZH_only:
        # Get U and V winds
        U = wrf.g_wind.get_u_destag(wrfin)
        V = wrf.g_wind.get_v_destag(wrfin)  

        # Interpolate wofs data to gridrad heights
        U_agl = wrf.interplevel(U, height, gridrad_heights)
        V_agl = wrf.interplevel(V, height, gridrad_heights)
        
        # Add units to winds to use metpy functions
        U = U_agl.values * (metpy.units.units.meter / metpy.units.units.second)
        V = V_agl.values * (metpy.units.units.meter / metpy.units.units.second)
        
        # Define grid spacings (for div and vort)
        #dx = 3000 * (metpy.units.units.meter) 
        #dy = 3000 * (metpy.units.units.meter) 
        dx = wrfin.DX * (metpy.units.units.meter) 
        dy = wrfin.DY * (metpy.units.units.meter) 

        # Calculate divergence and vorticity
        div = metpy.calc.divergence(U, V, dx=dx, dy=dy)
        vort = metpy.calc.vorticity(U, V, dx=dx, dy=dy)
        
        # Grab data, remove metpy.units and xarray.Dataset stuff 
        div = np.asarray(div)
        vort = np.asarray(vort)
    
    Z_agl = Z_agl.values 
    uh = wrf.getvar(wrfin, 'UP_HELI_MAX')
    uh = uh.values

    wrfin.close()

    return Z_agl, div, vort, uh

def make_patches(args, radar, window):
    '''
    Create a Dataset of concatenated along the patches

    @param args: parsed command line args
            Relevant args: see create_argsparser for more details
                ZH_only: only extract ZH
                patch_shape: shape of the patches
    @param radar: WoFS data
    @param window: duration time in minutes of the forecast window (i.e., the 
                    time between the initialization of the simulation and the 
                    time of the current forecast)

    @return: xarray Dataset of patches
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

            # Create the patch
            data_vars = dict(
                            ZH=(["x", "y", "z"], radar.ZH.isel(Latitude=slice(xi, xi + xsize), 
                                                              Longitude=slice(yi, yi + ysize), 
                                                              time=0).values.swapaxes(0,2).swapaxes(1,0)), 
                            UH=(["x", "y"], radar.UH.isel(Latitude=slice(xi, xi + xsize), 
                                                         Longitude=slice(yi, yi + ysize), 
                                                         time=0, Altitude=0).values),
                            stitched_x=(["x"], range(xi, xi + xsize)),
                            stitched_y=(["x"], range(yi, yi + ysize)),
                            n_convective_pixels = ([], np.count_nonzero(radar.ZH.isel(Latitude=slice(xi, xi + xsize),
                                                                                     Longitude=slice(yi, yi + ysize), 
                                                                                     time=0).fillna(0).values.swapaxes(0,2).swapaxes(1,0).max(axis=(2)) >= 30)),
                            n_uh_pixels = ([], np.count_nonzero(radar.UH.isel(Latitude=slice(xi, xi + xsize), 
                                                                             Longitude=slice(yi, yi + ysize),
                                                                             time=0).fillna(0).values > 0)),
                            lat=([], radar.Latitude.values[xi]),
                            lon=([], radar.Longitude.values[yi]),
                            time=([], radar.time.values[0]),
                            forecast_window=([], window))
            if not Z_only:
                div = dict(DIV=(["x", "y", "z"], radar.DIV.isel(Latitude=slice(xi, xi + xsize), 
                                                                Longitude=slice(yi, yi + ysize), 
                                                                time=0).values.swapaxes(0,2).swapaxes(1,0)))
                vor = dict(VOR=(["x", "y", "z"], radar.VOR.isel(Latitude=slice(xi, xi + xsize), 
                                                                Longitude=slice(yi, yi + ysize), 
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

def impute_nans(data, method, value=None, k=3):
    '''
    Impute NaN values using the specified method. If method='value' and value is
            set, use the value
    :param data: the nd array to impute
    :param method: string or function handle for the method to use. options are
            'value': use the provided value in the parameter value

            'kmean': use the mean of the k-neighbors
            'kmedian': use the median of the k-neighbors
            'kmin': use the min of the k-neighbors
            'kmax': use the max of the k-neighbors

            'mean': use the overall mean of the data
            'median': use the overall median of the data
            'min': use the overall min of the data
            'max': use the overall max of the data
    :param value: float. specific value to set all NaNs to if method == 'value'
    :param k: int for the square neighborhood size to consider
    :return: the imputed data array
    '''
    # dict of summary function options
    funcs = {'mean': np.nanmean, 'median': np.nanmedian, 
               'min': np.nanmin, 'max': np.nanmax}
    
    are_nan = np.isnan(data)
    
    data_imputed = None
    if method == 'value' and value is not None:
        data_imputed = np.nan_to_num(data, True, value)

    elif method in ['kmean', 'kmedian', 'kmin', 'kmax']:
        data_filter = data.copy()
        key = method[1:]
        sum_func = funcs[key]
        data_filter = generic_filter(data_filter, sum_func, size=k)

        data_imputed = data.copy()
        data_imputed[are_nan] = data_filter[are_nan]

    elif method in ['mean', 'median', 'min', 'max']:
        sum_func = funcs[key]
        data_imputed = data.copy()
        data_imputed[are_nan] = sum_func(data)

    else: 
        raise ValueError(f'impute_nans:: method must be one of [value, kmean, kmedian, kmin, kmax, mean, median, min, max]. but was {method}')
    return data_imputed

def to_gridrad(args, wofs, wofs_netcdf, method=0, gridrad_spacing=48,
               gridrad_heights=np.arange(1, 13, step=1, dtype=int), DB=0):
    '''
    Convert WoFS data into the GridRad grid. If with_nans is set, change grid points 
    with reflectivity=0 to NaN

    @param args: command line args. see create_argsparser()
            Relevant args:
                dir_patches: directory to save patched WoFS data in the GridRad grid
                write: flag whether to the patched data
                with_nans: flag whether to set reflectivity values that are 0 to NaN
                dry_run: 
    @param wofs: xarray Dataset
    @param wofs_netcdf: netcdf Dataset
    @param method: int to select interpolation method (default method==0)
            method==0: use scipy.interpolate.RectBivariateSpline (prefered)
            method==1: use scipy.spatial.cKDTree
    @param gridrad_spacing: Gridrad files have grid spacings of 1/48th degrees lat/lon
            1 / (gridrad_spacing) degrees
    @param gridrad_heights: 1D array of the elevation levels in the GridRad data. 
            Height values we want to interpolate to
    @param DB: int debug flag to print out additional debug information

    @return: xarray Dataset with the interpolated and patched data
    '''
    DB = args.dry_run
    Z_only = args.ZH_only
    dir_figs = args.dir_preds if args.dir_figs is None  else args.dir_figs

    # There is a difference sometimes between the day in the filepath and the day in the filename
    # All filepath dates have '_path' and all filename dates have '_day'
    
    # lat/lon points for regridded grid points
    new_gridrad_lats_lons, new_gridrad_lats, new_gridrad_lons = calculate_output_lats_lons(wofs, 
                                                                    gridrad_spacing=gridrad_spacing)
    lats_len = new_gridrad_lats.size
    lons_len = new_gridrad_lons.size
    
    # Pull out the desired data from the wofs file
    Z_agl, div, vort, uh = extract_netcdf_dataset_fields(args, wofs_netcdf, gridrad_heights)

    # Remove axes of length 1
    xlat_vals = np.squeeze(wofs.XLAT.values)
    xlon_vals = np.squeeze(wofs.XLONG.values)


    nheights = Z_agl.shape[0]
    REFL_10CM_final = np.zeros((1, nheights, lats_len, lons_len), float)
    uh_final = np.zeros((1, lats_len, lons_len), float)
    vort_final = None
    div_final = None
    if method == 0:
        # WoFS latitude and longitude coordinates to interpolate from
        wlats = xlat_vals[:, 0]
        wlons = xlon_vals[0, :]

        for h in range(nheights):
            are_nan = np.isnan(Z_agl[h])
            print(f" {h}: # NANs Z_agl =", np.isnan(Z_agl[h].ravel()).sum())
            Z_imputed = impute_nans(Z_agl[h], method='kmedian', value=None, k=3)
            print(f" {h}: # NANs Z_imputed =", np.isnan(Z_imputed.ravel()).sum())

            Z_interpolator = RectBivariateSpline(wlats, wlons, Z_imputed) #Z_agl[h]) #

            # Evaluate each (x,y) coordinate
            REFL_10CM_final[0, h] = Z_interpolator(new_gridrad_lats, new_gridrad_lons, grid=True)
            print(f" {h}: # NANs REFL_10CM_final =", np.isnan(REFL_10CM_final[0, h].ravel()).sum())

        # Interpolate updraft helicity
        uh_interpolator = RectBivariateSpline(wlats, wlons, uh) 
        uh_final[0] = uh_interpolator(new_gridrad_lats, new_gridrad_lons, grid=True)
        
        # Interpolate divergence and vorticity
        if not Z_only:
            nvort_levels = vort.shape[0]
            vort_final = np.zeros((1, lats_len, lons_len, nvort_levels), float)
            div_final = np.zeros((1, lats_len, lons_len, nvort_levels), float)

            for h in range(nvort_levels):
                v_interp = RectBivariateSpline(wlats, wlons, vorts[h]) 
                vort_final[0, :, :, h] = v_interp(new_gridrad_lats, new_gridrad_lons, grid=True)

                d_interp = RectBivariateSpline(wlats, wlons, divs[h]) 
                div_final[0, :, :, h] = d_interp(new_gridrad_lats, new_gridrad_lons, grid=True)

    elif method == 1:
        # List of all lat/lon grid points from original wofs grid
        wofs_lats_lons = np.stack((xlat_vals.ravel(), xlon_vals.ravel())).T

        # KD Tree with wofs grid points
        tree = spatial.cKDTree(wofs_lats_lons)
        
        # For each gridrad point, find k(=4) closest wofs grid points
        distances, points = tree.query(new_gridrad_lats_lons, k=4)
        if DB:
            print(" new_gridrad_lats_lons", new_gridrad_lats_lons.shape)
            print(" wofs_lats_lons", wofs_lats_lons.shape)
            print(" # inf dists", np.isinf(distances).sum())
        
        # For points found above, identify each wofs coordinate value (bottom_top is constant)
        d1_len = xlat_vals.shape[0]
        d2_len = xlat_vals.shape[1]
        time_points, south_north_points, west_east_points = np.unravel_index(points, (1, d1_len, d2_len)) 
        #np.unravel_index(points, (1, 300, 300))
        
        # Repeat the x,y indices 29 times to select all data
        heights_len = gridrad_heights.size
        south_north = south_north_points.reshape(south_north_points.shape[0], 1, south_north_points.shape[1])
        south_north_points_3d = np.tile(south_north, (1, heights_len, 1))
        west_east = west_east_points.reshape(west_east_points.shape[0], 1, west_east_points.shape[1])
        west_east_points_3d = np.tile(west_east, (1, heights_len, 1))
        distances_3d = np.tile(distances.reshape(distances.shape[0], 1, distances.shape[1]), (1, heights_len, 1))

        # Make z coordinate indices. TODO: This might be able to be improved, but works fine.
        for i in range(heights_len):
            if i == 0:
                bottom_top_3d = np.zeros((south_north_points.shape[0],
                            south_north_points.shape[1], 1), dtype=int) 
            else:
                bottom_top_3d = np.append(bottom_top_3d, np.ones((south_north_points.shape[0], 
                            south_north_points.shape[1], 1), dtype=int) * i, axis=2)
        bottom_top_3d = np.swapaxes(bottom_top_3d, 1, 2)

        # Select data 
        refls = Z_agl[bottom_top_3d, south_north_points_3d, west_east_points_3d]
        uhs = uh[south_north_points, west_east_points]

        if not Z_only:
            vorts = vort[bottom_top_3d, south_north_points_3d, west_east_points_3d]
            divs = div[bottom_top_3d, south_north_points_3d, west_east_points_3d]

        # Perform IDW interpolation and reshape data to shape to (Time, Altitude, Latitude, Longitude)
        dist3d_sq = distances_3d**2
        dist_sq = distances**2
        if DB:
            print("uh shape", uhs.shape)
            print(f"dist3d_sq={np.quantile(dist3d_sq, [0, .5, 1])}")
            print(f"dist_sq={np.quantile(dist_sq, [0, .5, 1])}")

        # x, y coords for wofs grid and equivalent for gridrad, separate for each level
        REFL_10CM_final = np.swapaxes(np.swapaxes((np.sum(refls / dist3d_sq, axis=2) / np.sum(1. / dist3d_sq, axis=2)).reshape(1, lats_len, lons_len, Z_agl.shape[0]), 1, 3), 2, 3).astype(np.float32)
        uh_final = (np.sum(uhs / dist_sq, axis=1) / np.sum(1. / dist_sq, axis=1)).reshape(1, lats_len, lons_len).astype(np.float32)

        if not Z_only:
            vort_final = np.swapaxes(np.swapaxes((np.sum(vorts / dist3d_sq, axis=2) / np.sum(1 / dist3d_sq, axis=2)).reshape(1, lats_len, lons_len, vort.shape[0]), 1, 3), 2, 3).astype(np.float32)
            div_final = np.swapaxes(np.swapaxes((np.sum(divs / dist3d_sq, axis=2) / np.sum(1 / dist3d_sq, axis=2)).reshape(1, lats_len, lons_len, vort.shape[0]), 1, 3), 2, 3).astype(np.float32)

    if DB:
        wofs_basefname = os.path.basename(args.loc_wofs)
        figsize = (12, 12)
        dpi = 300

        print(" uh shape", uh.shape)
        print(" refls final", REFL_10CM_final.shape)
        print(f" to gridrad:: Zaglmax={np.nanmax(REFL_10CM_final)}")
        print(" n NANs gridrad", np.isnan(REFL_10CM_final.ravel()).sum())
        print(" n inf gridrad", np.count_nonzero(np.isinf(REFL_10CM_final.ravel())))

        _Zcomposit = np.nanmax(REFL_10CM_final[0], axis=0)
        fname = wofs_basefname + f'__debugtogridrad_Z_composite_gridrad.png'
        fnpath = os.path.join(dir_figs, fname)
        print(fname)
        plot_pcolormesh(args, _Zcomposit, fnpath, 
                        title=f'Z Max (GridRad Grid)', vmin=0, vmax=60, 
                        cb_label='dBZ', cmap="Spectral_r", dpi=dpi, 
                        figsize=(12, 10), close=True, save=(args.write in [3, 4]))

        fname = wofs_basefname + f'__debugtogridrad_Z00_gridrad.png'
        fnpath = os.path.join(dir_figs, fname)
        print(fname)
        plot_pcolormesh(args, REFL_10CM_final[0, 0], fnpath, 
                        title=f'Z00 (GridRad Grid)', vmin=0, vmax=60, 
                        cb_label='dBZ', cmap="Spectral_r", dpi=dpi, 
                        figsize=(12, 10), close=True, save=(args.write in [3, 4]))
        
        fname = wofs_basefname + f'__debugtogridrad_Zs_gridrad.png'
        fnpath = os.path.join(dir_figs, fname)
        print(fname)
        fig, axs = plt.subplots(3, 4, figsize=(15, 12))
        axs = axs.ravel()
        for i, ax in enumerate(axs):
            cb_label = None if i < 11 else 'dBZ'
            plot_pcolormesh(args, REFL_10CM_final[0, i], fnpath, 
                            fig_ax=(fig, ax, axs.tolist()),
                            title=f'Z{i} (GridRad Grid)', vmin=0, vmax=60, 
                            cb_label=cb_label, cmap="Spectral_r", dpi=400, 
                            figsize=figsize, close=(i == 11), save=(i == 11))

        fname = wofs_basefname + f'__debugtogridrad_Zhist_gridrad.png'
        fnpath = os.path.join(dir_figs, fname)
        print(fname)
        refl_final = np.nan_to_num(REFL_10CM_final.ravel(), copy=True, nan=-20)
        plot_hist(args, refl_final, title='GridRad Grid (all levels)', xlabel='Reflectivity',
                  ylabels=['density', 'cum density'], fname=fnpath, figsize=(8, 6),
                  dpi=120) #, **hist_args)

    
    # Put data into DataArrays with same dimensions, coordinates, and variable fields as gridrad
    dtime = wofs['Time'].values[0]
    wofs_regridded_refc = xr.DataArray(
        data=REFL_10CM_final,
        dims=("time", "Altitude", "Latitude", "Longitude"),
        coords={"time": [dtime], "Altitude": gridrad_heights, 
                "Latitude": new_gridrad_lats, "Longitude": new_gridrad_lons + 360},
    )
    wofs_regridded_uh = xr.DataArray(
        data=uh_final,
        dims=("time", "Latitude", "Longitude"),
        coords={"time": [dtime], "Latitude": new_gridrad_lats, 
                "Longitude": new_gridrad_lons + 360},
    )
    if not Z_only:
        wofs_regridded_vort = xr.DataArray(
            data=vort_final,
            dims=("time", "Altitude", "Latitude", "Longitude"),
            coords={"time": [dtime], "Altitude": gridrad_heights, 
                    "Latitude": new_gridrad_lats, "Longitude": new_gridrad_lons + 360}
        )
        wofs_regridded_div = xr.DataArray(
            data=div_final,
            dims=("time", "Altitude", "Latitude", "Longitude"),
            coords={"time": [dtime], "Altitude": gridrad_heights, 
                    "Latitude": new_gridrad_lats, "Longitude": new_gridrad_lons + 360},
        )
    
    # Combine DataArrays into a Dataset
    wofs_regridded = xr.Dataset(coords={"time": [dtime], 
                                        "Latitude": new_gridrad_lats, 
                                        "Longitude": new_gridrad_lons + 360, 
                                        "Altitude": gridrad_heights})
    wofs_regridded["ZH"] = wofs_regridded_refc
    wofs_regridded["UH"] = wofs_regridded_uh
    if not Z_only:
        wofs_regridded["VOR"] = wofs_regridded_vort
        wofs_regridded["DIV"] = wofs_regridded_div
    
    # Where reflectivity=0, set to nans
    if args.with_nans:
        wofs_regridded = wofs_regridded.where(wofs_regridded.ZH > 0)

        if DB:
            fname = wofs_basefname + f'__debugtogridrad_with_nans_Z00_gridrad.png'
            fnpath = os.path.join(dir_figs, fname)
            print(fname)
            plot_pcolormesh(args, wofs_regridded["ZH"][0, 0], fnpath, 
                            title=f'Z00 (GridRad Grid)', vmin=0, vmax=60, 
                            cb_label='dBZ', cmap="Spectral_r", dpi=dpi, 
                            figsize=(12, 10), close=True, save=(args.write in [3, 4]))
            
            fname = wofs_basefname + f'__debugtogridrad_with_nans_Zs_gridrad.png'
            fnpath = os.path.join(dir_figs, fname)
            print(fname)
            fig, axs = plt.subplots(3, 4, figsize=(15, 12))
            axs = axs.ravel()
            for i, ax in enumerate(axs):
                cb_label = None if i < 11 else 'dBZ'
                plot_pcolormesh(args, wofs_regridded["ZH"][0, i], fnpath, 
                                fig_ax=(fig, ax, axs.tolist()),
                                title=f'Z{i} (GridRad Grid)', vmin=0, vmax=60, 
                                cb_label=cb_label, cmap="Spectral_r", dpi=400, 
                                figsize=figsize, close=(i == 11), save=(i == 11))

            fname = wofs_basefname + f'__debugtogridrad_with_nans_Zhist_gridrad.png'
            fnpath = os.path.join(dir_figs, fname)
            print(fname)
            refl_final = np.nan_to_num(wofs_regridded["ZH"].data.ravel(), copy=True, nan=-20)
            plot_hist(args, refl_final, title='GridRad Grid (all levels)', xlabel='Reflectivity',
                    ylabels=['density', 'cum density'], fname=fnpath, figsize=(8, 6),
                    dpi=120)

    # Calculate forecast window
    datetime_forecast_str = wofs.Times.data[0].decode('ascii') # decode from binary str to regular text str
    datetime_forecast = datetime.fromisoformat(datetime_forecast_str)
    datetime_init = datetime.fromisoformat(wofs.START_DATE) 
    forecast_window = (datetime_forecast - datetime_init).seconds / 60
    if DB: print(f"Forecast window: {forecast_window} min\n")

    # Make patches
    ds_patches = make_patches(args, wofs_regridded, forecast_window)
    #ds_patches['Times'] = wofs.Times
    ds_patches.attrs = wofs.attrs
    
    if args.write in [2, 4] and not args.dir_patches is None:
        fname = os.path.basename(wofs.filenamepath)
        fname, _ext = os.path.splitext(fname)
        patch_shape = [f'{c:03d}' for c in args.patch_shape]
        patch_shape_str = '_'.join(patch_shape)
        savepath = os.path.join(args.dir_patches, f'{fname}_patched_{patch_shape_str}.nc')
        print(f"Saving patched WoFS data interpolated to GridRad grid to {savepath}\n")
        #>>ds_patches.to_netcdf(savepath) #, engine='netcdf4'

    return ds_patches, wofs_regridded


"""
Tornado prediction methods.
"""
def load_trainset_stats(args, engine='netcdf4', DB=0, **kwargs):
    ''' 
    Open args.file_trainset_stats with the mean and std of the training data. These 
    will be used for normalizing the data prior to prediction

    @param args: command line args. see create_argsparser()
            Relevant args:
                file_trainset_stats: file path to the mean and STD of the fields
                fields: list of fields
    @param engine: Engine to use when reading file. see xarray Dataset 
            documentation 
    @param DB: int debug flag to print out additional debug information
    @param kwargs: keyword args for xr.open_dataset()

    @return: the dataset with the means and stds of the features
    '''
    fpath = args.file_trainset_stats

    if DB: print("Training meta data used:", fpath)

    train_stats = xr.open_dataset(fpath, engine=engine, **kwargs).load() #, cache=False
    train_stats.close()

    return train_stats

def predict(args, wofs, stats, from_weights=False, eval=False, DB=0, **fss_args):
    '''
    Load the model and perform the predictions.
    TODO: predict on arbitrary set of fields

    @param args: command line args. see create_argsparser()
            Relevant attributes:
                loc_model: location to the the trained model to use
    @param wofs: data to predict on
    @param stats: data field statistics such as mean and standard deviation of 
            the WoFS fields from the training set data
    @param from_weigths: boolean flag, whether to load the model from weights
    @param eval: TODO whether to compute evaluation results. only available if
            the true labels are also available
    @param DB: int debug flag to print out additional debug information
    @param fss_args: keyword args for make_fractions_skill_score()
            default: {'mask_size': 2, 'num_dimensions': 2, 'c':1.0, 
                        'cutoff': 0.5, 'want_hard_discretization': False}

    @return: the predictions as a numpy array
    '''

    model_path = args.loc_model

    if DB: print("Loading model:", model_path)

    model = None
    if not from_weights:
        if fss_args is None or fss_args == {}:
            fss_args = {'mask_size': 2, 'num_dimensions': 2, 'c':1.0, 
                        'cutoff': 0.5, 'want_hard_discretization': False}
        fss = make_fractions_skill_score(**fss_args)
        model = keras.models.load_model(model_path, custom_objects={'fractions_skill_score': fss, 
                                        'MaxCriticalSuccessIndex': MaxCriticalSuccessIndex,
                                        'GELU': GELU})

    else:
        # TODO: args.hp_path args.hp_idx
        # Convert csv to Keras Hyperparameters
        df_hps = pd.read_csv(args.hp_path)
        best_hps = df_hps.drop(columns=['Unnamed: 0', 'args'])
        hps_dict = best_hps.iloc[args.hp_idx].to_dict()

        # Set hyperparameters
        hp = HyperParameters()
        for k, v in hps_dict.items(): 
            hp.Fixed(k, value=v)

        #(32, 32, 12)
        hmodel = UNetHyperModel(input_shape=wofs.ZH.shape[1:], n_labels=1)
        model = hmodel.build(hp)
        model.load_weights(model_path)

    # Normalize the reflectivity data
    ZH_mu = float(stats.ZH_mean.values)
    ZH_std = float(stats.ZH_std.values)
    X = (wofs.ZH - ZH_mu) / ZH_std
    #Xds = tf.data.Dataset.from_tensor_slices(X).batch(512)

    # Predict with the data
    if DB: 
        print(f"Performing predictions (wofs dims={wofs.dims}) ...")
        print(f"Performing predictions (wofs coords={wofs.coords}) ...")
    print(f"Performing predictions (input shape={X.shape}) ...")
    preds = model.predict(X)

    if DB:
        print(" preds precalib quantiles [0, .25, .5, .75, 1]:", 
              np.nanquantile(preds, [0, .25, .5, .75, 1]))
    # Calibrate the model predictions
    if args.loc_model_calib:
        with open(args.loc_model_calib, 'rb') as fid:
            calib = pickle.load(fid)
            shape = preds.shape
            preds = calib.predict(preds.reshape(-1, 1))
            preds = preds.reshape(shape)
            if DB:
                print(" preds calib quantiles [0, .25, .5, .75, 1]:", 
                      np.nanquantile(preds, [0, .25, .5, .75, 1]))

    return preds

def combine_fields(args, wofs, preds, gridrad_heights=range(1, 13), DB=0):
    '''
    Combine the fields of interest with true and predicted data into a single Dataset
    Combine the predictions, reflectivity, U, V.
    TODO: extend to arbitrary sets of fields

    @param args: command line args. see create_argsparser()
                Relevent attributes:
                    ZH_only: flag whether to only include reflectivity
                    fields: list of additional fields to write
    @param wofs: xarray Dataset with all the wofs data
    @param preds: the predictions from the reflectivity
    @param gridrad_heights: indicies of the vertical levels 
    @param DB: int debug flag to print out additional debug information

    @return: xarray dataset with the combined fields and predictions
    '''
    fields = ['ZH', 'UH', 'stitched_x', 'stitched_y', 'n_convective_pixels', 
              'n_uh_pixels', 'lat', 'lon', 'time', 'forecast_window']
    if not args.ZH_only:
        fields += ['DIV', 'VOR']
    '''if not args.fields is None:
        fields += args.fields
        fields = set(fields)'''
    wofs_preds = wofs[fields].copy(deep=True)

    dims = dict(wofs.dims)
    coords = dict(wofs.coords)
    coords.pop('z')
    print(wofs.dims)
    print(wofs.coords)

    wofs_preds['ZH_composite'] = xr.DataArray(
        data=wofs.ZH.values.max(axis=3), #["patch", "x", "y"]
        #dims=dims,
        coords=coords
    )
    _preds = np.zeros((*preds.shape[:-1], 2))
    # If only the prob of tornado is provided, expand the data to 2D
    if preds.shape[-1] == 1:
        _p = np.squeeze(preds.copy())
        _preds[:, :, :, 1] = _p     # probab of tor
        _preds[:, :, :, 0] = 1 - _p # probab of no tor
    else: _preds = preds.copy()
    wofs_preds['predicted_no_tor'] = xr.DataArray(
        data=_preds[:, :, :, 0], #["patch", "x", "y"]
        #dims=wofs.dims,
        coords=coords
    ) 
    wofs_preds['predicted_tor'] = xr.DataArray(
        data=_preds[:, :, :, 1], #["patch", "x", "y"]
        #dims=wofs.dims,
        coords=coords
    )

    return wofs_preds

def to_wofsgrid(args, wofs_orig, wofs_gridrad, stats, gridrad_spacing=48, 
                 method=0, seconds_since='seconds since 2001-01-01', DB=0):
    '''
    Interpolate the WofS predictions data back into the original WoFS grid

    @param args: command line args. see crate_argsparser()
            Relevant arguements:
                dir_preds: directory to save the WoFS grid predictions
                ZH_only: flag whether to only include reflectivity
                fields: list of additional fields to write
                write: flag whether to write out the predictions
    @param wofs_orig: original WoFS data prior to interpolating to GridRad. Used 
            to obtain the original WoFS grid spacing
    @param wofs_gridrad: WoFS data in GridRad grid system, with the predictions
    @param stats: dataset containing the mean and STD from the training data
    @param gridrad_spacing: Gridrad files have grid spacings of 1/48th degrees lat/lon
            1 / (gridrad_spacing) degrees
    @param method: int to select interpolation method (default method==0)
            method==0: use scipy.interpolate.RectBivariateSpline (prefered)
            method==1: use scipy.spatial.cKDTree
    @param seconds_since: string seconds since date statement to use for creating
            NETCDF4 datetime integers. string of the form since describing the 
            time units. can be days, hours, minutes, etc. see netcdf4.date2num() 
            documentation for more information
    @param DB: int debug flag to print out additional debug information

    @return: a tuple with stitched WoFS prediction patches (in gridrad system), 
            and the WoFS data as an xarray Dataset in the original WoFS grid 
            system
    '''
    # Stitch back together the patches of the predictions
    predictions =  stitch_patches(args, wofs_gridrad, stats, 
                                    gridrad_spacing=gridrad_spacing, 
                                    seconds_since=seconds_since, DB=DB)
    
    wofs_lats = wofs_orig.XLAT.values[0, :, 0] #wofs_orig.XLAT.values[0].ravel()
    wofs_lons = wofs_orig.XLONG.values[0, 0, :] #wofs_orig.XLONG.values[0].ravel()
    predictions_idw = np.zeros((wofs_lats.size, wofs_lons.size), float)
    gridrad_lats = None
    gridrad_lons = None
    if method == 0:
        _, gridrad_lats, gridrad_lons = calculate_output_lats_lons(wofs_orig, gridrad_spacing=gridrad_spacing)

        _preds = predictions.predicted_tor.isel(time=0)
        p_interpolator = RectBivariateSpline(gridrad_lats[:-1], gridrad_lons[:-1],
                                             _preds) 
        # Evaluate each (x,y) coordinate
        predictions_idw = p_interpolator(wofs_lats, wofs_lons, grid=True)
        if DB:
            print(" preds towofs quantiles [0, .25, .5, .75, 1]:", 
                  np.nanquantile(predictions_idw, [0, .25, .5, .75, 1]))

        # Clip values outside the range [0,1]
        predictions_idw[predictions_idw < 0] = 0
        predictions_idw[predictions_idw > 1] = 1
        if DB:
            print(" preds clipped quantiles [0, .25, .5, .75, 1]:", 
                  np.nanquantile(predictions_idw, [0, .25, .5, .75, 1]))

    elif method == 1:
        # Calculate number of grid points required to add to each side to get to the full WoFS grid
        to_add_east_west = round((wofs_orig.XLONG.max().values - predictions.lon.max().values + 360) * gridrad_spacing + 1)
        to_add_north_south = round((wofs_orig.XLAT.max().values - predictions.lat.max().values) * gridrad_spacing + 1)

        # Pad the top, bottom, left and right of the grid toobtain the exact size of the original WoFS file grid
        zeros_east_west = np.zeros((predictions.predicted_tor.isel(time=0).shape[0], to_add_east_west))
        padded_east_west = np.concatenate((zeros_east_west, predictions.predicted_tor.isel(time=0).values, zeros_east_west), axis=1)
        zeros_north_south = np.zeros((to_add_north_south, padded_east_west.shape[1]))
        padded_predictions = np.concatenate((zeros_north_south, padded_east_west, zeros_north_south), axis=0)

        # Calculate lats and lons of gridrad grid, which is padded with 0s on each side
        padded_lats = np.linspace(predictions.lat.min().values - to_add_north_south / gridrad_spacing, 
                                    predictions.lat.max().values + to_add_north_south / gridrad_spacing, padded_predictions.shape[0])
        padded_lons = np.linspace(predictions.lon.min().values - to_add_east_west / gridrad_spacing, 
                                    predictions.lon.max().values + to_add_east_west / gridrad_spacing, padded_predictions.shape[1])

        # Place lats and lons into 2D arrays from 1D arrays
        plons_len = padded_lons.size
        plats_len = padded_lats.size
        gridrad_lats = np.tile(padded_lats, plons_len).reshape(plons_len, plats_len).T
        gridrad_lons = np.tile(padded_lons - 360, plats_len).reshape(plats_len, plons_len)
        # Combine these 2 arrays into 1
        gridrad_lats_lons = np.stack((np.ravel(gridrad_lats), np.ravel(gridrad_lons))).T
        
        # Extract the lats and lons from the wofs grid which we are interpolating to
        wofs_lats_lons = np.stack((np.ravel(wofs_orig.XLAT.values[0]), np.ravel(wofs_orig.XLONG.values[0]))).T
        
        # Make the KD Tree
        tree = spatial.cKDTree(gridrad_lats_lons)

        # Query the tree at the desired points
        # For each wofs point, find the closest gridrad gridpoint
        distances, points = tree.query(wofs_lats_lons, k=4)

        # Get the indices of the interpolation points in the new padded gridrad grid
        lat_points, lon_points = np.unravel_index(points, (plats_len, plons_len))

        # Use inverse distance weighting to get the new grid
        predictions_not_idw = padded_predictions[lat_points, lon_points]
        dist_sq = distances**2
        predictions_idw = (np.sum(predictions_not_idw / dist_sq, axis=1) / np.sum(1. / dist_sq, axis=1)).reshape(wofs_orig.XLAT.values.shape[1:])

        if args.dry_run:
            print(f" to_wofs:: dist_sq={np.quantile(dist_sq, [0, .5, 1])}")
            print(" padded_predictions", padded_predictions.shape)
            print(" predictions_not_idw", predictions_not_idw.shape)

    if args.dry_run:
        print(" to_wofs:: predictions.lat", predictions.lat.shape)
        print(" gridrad_lats", gridrad_lats.shape)
        print(" predictions.lat", np.quantile(predictions.lat, [0, .5, 1]))
        print(" gridrad_lats", np.quantile(gridrad_lats, [0, .5, 1]))
        print(" predictions.lon", predictions.lon.shape)
        print(" gridrad_lons", gridrad_lons.shape)
        print(" predictions.lon", np.quantile(predictions.lon, [0, .5, 1]))
        print(" gridrad_lons", np.quantile(gridrad_lons, [0, .5, 1]))
        print(" predictions.predicted_tor.isel(time=0)", predictions.predicted_tor.isel(time=0).shape)


    # Create Dataset formatted like the raw WoFS file
    wofs_like = xr.Dataset(data_vars=dict(ML_PREDICTED_TOR=(["Time", "south_north", "west_east"], 
                                            predictions_idw.reshape((1,) + predictions_idw.shape))),
                            coords=dict(XLAT=(["Time", "south_north", "west_east"], wofs_orig.XLAT.values),
                                            XLONG=(["Time", "south_north", "west_east"], wofs_orig.XLONG.values)),
                            attrs=wofs_orig.attrs
                            )

    # Include select WoFS fields
    fields = ['COMPOSITE_REFL_10CM', 'REFL_10CM', 'Times', 'UP_HELI_MAX']
    if not args.ZH_only:
        fields += ['U', 'U10', 'V', 'V10']
    if not args.fields is None:
        #, 'REL_VORT', 'W', 'W_UP_MAX'
        fields += args.fields
        fields = set(fields)
    wofs_fields = wofs_orig[fields].copy(deep=True)
    wofs_like = xr.merge([wofs_like, wofs_fields])
    if DB:
        print("dims\n", wofs_like.dims)
        print("coords\n", wofs_like.coords)
        print("data_vars\n", wofs_like.data_vars)
        print("attr 'START_DATE'\n", wofs_like.START_DATE)

    # Save out the interpolated file
    if args.write in [1, 2, 4]:
        savepath = wofs_orig.filenamepath.replace('wofs-preds-2023-hgt', 'wofs-preds-2023-update')
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        print(f"Save WoFS grid predictions to {savepath}\n")
        wofs_like.to_netcdf(savepath)
    
    return predictions, wofs_like, savepath

def stitch_patches(args, wofs, stats, gridrad_spacing=48, 
                   seconds_since='seconds since 2001-01-01', DB=0):
    """ 
    Reconstruct the stitched grid for a single WoFS forecast time

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
    total_in_lat = int(wofs.stitched_x.max().values + 1)
    total_in_lon = int(wofs.stitched_y.max().values + 1)

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

    # Loop through all the patches
    zeros = np.ones((xsize, ysize))
    ZH_mu = float(stats.ZH_mean.values)
    ZH_std = float(stats.ZH_std.values)
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
    
    if DB:
        print(" preds stitched quantiles [0, .25, .5, .75, 1]:", 
              np.nanquantile(tor_preds, [0, .25, .5, .75, 1]))
    
    # Compute average of each patch by dividing by the total number of patches that contained each pixel
    tor_preds = tor_preds / overlap_array
    uh_array = uh_array / overlap_array
    zh = zh_low_level
    zh_low_level = zh_low_level / overlap_array

    if DB:
        print(" preds stitched-normed quantiles [0, .25, .5, .75, 1]:", 
              np.nanquantile(tor_preds,[0, .25, .5, .75, 1]))

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


"""
Visualization methods
"""
def contour_fmt(x):
    return f"{x:.02f}"
    #return rf"{s}" if plt.rcParams["text.usetex"] else f"{s}"

def plot_hist(args, data, title, xlabel, ylabels, fname, show_text=True,
              figsize=(8, 6), dpi=120, save=False, **hist_args):
    '''
    @param args: command line args. see create_argsparser()
        Relevant arguments
            write
            #dir_preds
            #dir_figs
    @param data: data to plot
    @param fname: name of the file with the image extension
    @param title: figure title
    @param xlabel: xlabel
    @param ylabel: list of length two of the ylabels for each plot
    @param figsize: as tuple for the figure width and height
    @param dpi: integer for the saved figure resolution as dots per inch
    @param **hist_args: key word args for np.histogram

    @return: the fig and the axes objects
    '''
    hvals, bins = np.histogram(data, **hist_args) #bins='fd', density=False)
    # Normalize the hist values

    hvals = hvals / np.sum(hvals)
    # Center the bins
    delta = (bins[1] - bins[0]) / 2
    cbins = bins[:-1] + delta

    fig, ax = plt.subplots(2, 1, figsize=figsize)
    #vals, bins, patches = ax.hist(hvals, bins=bins, histtype='bar', label='Histogram')
    #vals, bins, patches = ax.hist(hvals, bins=bins, histtype='step', label='Cumulative' , cumulative=True)
    ax[0].plot(cbins, hvals, label='Histogram')
    ax[0].set(title=title, ylabel=ylabels[0]) #title='Tor Predictions Distribution', ylabel='density'
    if show_text:
        for i, (c, v) in enumerate(zip(cbins, hvals)):
            if i < 3 or len(cbins) < 12:
                txt = f'{v:.2e}% ' if v < .01 else f'{v:.02f}' #f'{v:.02f}'
                ax[0].text(c, v+.01, txt, fontsize=11) 
    cvals = np.cumsum(hvals)
    ax[1].plot(cbins, cvals, label='Cumulative')
    ax[1].set(xlabel=xlabel, ylabel=ylabels[1]) #xlabel='Probability', ylabel='cumulative density'
    
    if args.write in [3, 4]:
        #dir_figs = args.dir_preds if args.dir_figs is None  else args.dir_figs
        #fpath = os.path.join(dir_figs, fname)
        print("  Saving", fname)
        plt.savefig(fname, dpi=dpi)
    plt.close()

    return fig, ax

def plot_pcolormesh(args, data, fname, title, cb_label=None, fig_ax=None,
                    data_contours=None, cmap="Spectral_r", vmin=0, vmax=50, 
                    alpha=None, dpi=250, figsize=(10, 9), tight=True, 
                    close=False, save=False):
    '''
    Use matplotlib pcolormesh to plot the data

    @param args: command line args. see create_argsparser()
            Relevant arguments
                #write
                #dir_preds
                #dir_figs
    @param data: data to plot
    @param data_contours: data to render in overlaping contours
            https://stackoverflow.com/questions/10490302/how-do-you-create-a-legend-for-a-contour-plot-in-matplotlib
            https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot
    @param fname: name of the file with the image extension
    @param title: figure title
    @param cb_label: colorbar label
    @param fig_ax: tuple with the first element is th figure and the second 
            element is the Axes object
    @param cmap: color map. See matplotlib for the colormap options
    @param vmin: color bar min value
    @param vmax: color bar max value
    @param alpha: float indicating the alpha value for the color plot
    @param dpi: integer for the saved figure resolution as dots per inch
    @param figsize: as tuple for the figure width and height
    @param tight: bool whether to use plt.tight_layout() on the figure
    @param close: bool whether to close the figure after plotting and saving
    @param save: bool whether to save the figure

    @return: the fig and the axes objects
    '''
    fig = None
    ax = None
    if fig_ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = fig_ax[0]
        ax = fig_ax[1]

    pcmesh = ax.pcolormesh(data, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha)
    if not data_contours is None:
        # Contour levels
        qlevels = [0, .2, .5, .9, 1]
        data_qtiles = np.nanquantile(data_contours, qlevels)
        clevels = [.2, .5] #data_qtiles[2:4]
        ccolors = ['dimgray', 'yellow']
        contour = ax.contour(data_contours, levels=clevels, #qlevels[1:3], 
                             colors=ccolors) #, cmap="cividis") 
        #[.1, .5] #[0. 0.08 0.16 0.24 0.32 0.4  0.48 0.56 0.64]
        print(" ** contour levels", data_qtiles, qlevels[1:3], np.quantile(data_contours, qlevels), contour.levels)
        #proxy = [plt.Rectangle((0,0), 1, 1, fc=pc.get_edgecolor()[0]) for pc in contour.collections]
        proxy = [plt.Rectangle((0,0), 1, 1, fc=pc.get_edgecolor()[0]) for i, pc in enumerate(contour.collections)]
        ax.legend(proxy, 
                  #[rf'$\geq${q:.2e} {l*100}th% ' if q < .01 else rf'$\geq${q:.02f} {l*100}th% ' for q, l in zip(data_qtiles[1:3], qlevels[1:3])], 
                  [rf'$\geq${q:.2e}% ' if q < .01 else rf'$\geq${q:.02f}' for q in clevels], 
                  loc='upper left', bbox_to_anchor=(1.16, 1), title='Tor Probability') #, alignment='left'

    ax.set_aspect('equal', 'box') #.axis('equal') #
    ax.set_title(title)

    if tight: plt.tight_layout()

    if not cb_label is None:
        _axs = ax if fig_ax is None else fig_ax[2]
        cb_ = fig.colorbar(pcmesh, ax=_axs) #fig.colorbar(pcmesh, ax=ax)
        cb_.set_label(cb_label, rotation=-90, labelpad=15)

    if save: 
        #dir_figs = args.dir_preds if args.dir_figs is None  else args.dir_figs
        #fpath = os.path.join(dir_figs, fname)
        print("  Saving", fname)
        plt.savefig(fname, dpi=dpi)

    if close: plt.close()

    return fig, ax

def create_gif(args, wofs, suffix, field0=None, field1=None, interval=50, 
               figsize=(10, 9), dpi=250, DB=0, **kwargs):
    ''' 
    Create a .gif or movie file of the storm data
    matplotlib documentation 
    https://www.c-sharpcorner.com/article/create-animated-gif-using-python-matplotlib/

    @param args: command line args. see create_argsparser()
            Relevant arguments
                write
                dir_preds
                dir_figs
    @param wofs: WoFS data 
    @param suffix: file name suffix to append to the file name
    @param field0: TODO WoFS field to render in the colormap
    @param field1: TODO WoFS field to render as contours
    @param interval: Delay between frames in milliseconds. See matplotlib 
            FuncAnimation documentation for more information
    @param figsize: tuple with the width and height of the figure
    @param DB: int debug flag to print out additional debug information
    @param kwargs: additional keyword args for the animator
            repeat: bool, default: True. Whether the animation repeats when the 
                    sequence of frames is completed.
            repeat_delay: int, default: 0. The delay in milliseconds between 
                    consecutive animation runs, if repeat is True.
    '''
    from matplotlib.animation import FuncAnimation, PillowWriter
    #import imageio

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    nframes = wofs['patch'].size

    def plotter(data_mesh, data_contour, i, ax, cmap="Spectral_r", vmin=0, vmax=60): 
        '''
        :param data_mesh: data to plot in pclolrmesh
        :param data_contour: data to plot as contours
        :param i: index
        '''
        ZH_distr = ax.pcolormesh(data_mesh, cmap=cmap, vmin=vmin, vmax=vmax)

        ax.contour(data_contour)
        ax.set_aspect('equal', 'box') #.axis('equal') #
        ax.set_title('Composite Reflectivity')
        ax.text(0, -3, f'patch {i:03d}')
        return ZH_distr, ax

    vmin = 0
    vmax = 60
    def draw_storm_frame(pi):
        '''
        Draw a patch as a single frame
        @param pi: patch index (aka draw frame index)
        '''
        ax.clear()

        plotter(wofs.ZH_composite.values[pi], wofs.predicted_tor.values[pi], pi, 
                ax, cmap="Spectral_r", vmin=vmin, vmax=vmax)

        '''
        ZH_distr = ax.pcolormesh(wofs.ZH_composite.values[pi], 
                                 cmap="Spectral_r", vmin=0, vmax=50)
        if pi == 0: 
            cb_ZH_distr = fig.colorbar(ZH_distr, ax=ax)
            cb_ZH_distr.set_label('dBZ', rotation=-90)

        ax.contour(wofs.predicted_tor.values[pi])
        ax.set_aspect('equal', 'box') #.axis('equal') #
        ax.set_title('Composite Reflectivity')
        ax.text(0, -3, f'patch {pi:03d}')
        '''

        fig.canvas.draw() # draw the canvas, cache renderer
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        WIDTH, HEIGHT = fig.canvas.get_width_height()
        image = image.reshape(HEIGHT, WIDTH, 3) # 3 for RGB
        return image

    #fndir = os.path.join(self.subjcode, self.session)
    #fnpath = os.path.join(fndir, f"{self.session}_kinematics_movie.gif")
    fname = os.path.basename(wofs.filenamepath) + f'__{suffix}.gif'

    # Plot first frame ot get colorbar to render properly
    ZH_distr, ax = plotter(wofs.ZH_composite.values[0], 
                           wofs.predicted_tor.values[0], 0,
                           ax, cmap="Spectral_r", vmin=vmin, vmax=vmax)
    cb_ZH_distr = fig.colorbar(ZH_distr, ax=ax)
    cb_ZH_distr.set_label('dBZ', rotation=-90)

    animator = FuncAnimation(fig, draw_storm_frame, frames=nframes, 
                             interval=interval, **kwargs)
    print(" n fig axes", len(fig.axes)) #fig.delaxes(fig.axes[-1])

    if args.write in [3, 4]:
        dir_figs = args.dir_preds if args.dir_figs is None  else args.dir_figs
        fnpath = os.path.join(dir_figs, fname)
        print("  Saving", fnpath)
        #writer = PillowWriter(fps=30)
        animator.save(fnpath, dpi=dpi) #, writer=writer)

    return fig, animator


def wofs_to_preds(wofs_filepath, args):
    
    DB = args.dry_run

    if DB: 
        print(args)
        print("FIELDS", args.fields, len(args.fields))

    wofs_files = []
    if os.path.isfile(wofs_filepath):
        wofs_files = wofs_filepath #.replace(":", "_")
    #TODO elif os.path.isdir(args.loc_wofs):
    #    wofs_files = os.listdir(args.loc_wofs)
    else: 
        raise ValueError(f"[ARGUMENT ERROR] --loc_wofs should be a file, but was {wofs_filepath}")
    
    # Opens all data files into a Dataset
    print("Open WoFS file(s)", wofs_files)

    #datetime_fmt = '%Y-%m-%d_%H:%M:%S' #YYYY-MM-DD_hh-mm-ss
    datetime_fmt = args.datetime_format
    print("Datetime format:", datetime_fmt)

    SECS_SINCE = 'seconds since 2001-01-01'
    ds_load_args = {'decode_times': False} #{'cache': False, 
    wofs, wofs_netcdf = load_wofs_file(wofs_files, args.filename_prefix, 
                                       wofs_datetime=None,
                                       datetime_format=datetime_fmt,
                                       seconds_since=SECS_SINCE, 
                                       engine='netcdf4', DB=DB, **ds_load_args)
                                       
    # Interpolate WoFS grid to GridRad grid
    GRIDRAD_SPACING = 48
    GRIDRAD_HEIGHTS = np.arange(1, 13, step=1, dtype=int)
    wofs_gridrad, wofs_regridded = to_gridrad(args, wofs, wofs_netcdf, 
                                        method=args.interp_method,
                                        gridrad_spacing=GRIDRAD_SPACING, 
                                        gridrad_heights=GRIDRAD_HEIGHTS, DB=DB)

    # Prepare data for prediction
    # Load training data statistics
    train_stats = load_trainset_stats(args, DB=DB)

    # Compute predictions
    from_weights = not args.load_options is None
    preds = predict(args, wofs_gridrad, train_stats, from_weights=from_weights, DB=DB) #, **fss_args)

    # TODO: optional display/output performance

    # Combine the predictions, reflectivity and select fields into a single dataset
    wofs_combo = combine_fields(args, wofs_gridrad, preds, DB=DB)

    # Interpolate back to WoFS grid
    preds_gridrad_stitched, preds_wofsgrid, savepath = to_wofsgrid(args, wofs, wofs_combo, 
                                           train_stats, method=args.interp_method,
                                           gridrad_spacing=GRIDRAD_SPACING, 
                                           seconds_since=SECS_SINCE, DB=DB)


    # Plot data
    if args.write in [3, 4]:
        save = True
        dir_figs = args.dir_preds if args.dir_figs is None  else args.dir_figs

        anim_args = {'repeat': True, 'repeat_delay': 100}
        create_gif(args, wofs_combo, 'patches_ZH_distr', interval=550, 
                   figsize=(10, 9), DB=DB, **anim_args)

        wofs_basefname = os.path.basename(wofs_files)

        Z_max = np.nanmax(wofs.REFL_10CM.values[0, 0])
        Z_agl_max = np.nanmax(preds_gridrad_stitched.ZH_1km[0])
        _mx = np.max([Z_max, Z_agl_max])

        # Original reflectivity
        fname = wofs_basefname + f'__ZH_level00.png'
        fnpath = os.path.join(dir_figs, fname)
        if DB: print(" ++ w shape REFL_10CM",  wofs.REFL_10CM.values.shape)
        plot_pcolormesh(args, wofs.REFL_10CM.values[0, 0], fnpath, 
                        title='Reflectivity 10 cm', cb_label='dBZ', 
                        cmap="Spectral_r", vmin=0, vmax=60, dpi=300, figsize=(10, 9), 
                        close=True, save=save)

        # Reflectivity GRIDRAD
        fname = wofs_basefname + f'__ZH_level00_gridrad.png'
        fnpath = os.path.join(dir_figs, fname)
        if DB:
            print("  ++ g shape ZH", wofs_gridrad.ZH.shape)
            print("  ++ g shape ZH_1km", preds_gridrad_stitched.ZH_1km.shape)
            print("  ++ regrid shape ZH", wofs_regridded.ZH.shape)
        plot_pcolormesh(args, preds_gridrad_stitched.ZH_1km[0], fnpath, #wofs_gridrad.ZH[0,:,:,0]
                        title='Reflectivity (GridRad Grid)', cb_label='Reflectivity', 
                        cmap="Spectral_r", vmin=0, vmax=60, dpi=300, figsize=(10, 9), 
                        close=True, save=save)
        
        fname = wofs_basefname + f'__ZH_level00_gridrad1.png'
        fnpath = os.path.join(dir_figs, fname)
        plot_pcolormesh(args, wofs_regridded.ZH[0, 0], fnpath, 
                        title='Reflectivity (GridRad Grid)', 
                        cb_label='Reflectivity', cmap="Spectral_r", vmin=0, vmax=60, 
                        dpi=300, figsize=(10, 9), close=True, save=save)


        # Predictions WOFS GRID
        pred_max = np.nanmax(preds_wofsgrid.ML_PREDICTED_TOR[0])
        if DB:
            print(" preds plot quantiles [0, .25, .5, .75, 1]:", 
                  np.nanquantile(preds_wofsgrid.ML_PREDICTED_TOR[0], 
                                 [0, .25, .5, .75, 1]))
        fname = wofs_basefname + f'__predictions.png'
        fnpath = os.path.join(dir_figs, fname)
        plot_pcolormesh(args, preds_wofsgrid.ML_PREDICTED_TOR[0], fnpath, 
                        title='Predicted Tor (WoFS Grid)', cb_label='$p_{tor}$', 
                        cmap="cividis", vmin=0, vmax=pred_max, dpi=300, 
                        figsize=(10, 9), close=True, save=save)
        # ^^ With contours
        fname = wofs_basefname + f'__predictions_contours.png'
        fnpath = os.path.join(dir_figs, fname)
        plot_pcolormesh(args, preds_wofsgrid.ML_PREDICTED_TOR[0], fnpath, 
                        title='Predicted Tor (WoFS Grid)', cb_label='$p_{tor}$', 
                        data_contours=preds_wofsgrid.ML_PREDICTED_TOR[0],
                        cmap="Spectral_r", vmin=0, vmax=pred_max, alpha=.6, 
                        dpi=300, figsize=(15, 9), close=True, save=save) #cividis
        # PROBABS Hist
        fname = wofs_basefname + f'__predictions_hist.png'
        fnpath = os.path.join(dir_figs, fname)
        preds_data = np.nan_to_num(preds_wofsgrid.ML_PREDICTED_TOR[0], copy=True, nan=-0.1)
        
        hist_args = {'bins': 'fd', 'density': False}
        fig, ax = plot_hist(args, preds_data, title='Tor Predictions Distribution', 
                            xlabel='Probability', ylabels=['density', 'cum. density'], 
                            show_text=False, fname=fnpath, figsize=(8, 6), **hist_args)

        # Predictions GRIDRAD GRID
        fname = wofs_basefname + f'__predictions_gridrad.png'
        fnpath = os.path.join(dir_figs, fname)
        plot_pcolormesh(args, preds_gridrad_stitched.predicted_tor[0], fnpath, 
                        title='Predicted Tor (GridaRad Grid)', cb_label='$p_{tor}$', 
                        cmap="cividis", vmin=0, vmax=pred_max, dpi=300, figsize=(10, 9), close=True, save=save) #1


        # Composite Reflectivity with prediction contours (WoFS)
        fname = wofs_basefname + f'__ZH_composite_predictions.png'
        fnpath = os.path.join(dir_figs, fname)
        if DB:
            print(" ++ shape predictions", preds_wofsgrid.ML_PREDICTED_TOR.shape)
            print(" ++ shape composite", preds_wofsgrid.COMPOSITE_REFL_10CM.shape)
        ZH_composite = preds_wofsgrid.COMPOSITE_REFL_10CM.values[0] #.nanmax(axis=3)
        plot_pcolormesh(args, ZH_composite, fnpath, 
                        title='Composite Reflectivity - (WoFS Grid)', cb_label='dBZ', 
                        data_contours=preds_wofsgrid.ML_PREDICTED_TOR[0], 
                        cmap="Spectral_r", vmin=0, vmax=60, alpha=.6, dpi=300, 
                        figsize=(15, 9), close=True, save=save)
        
        # Updraft
        fname = wofs_basefname + f'__updraft_helicity_predictions.png'
        fnpath = os.path.join(dir_figs, fname)
        uh_min = np.nanmin(preds_wofsgrid.UP_HELI_MAX.values[0]) #.ML_PREDICTED_TOR[0])
        uh_max = np.nanmax(preds_wofsgrid.UP_HELI_MAX.values[0]) #.ML_PREDICTED_TOR[0])
        if DB: print(" ++ shape uh", preds_wofsgrid.UP_HELI_MAX.shape)
        UH = preds_wofsgrid.UP_HELI_MAX.values[0] #.nanmax(axis=3)
        plot_pcolormesh(args, UH, fnpath, 
                        title='Updraft Helicity (WoFS Grid)', 
                        cb_label='', data_contours=preds_wofsgrid.ML_PREDICTED_TOR[0],
                        cmap="Spectral_r", vmin=uh_min, vmax=uh_max, alpha=.6, dpi=300, 
                        figsize=(15, 9), close=True, save=True)

        npatches = wofs_combo.ZH_composite.values.shape[0]
        for pi in range(0, npatches, 25):
            fig, ax = plt.subplots(1, 1, figsize=(10, 9))
            ZH_distr = ax.pcolormesh(wofs_combo.ZH_composite.values[pi], 
                                     cmap="Spectral_r", vmin=0, vmax=60)
            ax.contour(wofs_combo.predicted_tor.values[pi])
            ax.set_aspect('equal', 'box') #.axis('equal') #
            ax.set_title('Composite Reflectivity')
            cb_ZH_distr = fig.colorbar(ZH_distr, ax=ax)
            cb_ZH_distr.set_label('dBZ', rotation=0)

            fname = wofs_basefname + f'__patch{pi:03d}_ZH_distr.png'
            fpath = os.path.join(dir_figs, fname)
            print("  Saving", fpath)
            plt.savefig(fpath, dpi=250)
            

            # Predictions heatmap
            fname = wofs_basefname + f'__patch{pi:03d}_ZH_distr_preds.png'
            fnpath = os.path.join(dir_figs, fname)
            plot_pcolormesh(args, wofs_combo.predicted_tor.values[pi], fnpath, 
                            title=f'Prediction (patch {pi:03d})', 
                            cb_label='$p_{tor}$', cmap="coolwarm", vmin=0, vmax=1, 
                            dpi=200, figsize=(9, 8), close=True, save=True)

    # Close Datasets
    #wofs_netcdf.close()
    #ds_stitched.close()
    #ds_wofs_as_gridrad.close()
    print(f"PREDICTIONS COMPLETE FOR {wofs_files}.")
    return savepath
