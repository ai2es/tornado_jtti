"""
author: Monique Shotande 
Functions based on those provided by Lydia

End to end script takes as input the raw WoFS (warn on Forecast System) data,
and outputs the predictions in the WoFS grid.

General Procedure:
1. read raw WoFS file(s)
2. interpolate WoFS grid to GridRad grid
3. construct patches (optional)
4. make predictions
5. interpolate back to WoFS grid

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
    method get_arguments(). Run
        python lydia_scripts/wofs_raw_predictions.py --h
        python lydia_scripts/wofs_raw_predictions.py --loc_wofs 
                                                     --datetime_format 
                                                     --datetime_init
                                                     (--filename_prefix | --datetime_forecast)
                                                     --dir_preds
                                                     --with_nans 
                                                     --loc_model
                                                     --file_training_metadata
                                                     --dry_run
"""
import re, os, sys, glob, argparse
from datetime import datetime, date, time
from dateutil.parser import parse as parse_date
import xarray as xr
print("xr version", xr.__version__)
import numpy as np
print("np version", np.__version__)
#import netCDF4
#print("netCDF4 version", netCDF4.__version__)
from netCDF4 import Dataset, date2num
#import scipy
from scipy import spatial
#print("scipy version", scipy.__version__)
import wrf #wrf-python=1.3.2.5==py38h0e9072a_0
print("wrf-python version", wrf.__version__)
import metpy
import metpy.calc
#import multiprocessing as mp
#import tqdm

import tensorflow as tf
print("tensorflow version", tf.__version__)
from tensorflow import keras
print("keras version", keras.__version__)

# Working directory expected to be tornado_jtti/
sys.path.append("lydia_scripts")
from custom_losses import make_fractions_skill_score
from custom_metrics import MaxCriticalSuccessIndex
from scripts_data_pipeline.wofs_to_gridrad_idw import calculate_output_lats_lons #, extract_gridrad_data_fields
sys.path.append("process_monitoring")
from process_monitor import ProcessMonitor
print(" ")


def create_argsparser(args_list=None):
    ''' 
    Create command line arguments parser

    @param args_list: list of strings with command line arguments to override
            any received arguments. default value is None and args are not overridden

    @return: the argument parser
    '''
    if not args_list is None:
        sys.argv = args_list #TODO: test
        
    parser = argparse.ArgumentParser(description='Tornado Prediction end-to-end from WoFS data', epilog='AI2ES')

    # WoFS file(s) path 
    '''group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--dir_wofs', type=str, #required=True, 
        help='Directory path to raw WoFS file(s)')
    #TODO subparser if file, init datetime and forecast datetime
    group.add_argument('--file_wofs', type=str, #required=True, 
        help='File path to the raw WoFS file')'''

    parser.add_argument('--loc_wofs', type=str, required=True, 
        help='Location of the WoFS file(s). Can be a path to a single file or a directory to several files')

    parser.add_argument('--datetime_format', type=str, required=True, 
        help='Date time format string used') # in the WoFS file name
    parser.add_argument('--datetime_init', type=str, required=True, 
        help='Date time of the initialization data used to produce the WoFS forecast data. Date time should follow the format provided in --datetime_format')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--filename_prefix', type=str, #required=True, 
        help='Prefix used in the WoFS file name')
    group.add_argument('--datetime_forecast', type=str, # required=True, 
        help='Forecast date time of the WoFS file following the format provided in --datetime_format')

    parser.add_argument('--dir_preds', type=str, required=True, 
        help='Directory to store the predictions. Prediction files are saved individually for each WoFS files. The prediction files are saved of the form: <WOFS_FILENAME>_predictions_<BOOL_PATCHED>.nc')

    #parser.add_argument('-p', '--patch_size', type=int, required=True, 
    #    help='Size of patches in each horizontal dimension. Ex: patch_size=32 would make patches of shape (32, 32, 12).')
    parser.add_argument('-p', '--patch_shape', type=tuple, default=(32, 32, 12), #required=True, 
        help='Shape of patches. Can be empty (), 2D (xy, h), 3D (x, y, h), or 4D (x, y, c, h) tuple. If empty tuple, patching is not performed. Last dimension must contain the number of GridRad elevation levels. If 2D, the x and y dimension are the same. Ex: (32, 12) or (32, 32, 12).')
    parser.add_argument('--with_nans', action='store_true', 
        help='Set flag such that data points with reflectivity=0 are stored as NaNs. Otherwise store as normal floats.')
    parser.add_argument('-Z', '--Z_only', action='store_true',
        help='Use flag to only extract the reflectivity and updraft data, excluding divergence and vorticity. Regardless of the value of this flag, only reflectivity is used for training and prediction.')
    parser.add_argument('-f', '--fields', type=list,
        help='List of additional WoFS fields to store. Regardless of whether these fields are specified, only reflectivity is used for training and prediction.')

    # Model directories and files
    parser.add_argument('--loc_model', type=str, required=True,
        help='Trained model directory or file path (i.e. file descriptor)')
    parser.add_argument('--file_training_metadata', type=str, required=True,
        help='Path to training metadata file to train the model being evaluating. Contains the means and std of the training data')
    #parser.add_argument('--dir_checkpoint', type=str, required=True,
    #    help='directory where the trained model is stored')

    parser.add_argument('-d', '--dry_run', action='store_true',
        help='For testing and debugging. Execute without running models or saving data and display output paths')

    #parser.add_argument('--exp', type=str, required=True,
    #    help='Experimental index')

    return parser

def parse_args(args_list=None):
    '''
    Create and parse the command line args parser

    @param args_list: list of strings with command line arguments to override
            any received arguments

    @return: the parsed arguments object
    '''
    parser = create_argsparser(args_list=args_list)
    args = parser.parse_args()
    return args

"""
WoFS to GridRad interpolation methods, refactored from lydia_scripts/scripts_data_pipeline
"""
def load_wofs_file(filepaths, filename_prefix=None, wofs_datetime=None, 
                   datetime_format='%Y-%m-%d_%H:%M:%S', seconds_since='seconds since 2001-01-01', 
                   engine='netcdf4', DB=0, **kwargs):
    ''' TODO
    Load the raw WoFS file as xarray dataset and netcdf dataset

    @param filepaths: list of WoFS file paths
    @param filename_prefix: prefix used for all the file names
    @param datetime_format: date time format the date time is expected to be in
    @param engine: TODO see documentation for xarray Dataset
    @param kwargs: any additional desired parameters to xarray.open_dataset(...)
    
    @return: xarray Dataset, netCDF4 Dataset
    '''
    if DB: print(f"Loading {filepaths}")

    # Create xarray dataset
    wofs = xr.open_dataset(filepaths[0], engine=engine, **kwargs)
    
    # TODO: if isintance(filepath, list)
    #wofs = xr.open_mfdataset(wofs_files, concat_dim='patch', combine='nested', 
    #                         parallel=True, engine=engine, **kwargs)

    _datetime, dt_int, dt_str = get_wofs_datetime(filepaths[0], filename_prefix=filename_prefix, 
                                                  wofs_datetime=wofs_datetime, seconds_since=seconds_since, 
                                                  datetime_format=datetime_format, DB=DB)
    wofs = wofs.reindex(Time=[dt_int])
    #wofs.Times.data[0]
    
    # Create netcdf4 dataset to use wrf-python 
    wofs_netcdf = Dataset(filepaths[0])
    return wofs, wofs_netcdf

def create_wofs_time(year, month, day, hour, min, sec, 
                     seconds_since='seconds since 2001-01-01', DB=0):
    ''' TODO
    Create corresponding data time object and return the date in GridRad time as seconds since

    @param year, month, day, hour, min, sec, 
    @param seconds_since: 

    @return: the WoFS datetime as a np.datetime int
    '''
    dtime = datetime.datetime(year, month, day, hour, min, sec)
    dtime = date2num(dtime, seconds_since)
    if DB: print(f"WoFS time: {dtime}")
    return dtime 

def get_wofs_datetime(fnpath, filename_prefix=None, wofs_datetime=None, 
                      datetime_format='%Y-%m-%d_%H:%M:%S', 
                      seconds_since='seconds since 2001-01-01', DB=0):
    ''' TODO
    Construct a datetime object from the raw WoFS file name. Raw WoFS files are 
    of the form: wrfwof_d01_2019-05-18_07:35:00 ('YYYY-MM-DD_hh:mm:ss'). However, this method  
    should be generic enough for most formats, assuming the datetime is part of the filename.

    @param fnpath: filename or filenamepath of the file. If a file path is provided,
            the filename is extracted
    @param filename_prefix: (optional) string prefix used for all the file names if wofs_datetime 
            not provided
    @param wofs_datetime: (optional) wofs_datetime if filename_prefix not provided. string 
            indicating the datetime of the WoFS forecast
    @param datetime_format: (optional) string indicating the expected date time 
            format for all datetimes. default '%Y-%m-%d_%H:%M:%S'
    @param seconds_since: 

    @return: a 3-tuple with the datetime object, the datetime int since seconds_since and the 
            formatted datetime string for the WoFS data file
    '''
    #re.findall('\d{4}-\d{2}-%d{2}_\d{2}:\d{2}:\d{2}', wofs_files)
    #wrfwof_d01_2019-05-18_07:35:00
    #datetime_fmt = '2019-05-18_07:35:00'
    #datetime_fmt = args.datetime_format
    #date.fromisoformat('2019-12-04') #datetime.date(2019, 12, 4)
    #date.strftime(datetime_fmt)
    #datetime_fmt = 'YYYY-MM-DD_hh:mm:ss' #%Y-%m-%d_%H:%M:%S
    #dateparser.parse('wrfwof_d01_2019-05-18_07:35:00', date_formats='YYYY-MM-DD_hh:mm:ss')
    #path_components = patches_dirs.split('/')

    fname = os.path.basename(fnpath) 
    fname, file_extension = os.path.splitext(fname)

    if not filename_prefix is None:
        fname = fname.replace(filename_prefix, '')
        wofs_datetime, _ = parse_date(fname, fuzzy_with_tokens=True)
        if DB: print(f"Extracted {wofs_datetime} from file {fnpath}:{fname}")

    datetime_obj = wofs_datetime
    if not isinstance(datetime_obj, datetime):
        # Convert datetime string to datetime object
        datetime_obj = datetime.strptime(wofs_datetime, datetime_fmt)
        if DB: print(f"\nExtracted datetime object:: {wofs_datetime}={datetime_obj} from file {fnpath}({fname})")

    datetime_int = date2num(datetime_obj, seconds_since)
    if DB: print(f"Extracted datetime int in {seconds_since}:: {datetime_int} from file {fnpath}({fname})")

    datetime_str = datetime_obj.strftime(datetime_format)
    if DB: print(f"Extracted datetime string:: {datetime_str} from file {fnpath}({fname})\n")

    return datetime_obj, datetime_int, datetime_str

def compute_forecast_window(init_time, forecast_time, DB=0): #hh, mm):
    ''' TODO
    Calculate the forecast window

    @param init_time: string indicating the initialization time of the WoFS simulation.
    @param forecast_time: string indicating the forecast time of the WoFS data

    @return: the duration of the forecast window in minutes
    '''
    # Initialization time
    t0 = time.fromisoformat(init_time) #datetime.timedelta(hours=int(model_init_hhhh[:2]))
    # Forecast time
    t1 = time.fromisoformat(forecast_time) #('20111104') #datetime.timedelta(hours=int(hhmmss[:2]), minutes=int(hhmmss[2:4]))
    forecast_window = (t1 - t0).total_seconds() / 60
    if DB: print(f"Forecast window: {forecast_window} min")    
    return forecast_window

def extract_fields(args, wrfin, gridrad_heights, fields=None):
    ''' 
    Extract reflectivity, vorticity, divergence and any additionally specified WoFS fields

    @param args: parsed command line args. see create_argsparser()
    @param wrfin: netCDF Dataset, not xarray, is required to use wrf-python functions
    @param gridrad_heights: list or numpy array of the elevations in the GridRad data in km
    @param fields: TODO list of strings indicating additional fields to extract

    @return: 4-tuple of numpy arrays containing the reflectivity, divergence, vorticity,
            and updraft helicity. ( Z_agl, div, vort, uh)
    '''
    # Get wofs heights
    height = wrf.getvar(wrfin, "height_agl", units='m')

    # Get reflectivity, and U and V winds
    Z = wrf.getvar(wrfin, "REFL_10CM")
    #if not Z_only:
    U = wrf.g_wind.get_u_destag(wrfin)
    V = wrf.g_wind.get_v_destag(wrfin)   
    
    # Interpolate wofs data to gridrad heights
    gridrad_heights = gridrad_heights * 1000
    Z_agl = wrf.interplevel(Z, height, gridrad_heights)
    #if not Z_only:
    U_agl = wrf.interplevel(U, height, gridrad_heights)
    V_agl = wrf.interplevel(V, height, gridrad_heights)
    
    # Add units to winds to use metpy functions
    U = U_agl.values * (metpy.units.units.meter / metpy.units.units.second)
    V = V_agl.values * (metpy.units.units.meter / metpy.units.units.second)
    
    # Define grid spacings (for div and vort)
    dx = 3000 * (metpy.units.units.meter)
    dy = 3000 * (metpy.units.units.meter)

    # Calculate divergence and vorticity
    div = metpy.calc.divergence(U, V, dx=dx, dy=dy)
    vort = metpy.calc.vorticity(U, V, dx=dx, dy=dy)
    
    # Grab data, remove metpy.units and xarray.Dataset stuff 
    div = np.asarray(div)
    vort = np.asarray(vort)
    Z_agl = Z_agl.values 
    uh = wrf.getvar(wrfin, 'UP_HELI_MAX')
    uh = uh.values

    return Z_agl, div, vort, uh

def make_patches(args, radar, window):
    '''
    Create a Dataset of concatenated along the patches

    @param args: parsed command line args
            Relevant args: see create_argsparser for more details
                patch_shape: shape of the patches
                Z_only: only extract ZH
    @param radar: WoFS data
    @param window: duration time in minutes of the forecast window (i.e., the 
                    time between the initialization of the simulation and the 
                    time of the current forecast)

    @return: xarray Dataset of patches
    '''
    # List of patches
    patches = []

    Z_only = args.Z_only
    
    lat_len = radar.Latitude.shape[0]
    lon_len = radar.Longitude.shape[0]

    xsize = None
    ysize = None
    csize = None
    ndims = len(args.patch_shape)
    if ndims >= 2:
        xsize = args.patch_shape[0]
    if ndims == 2:
        ysize = xsize
    elif ndims >= 3:
        ysize = args.patch_shape[1]
    if ndims == 4:
        # Number of channels
        csize = args.patch_shape[2]
    #else: raise ValueError(f"[ARGUMENTS] patch_shape number of dimensions should be 0, 2, 3, or 5 but was {ndims}")
    
    lat_range = range(0, lat_len, xsize - 4)
    lon_range = range(0, lon_len, ysize - 4)
    
    naltitudes = radar.Altitude.values.shape[0]

    # Iterate over Latitude and Longitude for every size-th pixel. Performed every (xi, yi) = multiples of size, and will give us a normal array
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
                                coords=dict(x=(["x"], np.arange(xsize)), y=(["y"], np.arange(ysize)), 
                                            z=(["z"], np.arange(1, naltitudes + 1))))

            to_add = to_add.fillna(0)

            patches.append(to_add)

    # Combine all patches into one dataset
    ds_patches = xr.concat(patches, 'patch')
    
    return ds_patches

def to_gridrad(args, wofs, wofs_netcdf, gridrad_spacing=48, gridrad_heights=np.arange(1, 13, step=1, dtype=int)):
    '''
    Convert WoFS data into the GridRad grid

    @param args: command line args. see create_argsparser()
    @param wofs: xarray Dataset
    @param wofs_netcdf: netcdf Dataset
    @param gridrad_spacing: Gridrad files have grid spacings of 1/48th degrees lat/lon
            1 / (gridrad_spacing) degrees
    @param gridrad_heights: 1D array of the elevation levels in the GridRad data. Height values we want to interpolate to

    @return: xarray Dataset with the interpolated and patched data
    '''
    DB = args.dry_run

    # There is a difference sometimes between the day in the filepath and the day in the filename
    # All filepath dates have '_path' and all filename dates have '_day'
    
    # lat/lon points for regridded grid points
    new_gridrad_lats_lons, new_gridrad_lats, new_gridrad_lons = calculate_output_lats_lons(wofs, 
                                                                    gridrad_spacing=gridrad_spacing)

    # Pull out the desired data from the wofs file
    Z_agl, div, vort, uh = extract_fields(args, wofs_netcdf, gridrad_heights)

    # Remove axes of length 1
    xlat_vals = np.squeeze(wofs.XLAT.values)
    xlon_vals = np.squeeze(wofs.XLONG.values)
    # List of all lat/lon grid points from original wofs grid
    wofs_lats_lons = np.stack((np.ravel(xlat_vals), np.ravel(xlon_vals))).T
    
    # KD Tree with wofs grid points
    tree = spatial.cKDTree(wofs_lats_lons)
    
    # For each gridrad point, find k(=4) closest wofs grid points
    distances, points = tree.query(new_gridrad_lats_lons, k=4)
    
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

    Z_only = args.Z_only

    # Selecting data 
    refls = Z_agl[bottom_top_3d, south_north_points_3d, west_east_points_3d]
    uhs = uh[south_north_points, west_east_points]
    if not Z_only:
        vorts = vort[bottom_top_3d, south_north_points_3d, west_east_points_3d]
        divs = div[bottom_top_3d, south_north_points_3d, west_east_points_3d]

    # Perform IDW interpolation and reshape data to shape to (Time, Altitude, Latitude, Longitude)
    dist3d_sq = distances_3d**2
    dist_sq = distances**2
    lats_len = new_gridrad_lats.size
    lons_len = new_gridrad_lons.size
    REFL_10CM_final = np.swapaxes(np.swapaxes((np.sum(refls / dist3d_sq, axis=2) / np.sum(1 / dist3d_sq, axis=2)).reshape(1, lats_len, lons_len, vort.shape[0]), 1, 3), 2, 3).astype(np.float32)
    uh_final = (np.sum(uhs / dist_sq, axis=1)/np.sum(1 / dist_sq, axis=1)).reshape(1, lats_len, lons_len).astype(np.float32)
    if not Z_only:
        vort_final = np.swapaxes(np.swapaxes((np.sum(vorts / dist3d_sq, axis=2) / np.sum(1 / dist3d_sq, axis=2)).reshape(1, lats_len, lons_len, vort.shape[0]), 1, 3), 2, 3).astype(np.float32)
        div_final = np.swapaxes(np.swapaxes((np.sum(divs / dist3d_sq, axis=2) / np.sum(1 / dist3d_sq, axis=2)).reshape(1, lats_len, lons_len,vort.shape[0]), 1, 3), 2, 3).astype(np.float32)
    
    # Create corresponding data time object and
    #dtime = create_wofs_time(year, month, day, hhmmss, DB=is_dry_run)
    dtime = wofs['Time'].values[0]
    # Put data into xarray DataArrays with same dimensions, coordinates, and variable fields as gridrad
    #TODO: include init and forecast datetime strings
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
    wofs_regridded = xr.Dataset(coords={"time": [dtime], "Longitude": new_gridrad_lons + 360, 
                                        "Latitude": new_gridrad_lats, "Altitude": gridrad_heights})
    wofs_regridded["ZH"] = wofs_regridded_refc
    wofs_regridded["UH"] = wofs_regridded_uh
    if not Z_only:
        wofs_regridded["VOR"] = wofs_regridded_vort
        wofs_regridded["DIV"] = wofs_regridded_div
    
    # Where reflectivity=0, set to nans
    if args.with_nans:
        wofs_regridded = wofs_regridded.where(wofs_regridded.ZH > 0)

    # Calculate forecast window
    datetime_forecast_str = wofs.Times.data[0].decode('ascii') # decode from binary string toregular text string
    datetime_forecast = datetime.fromisoformat(datetime_forecast_str)
    datetime_init = datetime.fromisoformat(wofs.START_DATE) #datetime.fromisoformat(args.datetime_init) #.strftime(datetime_fmt) #datetime.strptime(args.datetime_init, datetime_fmt)
    #forecast_window = wofs_forecast_window(year, month, day, hhmmss, DB=is_dry_run)
    forecast_window = (datetime_forecast - datetime_init).seconds / 60
    if DB: print(f"Forecast window: {forecast_window} min")

    # TODO: make  patches
    ds_patches = make_patches(args, wofs_regridded, forecast_window)
    ds_patches.attrs = wofs.attrs
    # Release resources
    wofs_regridded.close()

    return ds_patches

"""
Tornado prediction methods.
"""
def normalize(args, wofs, DB=0):
    ''' TODO
    Open args.file_training_metadata with the mean and std of the training data to 
    normalize the data

    @param args: command line args. see create_argsparser()

    @return: the normalized feature data
    '''
    fpath = args.file_training_metadata

    if DB: print("Training meta data used:", fpath)
    train_stats = xr.open_dataset(fpath)
    mu_ZH = float(train_stats.ZH_mean.values)
    std_ZH = float(train_stats.ZH_std.values)
    train_stats.close()

    zh_norm = (wofs.ZH - mu_ZH) / std_ZH

    return zh_norm

def load_evaluate(args, x, DB=0, **fss_args):
    '''TODO
    Load and evaluate the model

    @param args: command line args. see create_argsparser()
    @param x:
    @param DB:
    @param fss_args: keyword args for make_fractions_skill_score()

    @return: the predictions
    '''
    model_path = args.fd_model

    if DB: print("Loading model:", model_path)
    #fss = make_fractions_skill_score(2, 2, c=1.0, cutoff=0.5, want_hard_discretization=False)
    fss = make_fractions_skill_score(**fss_args)
    model = keras.models.load_model(model_path, custom_objects={'fractions_skill_score': fss, 
                                                'MaxCriticalSuccessIndex': MaxCriticalSuccessIndex})

    # Predict with the data
    print(f"Performing predictions (data shape={x.shape}) ...")
    preds = model.predict(x)

    return preds

def save(args, wofs, preds, DB=0):
    # TODO
    # Dataset with true and predicted patch data
    # Extract the wofs UH data and metadata

    zh = normalize(args, wofs, DB=DB)
    uh = wofs.UH.values
    n_convective_pixels = wofs.n_convective_pixels.values
    n_uh_pixels = wofs.n_uh_pixels.values
    lat = wofs.lat.values
    lon = wofs.lon.values
    dtime = wofs.time.values
    forecast_window = wofs.forecast_window.values

    ds_wofs_as_gridrad = xr.Dataset(data_vars=dict(UH = (["patch", "x", "y"], uh),
                                        ZH_composite = (["patch", "x", "y"], zh.values.max(axis=3)),
                                        ZH = (["patch", "x", "y", "z"], zh.values),
                                        stitched_x = (["patch", "x"], wofs.stitched_x.values),
                                        stitched_y = (["patch", "y"], wofs.stitched_y.values),
                                        predicted_no_tor = (["patch", "x", "y"], preds[:,:,:,0]),
                                        predicted_tor = (["patch", "x", "y"], preds[:,:,:,1]),
                                        n_convective_pixels = (["patch"], n_convective_pixels),
                                        n_uh_pixels = (["patch"], n_uh_pixels),
                                        lat = (["patch"], lat),
                                        lon = (["patch"], lon),
                                        time = (["patch"], dtime),
                                        forecast_window = (["patch"], forecast_window)),
                                    coords=dict(patch = range(preds.shape[0]),
                                                x = range(32),
                                                y = range(32),
                                                z = range(1,13)))

    return ds_wofs_as_gridrad

def separate_times(args, wofs):
    '''TODO
    @param args:
    @param wofs: 

    @return:
    '''
    #attrs={'START_DATE'}
    #wofs.Times.data[0]
    dtimes = set(wofs.time.values)
    ds_times = []


if __name__ == '__main__':
    args = parse_args()

    wofs_files = []
    if os.path.isfile(args.loc_wofs):
        wofs_files.append(args.loc_wofs)
    elif os.path.isdir(args.loc_wofs):
        wofs_files = os.listdir(args.loc_wofs)
    else: raise ValueError(f"[ARGUMENT ERROR] --loc_wofs should either be a file or directory was {args.loc_wofs}")
    '''wofs_files = [args.file_wofs]
    if args.dir_wofs != '':
        # Glob all available data files
        wofs_files = glob.glob(args.dir_wofs)'''
    
    # Opens all data files into a Dataset
    print("Open WoFS file(s)", wofs_files)
    #'%Y-%m-%d_%H:%M:%S'
    #--datetime_format="^%Y-^%m-^%d_^%H^:^%M^:^%S"  
    #--datetime_init="2019-05-17_03^:00^:00" 
    #--datetime_forecast="2019-05-18_05^:25^:00"
    datetime_fmt = '%Y-%m-%d_%H:%M:%S' #args.datetime_format
    print(args.datetime_forecast, datetime_fmt)

    # TODO
    #args.datetime_init = "2019-05-17_03:00:00" 
    args.datetime_forecast = "2019-05-18_05:25:00"

    datetime_forecast = args.datetime_forecast #datetime.fromisoformat(args.datetime_forecast)
    
    DB = args.dry_run
    wofs, wofs_netcdf = load_wofs_file(wofs_files, args.filename_prefix, 
                                       wofs_datetime=datetime_forecast,
                                       datetime_format=datetime_fmt,
                                       seconds_since='seconds since 2001-01-01', 
                                       engine='netcdf4', DB=DB)
                                       
    # TODO: Interpolate WoFS grid to GridRad grid
    GRIDRAD_HEIGHTS = np.arange(1, 13, step=1, dtype=int)
    GRIDRAD_SPACING = 48
    wofs = to_gridrad(args, wofs, wofs_netcdf, gridrad_spacing=GRIDRAD_SPACING, gridrad_heights=GRIDRAD_HEIGHTS)

    # TODO: Compute predictions
    #preds = 

    # TODO: Interpolate back to WoFS grid
    #preds = 

    # TODO: Save the predictions
    #save(args, wofs, preds, DB=DB)

    # Close the netcdf dataset
    wofs_netcdf.close()

    #output preds, reflectivity, U, V, command line options for addtional fields
