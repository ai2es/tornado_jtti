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
                                                     --with_nans 
                                                     (--fields)
                                                     --loc_model
                                                     --file_trainset_stats
                                                     --write=0
                                                     --debug_on
"""
import re, os, sys, glob, argparse, logging
from datetime import datetime, date, time
from dateutil.parser import parse as parse_date
import xarray as xr
import numpy as np
from netCDF4 import Dataset, date2num, num2date
from scipy import spatial
import wrf
import metpy
import metpy.calc

import json, time, argparse, subprocess
from azure.storage.queue import QueueClient, TextBase64EncodePolicy, TextBase64DecodePolicy

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow import keras

# Working directory expected to be tornado_jtti/
sys.path.append("lydia_scripts")
from custom_losses import make_fractions_skill_score
from custom_metrics import MaxCriticalSuccessIndex
from scripts_data_pipeline.wofs_to_gridrad_idw_azure import calculate_output_lats_lons
import gc
gc.collect()


"""
WoFS to GridRad interpolation methods, refactored from lydia_scripts/scripts_data_pipeline
"""
def load_wofs_file(filepath, filename_prefix=None, wofs_datetime=None, 
                   datetime_format='%Y-%m-%d_%H:%M:%S', 
                   seconds_since='seconds since 2001-01-01', 
                   engine='netcdf4', debug=0, **kwargs):
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
    @param debug: int debug flag to print out additional debug information
    @param kwargs: any additional desired parameters to xarray.open_dataset(...)
    
    @return: tuple with xarray Dataset and netCDF4 Dataset
    '''
    if debug: print(f"Loading {filepath}")

    # Create Dataset
    wofs = None
    wofs = xr.open_dataset(filepath, engine=engine, decode_times=False, **kwargs).load()
    wofs.close()
    wofs.attrs['filenamepath'] = filepath

    if wofs_datetime is None:
        wofs_datetime = wofs.Times.data[0].decode('ascii') # decode from binary string to text string
    _datetime, dt_int, dt_str = get_wofs_datetime(filepath, filename_prefix=filename_prefix, 
                                                wofs_datetime=wofs_datetime, seconds_since=seconds_since, 
                                                datetime_format=datetime_format, debug=debug)
    wofs = wofs.reindex(Time=[dt_int])
    wofs.attrs['FORECAST_DATETIME'] = dt_str
    
    # Create netcdf4 Dataset to use wrf-python 
    wofs_netcdf = Dataset(filepath)
    return wofs, wofs_netcdf

def load_wofs_files(filepaths, filename_prefix, datetime_format, engine='netcdf4',
                   seconds_since='seconds since 2001-01-01', debug=0, **kwargs):
    ''' TODO: maybe. stretch
    (not implemented) Load multiple WoFS files
    '''
    wofs = xr.open_mfdataset(filepaths, concat_dim='Times', combine='nested', 
                             parallel=True, engine=engine, **kwargs) #concat_dim='patch'

    for f, filepath in enumerate(filepaths):
        wofs_datetime = wofs.Times.data[f]
        _datetime, dt_int, dt_str = get_wofs_datetime(filepath, filename_prefix=filename_prefix, 
                                                wofs_datetime=wofs_datetime, seconds_since=seconds_since, 
                                                datetime_format=datetime_format, debug=debug)
        wofs = wofs.reindex(Time=[dt_int])
        wofs_netcdf = Dataset(filepath)

    #return wofs
    pass

def create_wofs_time(year, month, day, hour, min, sec, 
                     seconds_since='seconds since 2001-01-01', debug=0):
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
    @param debug: int debug flag to print out additional debug information

    @return: the WoFS datetime as a np.datetime int
    '''
    dtime = datetime.datetime(year, month, day, hour, min, sec)
    dtime = date2num(dtime, seconds_since)
    if debug: print(f"WoFS time: {dtime}")
    return dtime 

def get_wofs_datetime(fnpath, filename_prefix=None, wofs_datetime=None, 
                      datetime_format='%Y-%m-%d_%H:%M:%S', 
                      seconds_since='seconds since 2001-01-01', debug=0):
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
    @param debug: int debug flag to print out additional debug information

    @return: a 3-tuple with the datetime object, the datetime int 
            since seconds_since and the formatted datetime string for the WoFS 
            data file
    '''
    fname = os.path.basename(fnpath) 
    fname, file_extension = os.path.splitext(fname)

    if not filename_prefix is None:
        fname = fname.replace(filename_prefix, '')
        wofs_datetime, _ = parse_date(fname, fuzzy_with_tokens=True)
        if debug: print(f"Extracted {wofs_datetime} from file {fnpath}:{fname}")

    datetime_obj = wofs_datetime
    if not isinstance(datetime_obj, datetime):
        # Convert datetime string to datetime object
        datetime_obj = datetime.strptime(wofs_datetime, datetime_format)
        if debug: print(f"\nExtracted datetime object:: {wofs_datetime}={datetime_obj} from file {fnpath}({fname})")

    # Convert datetime object to np.datetime int
    datetime_int = date2num(datetime_obj, seconds_since)
    if debug: print(f"Extracted datetime int in {seconds_since}:: {datetime_int} from file {fnpath}({fname})")

    # Convert datetime object to string with the desired format
    datetime_str = datetime_obj.strftime(datetime_format)
    if debug: print(f"Extracted datetime string:: {datetime_str} from file {fnpath}({fname})\n")

    return datetime_obj, datetime_int, datetime_str

def compute_forecast_window(init_time, forecast_time, debug=0):
    ''' 
    Calculate the forecast window as the time between the WoFS initialization 
    time and the WoFS forecast time.

    @param init_time: string indicating the initialization time of the WoFS 
            simulation.
    @param forecast_time: string indicating the forecast time of the WoFS data
    @param debug: int debug flag to print out additional debug information

    @return: integer duration of the forecast window in minutes
    '''
    # Initialization time
    t0 = time.fromisoformat(init_time) 
    # Forecast time
    t1 = time.fromisoformat(forecast_time) 

    forecast_window = (t1 - t0).total_seconds() / 60
    if debug: print(f"Forecast window: {forecast_window} min\n")  

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

    # Get reflectivity, and U and V winds
    Z = wrf.getvar(wrfin, "REFL_10CM")
    #if not ZH_only:
    U = wrf.g_wind.get_u_destag(wrfin)
    V = wrf.g_wind.get_v_destag(wrfin)   
    
    # Interpolate wofs data to gridrad heights
    gridrad_heights = gridrad_heights * 1000
    Z_agl = wrf.interplevel(Z, height, gridrad_heights)
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

def to_gridrad(args, rel_path, wofs, wofs_netcdf, gridrad_spacing=48,
               gridrad_heights=np.arange(1, 13, step=1, dtype=int), debug=0):
    '''
    Convert WoFS data into the GridRad grid. If with_nans is set, change grid points 
    with reflectivity=0 to NaN

    @param args: command line args. see create_argsparser()
            Relevant args:
                dir_patches: directory to save patched WoFS data in the GridRad grid
                write: flag whether to the patched data
                with_nans: flag whether to set reflectivity values that are 0 to NaN
                debug_on: flag whether to run in debug mode (null) or not (not null)
    @param wofs: xarray Dataset
    @param wofs_netcdf: netcdf Dataset
    @param gridrad_spacing: Gridrad files have grid spacings of 1/48th degrees lat/lon
            1 / (gridrad_spacing) degrees
    @param gridrad_heights: 1D array of the elevation levels in the GridRad data. 
            Height values we want to interpolate to
    @param debug: int debug flag to print out additional debug information

    @return: xarray Dataset with the interpolated and patched data
    '''
    # There is a difference sometimes between the day in the filepath and the day in the filename
    # All filepath dates have '_path' and all filename dates have '_day'
    
    # lat/lon points for regridded grid points
    new_gridrad_lats_lons, new_gridrad_lats, new_gridrad_lons = calculate_output_lats_lons(wofs, 
                                                                    gridrad_spacing=gridrad_spacing)

    # Pull out the desired data from the wofs file
    Z_agl, div, vort, uh = extract_netcdf_dataset_fields(args, wofs_netcdf, gridrad_heights)

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

    Z_only = args.ZH_only

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

    REFL_10CM_final = np.swapaxes(np.swapaxes((np.sum(refls / dist3d_sq, axis=2) / np.sum(1 / dist3d_sq, axis=2)).reshape(1, lats_len, lons_len, Z_agl.shape[0]), 1, 3), 2, 3).astype(np.float32)
    uh_final = (np.sum(uhs / dist_sq, axis=1) / np.sum(1 / dist_sq, axis=1)).reshape(1, lats_len, lons_len).astype(np.float32)

    if not Z_only:
        vort_final = np.swapaxes(np.swapaxes((np.sum(vorts / dist3d_sq, axis=2) / np.sum(1 / dist3d_sq, axis=2)).reshape(1, lats_len, lons_len, vort.shape[0]), 1, 3), 2, 3).astype(np.float32)
        div_final = np.swapaxes(np.swapaxes((np.sum(divs / dist3d_sq, axis=2) / np.sum(1 / dist3d_sq, axis=2)).reshape(1, lats_len, lons_len, vort.shape[0]), 1, 3), 2, 3).astype(np.float32)
    
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
                                        "Longitude": new_gridrad_lons + 360, 
                                        "Latitude": new_gridrad_lats, 
                                        "Altitude": gridrad_heights})
    wofs_regridded["ZH"] = wofs_regridded_refc
    wofs_regridded["UH"] = wofs_regridded_uh
    if not Z_only:
        wofs_regridded["VOR"] = wofs_regridded_vort
        wofs_regridded["DIV"] = wofs_regridded_div
    
    # Where reflectivity=0, set to nans
    if args.with_nans:
        wofs_regridded = wofs_regridded.where(wofs_regridded.ZH > 0)

    # Calculate forecast window
    datetime_forecast_str = wofs.Times.data[0].decode('ascii') # decode from binary string to regular text string
    datetime_forecast = datetime.fromisoformat(datetime_forecast_str)
    datetime_init = datetime.fromisoformat(wofs.START_DATE)
    forecast_window = (datetime_forecast - datetime_init).seconds / 60
    if debug: print(f"Forecast window: {forecast_window} min\n")

    # Make patches
    ds_patches = make_patches(args, wofs_regridded, forecast_window)
    #ds_patches['Times'] = wofs.Times
    ds_patches.attrs = wofs.attrs
    
    if args.write in [2, 4] and not os.path.join(args.dir_patches) is None:
        fname = os.path.basename(wofs.filenamepath)
        fname, _ext = os.path.splitext(fname)
        patch_shape = [f'{c:03d}' for c in args.patch_shape]
        patch_shape_str = '_'.join(patch_shape)
        os.makedirs(os.path.join(args.vm_datadrive, args.dir_patches, rel_path), mode=0o775, exist_ok=True)
        savepath = os.path.join(args.vm_datadrive, args.dir_patches, rel_path, f'{fname}_patched_{patch_shape_str}.nc')
        if args.debug_on: print(f"Saving patched WoFS data interpolated to GridRad grid to {savepath}\n")
        #ds_patches.to_netcdf(savepath)
        
        #blobpath = os.path.join(args.blob_path_ncar, args.dir_patches, rel_path, f'{fname}_patched_{patch_shape_str}.nc')
        #subprocess.run(["azcopy",
        #                "copy",
        #                "--log-level=ERROR",
        #                f"{savepath}",
        #                f"{blobpath}"])         

    return ds_patches


"""
Tornado prediction methods.
"""
def load_trainset_stats(args, engine='netcdf4', debug=0, **kwargs):
    ''' 
    Open args.file_trainset_stats with the mean and std of the training data. These 
    will be used for normalizing the data prior to prediction

    @param args: command line args. see create_argsparser()
            Relevant args:
                file_trainset_stats: file path to the mean and STD of the fields
                fields: list of fields
    @param engine: Engine to use when reading file. see xarray Dataset 
            documentation 
    @param debug: int debug flag to print out additional debug information
    @param kwargs: keyword args for xr.open_dataset()

    @return: the dataset with the means and stds of the features
    '''
    fpath = args.file_trainset_stats

    if debug: print("Training meta data used:", fpath)

    train_stats = xr.open_dataset(fpath, engine=engine, **kwargs).load() #, cache=False
    train_stats.close()

    return train_stats

def predict(args, wofs, stats, eval=False, debug=0, **fss_args):
    '''
    Load the model and perform the predictions.
    TODO: predict on arbitrary set of fields

    @param args: command line args. see create_argsparser()
            Relevant attributes:
                loc_model: location to the the trained model to use
    @param wofs: data to predict on
    @param stats: data field statistics such as mean and standard deviation of 
            the WoFS fields from the training set data
    @param eval: TODO whether to compute evaluation results. only available if
            the true labels are also available
    @param debug: int debug flag to print out additional debug information
    @param fss_args: keyword args for make_fractions_skill_score()
            default: {'mask_size': 2, 'num_dimensions': 2, 'c':1.0, 
                        'cutoff': 0.5, 'want_hard_discretization': False}

    @return: the predictions as a numpy array
    '''

    model_path = args.loc_model

    if debug: print("Loading model:", model_path)

    if fss_args is None or fss_args == {}:
        fss_args = {'mask_size': 2, 'num_dimensions': 2, 'c':1.0, 
                    'cutoff': 0.5, 'want_hard_discretization': False}
    fss = make_fractions_skill_score(**fss_args)
    model = keras.models.load_model(model_path, custom_objects={'fractions_skill_score': fss, 
                                                'MaxCriticalSuccessIndex': MaxCriticalSuccessIndex})

    # Normalize the reflectivity data
    ZH_mu = float(stats.ZH_mean.values)
    ZH_std = float(stats.ZH_std.values)
    X = (wofs.ZH - ZH_mu) / ZH_std

    preds = model.predict(X)

    return preds

def combine_fields(args, wofs, preds, gridrad_heights=range(1, 13), debug=0):
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
    @param debug: int debug flag to print out additional debug information

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

    wofs_preds['ZH_composite'] = xr.DataArray(
        data=wofs.ZH.values.max(axis=3), #["patch", "x", "y"]
        #dims=dims,
        coords=coords
    )
    wofs_preds['predicted_no_tor'] = xr.DataArray(
        data=preds[:, :, :, 0], #["patch", "x", "y"]
        #dims=wofs.dims,
        coords=coords
    ) 
    wofs_preds['predicted_tor'] = xr.DataArray(
        data=preds[:, :, :, 1], #["patch", "x", "y"]
        #dims=wofs.dims,
        coords=coords
    )

    return wofs_preds

def to_wofsgrid(args, rel_path, wofs_orig, wofs_gridrad, stats, gridrad_spacing=48, 
                 seconds_since='seconds since 2001-01-01', debug=0):
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
    @param seconds_since: string seconds since date statement to use for creating
            NETCDF4 datetime integers. string of the form since describing the 
            time units. can be days, hours, minutes, etc. see netcdf4.date2num() 
            documentation for more information
    @param debug: int debug flag to print out additional debug information

    @return: a tuple with stitched WoFS prediction patches (in gridrad system), 
            and the WoFS data as an xarray Dataset in the original WoFS grid 
            system
    '''
    # Stitch back together the patches of the predictions
    predictions =  stitch_patches(args, wofs_gridrad, stats, 
                                    gridrad_spacing=gridrad_spacing, 
                                    seconds_since=seconds_since, debug=debug)
    
    # Add the stitched file to the list of stitched files     
    #stitched_preds.append(predictions)

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
    padded_lons =  np.linspace(predictions.lon.min().values - to_add_east_west / gridrad_spacing, 
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
    predictions_idw = (np.sum(predictions_not_idw / dist_sq, axis=1) / np.sum(1 / dist_sq, axis=1)).reshape(wofs_orig.XLAT.values.shape[1:])

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
        fields += args.fields
        fields = set(fields)
    wofs_fields = wofs_orig[fields].copy(deep=True)
    wofs_like = xr.merge([wofs_like, wofs_fields])

    # Save out the interpolated file
    if args.write in [1, 2, 4]:
        fname = os.path.basename(wofs_orig.filenamepath)
        fname, file_extension = os.path.splitext(fname)
        savepath = os.path.join(args.vm_datadrive, args.dir_preds, rel_path)
        os.makedirs(savepath, mode=0o775, exist_ok=True)
        if args.debug_on: print(f"Save WoFS grid predictions to {savepath}\n")
        wofs_like.to_netcdf(os.path.join(savepath, f"{str(fname)}_predictions.nc"))
        
        #blobpath = os.path.join(args.blob_path_ncar, args.dir_preds, rel_path, f'{fname}_predictions.nc')
        #subprocess.run(["azcopy",
        #                "copy",
        #                "--log-level=ERROR",
        #                f"{savepath}",
        #                f"{blobpath}"])   
        
    return predictions, wofs_like, vm_filepath

def stitch_patches(args, wofs, stats, gridrad_spacing=48, 
                   seconds_since='seconds since 2001-01-01', debug=0):
    """ 
    Reconstruct the stitched grid for a single WoFS forecast time

    @param args: command line args. see create_argsparser()
            Relevant arguments
                patch_shape: shape of the patches
    @param wofs: WoFS data from a single time point
    @param stats: xarray dataset with training mean and std of the reflectivity 
            and other fields
    @param gridrad_spacing: Gridrad files have grid spacings of 1/48th degrees lat/lon
            1 / (gridrad_spacing) degrees
    @param seconds_since: string seconds since date statement to use for creating
            NETCDF4 datetime integers. string of the form since describing the 
            time units. can be days, hours, minutes, etc. see netcdf4.date2num() 
            documentation for more information
    @param debug: int debug flag to print out additional debug information

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
    
    # Compute average of each patch by dividing by the total number of patches that contained each pixel
    tor_preds = tor_preds / overlap_array
    uh_array = uh_array / overlap_array
    zh_low_level = zh_low_level / overlap_array

    # Obtain the time of the forecast
    datetime_int = num2date(wofs.time.values[0], seconds_since) #
    forecast_time = np.datetime64(datetime_int)

    # Put stitched grid into Dataset 
    wofs_stiched = xr.Dataset(data_vars=dict(UH=(["time", "lat", "lon"],
                                                    uh_array.reshape(1, total_in_lat, total_in_lon)),
                                            ZH_1km=(["time", "lat", "lon"], 
                                                    zh_low_level.reshape(1, total_in_lat, total_in_lon)),
                                            predicted_tor=(["time", "lat", "lon"], 
                                                    tor_preds.reshape(1, total_in_lat, total_in_lon))),
                            coords=dict(time=[forecast_time], lon=lons, lat=lats),
                            attrs=wofs.attrs
                            )
    wofs_stiched.ZH_1km.attrs['units'] = 'dBZ'
    wofs_stiched.UH.attrs['units'] = 'm^2/s^2'

    return wofs_stiched


def wofs_to_preds(ncar_filepath, args):
    
    if args.debug_on: print("NO OF FIELDS: ", len(args.fields))
    if args.debug_on: print("FIELDS: ", args.fields)
    if args.debug_on: print(f'\tProcessing {ncar_filepath}')
    
    SECS_SINCE = 'seconds since 2001-01-01'
    GRIDRAD_SPACING = 48
    GRIDRAD_HEIGHTS = np.arange(1, 13, step=1, dtype=int)
    train_stats = load_trainset_stats(args, debug=args.debug_on) 

    rel_path = ncar_filepath.rsplit('/', 1)[0].split('wrf-wofs/')[1] + '/'
    filename = ncar_filepath.rsplit('/', 1)[1]
    path = os.path.join(args.vm_datadrive, args.dir_wofs, rel_path)
    os.makedirs(path, mode=0o775, exist_ok=True)

    subprocess.run(["azcopy",
                    "copy",
                    f"{ncar_filepath}",
                    f"{path}/{filename}"]) 
            
    wofs, wofs_netcdf = load_wofs_file(f"{path}/{filename}",
                                       wofs_datetime=None,
                                       datetime_format=args.datetime_format,
                                       seconds_since=SECS_SINCE,
                                       engine='netcdf4',
                                       debug=args.debug_on)

    # Interpolate WoFS grid to GridRad grid
    wofs_gridrad = to_gridrad(args, rel_path, wofs, wofs_netcdf,
                              gridrad_spacing=GRIDRAD_SPACING,
                              gridrad_heights=GRIDRAD_HEIGHTS,
                              debug=args.debug_on)
    
    # Compute predictions
    preds = predict(args, wofs_gridrad, train_stats, debug=args.debug_on) #, **fss_args)

    # Combine the predictions, reflectivity and select fields into a single dataset
    wofs_combo = combine_fields(args, wofs_gridrad, preds, debug=args.debug_on)

    # Interpolate back to WoFS grid
    preds_gridrad_stitched, preds_wofsgrid, vm_filepath = to_wofsgrid(args, rel_path, wofs, wofs_combo,
                                                                      train_stats, gridrad_spacing=GRIDRAD_SPACING,
                                                                      seconds_since=SECS_SINCE, debug=args.debug_on)
            
    os.remove(f"{path}{filename}")
    