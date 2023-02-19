"""
Made by Lydia 2022
Modified by Monique Shotande Feb 2023

Evaluate a tensorflow model on WoFS (Warn on Forecast System) data
that has already been interpolated to the GridRad grid. The output
of the model is predictions for each pixel, interpolated back to the 
original WoFS grid.

Execution Instructions:
    Conda environment requires are in the environment.yml
    file. Execute:
        conda env create --name tornado --file environment.yml
    to create the conda environment necessary to run this script.
    Custom module requirements:
        The custom modules: custom_losses, custom_metrics, and process_monitoring
        are in the tornado_jtti project directory.
    Information about the required command line arguments are described in the 
    method get_arguments(). Run
        python evaluate_wofs_ZH_only.py --h
"""

from datetime import datetime
from tensorflow import keras
print("keras version", keras.__version__)
import xarray as xr
print("xr version", xr.__version__)
import numpy as np
print("np version", np.__version__)
import csv
import tensorflow as tf
print("tensorflow version", tf.__version__)
import glob
import argparse
import os
import sys
import scipy
print("scipy version", scipy.__version__)
import netCDF4
print("netCDF4 version", netCDF4.__version__)
from scipy import spatial
sys.path.append("/home/lydiaks2/tornado_project/")
from custom_losses import make_fractions_skill_score
from custom_metrics import MaxCriticalSuccessIndex
#sys.path.append("/home/momoshog/Tornado/tornado_jtti/process_monitoring")
sys.path.append("process_monitoring")
from process_monitor import ProcessMonitor



def get_arguments():

    #Define the strings that explain what each input variable means
    INIT_TIME_HELP_STRING = 'i, where i indicates the ith file of the storm masks from which to make the patches.'
    PATCHES_DIR_HELP_STRING = 'The directory where the unet mask files are stored. This directory should start from the root directory \'/\'.'
    ML_MODEL_DIR_HELP_STRING = 'The directory where the trained model is stored. This directory should start from the root directory \'/\'.'
    OUTFILE_DIR_HELP_STRING = 'The directory where the model predictions from wofs data will be stored. This directory should start from the root directory \'/\'.'
    METADATA_PATH_HELP_STRING = 'The path to the training metadata file used to train the ML model we are evaluating.'
    WOFS_DIR_HELP_STRING = 'Path to wofs is the location of the raw wofs files on schooner. This directory should start from the root directory \'/\'.'
    PATCH_SIZE_HELP_STRING = 'Patch size is the size of each patch in each horizontal dimension. Patch size of 32 will produce (32,32,12)-sized patches.'

    #Tell python what arguments to expect from the .sh file
    INPUT_ARG_PARSER = argparse.ArgumentParser()
        
    INPUT_ARG_PARSER.add_argument(
        '--wofs_day_idx', type=str, required=True,
        help=INIT_TIME_HELP_STRING)
        
    INPUT_ARG_PARSER.add_argument(
        '--input_patch_dir_name', type=str, required=True,
        help=PATCHES_DIR_HELP_STRING)
        
    INPUT_ARG_PARSER.add_argument(
        '--output_model_checkpoint_path', type=str, required=True,
        help=ML_MODEL_DIR_HELP_STRING)
        
    INPUT_ARG_PARSER.add_argument(
        '--output_predictions_checkpoint_path', type=str, required=True,
        help=OUTFILE_DIR_HELP_STRING)
        
    INPUT_ARG_PARSER.add_argument(
        '--training_data_metadata_path', type=str, required=True,
        help=METADATA_PATH_HELP_STRING)
        
    INPUT_ARG_PARSER.add_argument(
        '--path_to_wofs', type=str, required=True,
        help=WOFS_DIR_HELP_STRING)
        
    INPUT_ARG_PARSER.add_argument(
        '--patch_size', type=int, required=True,
        help=PATCH_SIZE_HELP_STRING)
        
    INPUT_ARG_PARSER.add_argument('-d', '--dry_run',
        action='store_true',
        help='For testing. Execute without running or saving data and verify output paths')

    #Pull out all of the input strings from the .sh file
    args = INPUT_ARG_PARSER.parse_args()
    #index of the wofs day we are evaluating
    global index_primer
    index_primer = getattr(args, 'wofs_day_idx')
    #path to patches is the location all the patch files are stored
    global path_to_patches
    path_to_patches = getattr(args, 'input_patch_dir_name')
    #checkpoint path is where the model is be stored
    global checkpoint_path
    checkpoint_path = getattr(args, 'output_model_checkpoint_path')
    #outfile_dir is the path to where the predictions will be stored
    global outfile_dir
    outfile_dir = getattr(args, 'output_predictions_checkpoint_path')
    #training data metadata path is the file path of the training metadata we will use to normalize the wofs predictions
    global training_data_metadata_path
    training_data_metadata_path = getattr(args, 'training_data_metadata_path')
    #path to wofs is the location of the raw wofs directory
    global path_to_wofs
    path_to_wofs = getattr(args, 'path_to_wofs')
    #patch size is the size of each patch in each horizontal direction
    global patch_size
    patch_size = getattr(args, 'patch_size')
    global is_dry_run
    is_dry_run = args.dry_run


def make_directory_structure(patches_dirs):

    # Pull out the year, day, model initialization time and ensemble member of this wofs run
    path_components = patches_dirs.split('/')
    yyyy = path_components[-4]  #patches_dirs[len(path_to_patches)+1:len(path_to_patches)+5]
    yyyymmdd = path_components[-3]  #patches_dirs[len(path_to_patches)+6:len(path_to_patches)+14]
    model_initialization = path_components[-2]  #patches_dirs[len(path_to_patches)+15:len(path_to_patches)+19]
    ens_mem = path_components[-1][8:]  #patches_dirs[len(path_to_patches)+28:]
    
    if is_dry_run:
        print("make_directory_structure(patches_dirs)")
        print(path_components)
        print('yyyy', yyyy)
        print('yyyymmdd', yyyymmdd)
        print('model_initialization', model_initialization) 
        print('ens_mem', ens_mem)
        
    # Make the directory structure we need for this wofs run
    # Likely many of these will run in parallel and could be creating files at the same time
    # Use error handling to prevent the parallel tasks from killing one another
    outfile_path = os.path.join(outfile_dir, 'predictions/', *path_components[-4:])
    try:
        if not is_dry_run: os.makedirs(outfile_path)
        print("Making predictions dir", outfile_path)
    except OSError as err:
        print(f"{err} in {err.filename}")

    '''
    #if(not os.path.exists(outfile_dir)):
    try:
        if not is_dry_run: os.mkdir(outfile_dir)
        print("Making directory", outfile_dir)
    except OSError as err:
        print(f"{err} in {err.filename}")

    new_predictions_dir = os.path.join(outfile_dir, 'predictions/')
    #if(not os.path.exists(new_predictions_dir)): #outfile_dir + '/predictions/'
    try:
        if not is_dry_run: os.mkdir(new_predictions_dir)
        print("Making directory", new_predictions_dir)
    except OSError as err:
        print(f"{err} in {err.filename}")

    new_yyyy_dir = os.path.join(new_predictions_dir, yyyy)
    #if(not os.path.exists(outfile_dir + '/predictions/' + '/' + yyyy)):
    #if(not os.path.exists(new_yyyy_dir)):
    try:
        if not is_dry_run: os.mkdir(outfile_dir + '/predictions/' + '/' + yyyy)
        print("Making directory", new_yyyy_dir)
    except OSError as err:
        print(f"{err} in {err.filename}")

    new_yyyymmdd_dir = os.path.join(new_yyyy_dir, yyyymmdd)
    #if(not os.path.exists(outfile_dir + '/predictions/' + '/' + yyyy + '/' + yyyymmdd)):
    #if(not os.path.exists(new_yyyymmdd_dir)):
    try:
        #if not is_dry_run: os.mkdir(outfile_dir + '/predictions/' + '/' + yyyy + '/' + yyyymmdd)
        if not is_dry_run: os.mkdir(new_yyyymmdd_dir)
        print("Making directory", new_yyyymmdd_dir)
    except OSError as err:
        print(f"{err} in {err.filename}")

    new_model_init_dir = os.path.join(new_yyyymmdd_dir, model_initialization)
    #if(not os.path.exists(outfile_dir + '/predictions/' + '/' + yyyy + '/' + yyyymmdd + '/' + model_initialization)):
    #if(not os.path.exists(new_model_init_dir)):
    try:
        #os.mkdir(outfile_dir + '/predictions/' + '/' + yyyy + '/' + yyyymmdd + '/' + model_initialization)
        if not is_dry_run: os.mkdir(new_model_init_dir)
        print("Making directory", new_model_init_dir)
    except OSError as err:
        print(f"{err} in {err.filename}")

    new_ensemble_dir = os.path.join(new_model_init_dir, 'ENS_MEM_' + ens_mem)
    #new_ensemble_dir = os.path.join(new_model_init_dir, ens_mem)
    #if(not os.path.exists(outfile_dir + '/predictions/' + '/' + yyyy + '/' + yyyymmdd + '/' + model_initialization + '/' + ens_mem)):
    #if(not os.path.exists(new_ensemble_dir)):
    try:
        #os.mkdir(outfile_dir + '/predictions/' + '/' + yyyy + '/' + yyyymmdd + '/' + model_initialization + '/ENS_MEM_' + ens_mem)
        if not is_dry_run: os.mkdir(new_ensemble_dir)
        print("Making directory", new_ensemble_dir)
    except OSError as err:
        print(f"{err} in {err.filename}")
    '''
    
    #outfile_path = outfile_dir + '/predictions/' + '/' + yyyy + '/' + yyyymmdd + '/' + model_initialization + '/ENS_MEM_' + ens_mem + '/'
    #outfile_path = os.path.join(new_model_init_dir, 'ENS_MEM_' + ens_mem)
    print("Output path", outfile_path)
    return outfile_path


def make_wofs_predictions(all_patches, outfile_path):

    # Define output filenames
    #metadata_outfile_name = outfile_dir + '/predictions/' + '/wofs_metadata.csv'
    #predictions_outfile_name = outfile_dir + '/predictions/' + '/wofs_predictions.nc'
    metadata_outfile_name = os.path.join(outfile_dir, 'predictions/', 'wofs_metadata.csv')
    predictions_outfile_name = os.path.join(outfile_dir, 'predictions/', 'wofs_predictions.nc')
    if is_dry_run:
        print("metadata_outfile_name", metadata_outfile_name)
        print("predictions_outfile_name", predictions_outfile_name)
    
    #open all the data in one DataArray
    print("Openning all the patches", all_patches)
    wofs = xr.open_mfdataset(all_patches, concat_dim='patch',combine='nested', parallel=True, engine='netcdf4')
    
    # Read in the mean and std for each variable field from the training set to normalize the data
    print("Opening training meta data", training_data_metadata_path)
    training_metadata = xr.open_dataset(training_data_metadata_path)
    global mean_train_ZH
    mean_train_ZH = float(training_metadata.ZH_mean.values)
    global std_train_ZH
    std_train_ZH = float(training_metadata.ZH_std.values)
    training_metadata.close()
    del training_metadata

    # Pull out each wofs field & normalize based on training mean/std
    zh = (wofs.ZH - mean_train_ZH)/std_train_ZH

    # Combine the wofs data into one training input array
    input_array = zh

    # Extract the wofs UH data and metadata
    uh = wofs.UH.values
    n_convective_pixels = wofs.n_convective_pixels.values
    n_uh_pixels = wofs.n_uh_pixels.values
    lat = wofs.lat.values
    lon = wofs.lon.values
    time = wofs.time.values
    forecast_window = wofs.forecast_window.values
    

    #Save out the metadata
    print("Saving metadata_outfile to", metadata_outfile_name)
    if not is_dry_run:
        with open(metadata_outfile_name, 'a') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['time'] + list(time))
            csvwriter.writerow(['lat'] + list(lat))
            csvwriter.writerow(['lon'] + list(lon))
            csvwriter.writerow(['forecast_window'] + list(forecast_window))
            csvwriter.writerow(['n_convective_pixels'] + list(n_convective_pixels))
            csvwriter.writerow(['n_uh_pixels'] + list(n_uh_pixels))

    #read in the unet
    print("Loading model", checkpoint_path)
    fss = make_fractions_skill_score(2, 2, c=1.0, cutoff=0.5, want_hard_discretization=False)
    model = keras.models.load_model(checkpoint_path, 
                    custom_objects={'fractions_skill_score': fss, 
                                    'MaxCriticalSuccessIndex': MaxCriticalSuccessIndex})

    #evaluate the unet on the testing data
    print("Evaluate Model. data size =", input_array.shape)
    print("predictions_outfile_name", predictions_outfile_name)

    print("Performing model predictions")
    y_hat = model.predict(input_array)

    #make a dataset of the true and predicted patch data
    ds_wofs_as_gridrad = xr.Dataset(data_vars=dict(UH = (["patch", "x", "y"], uh),
                            ZH_composite = (["patch", "x", "y"], zh.values.max(axis=3)),
                            ZH = (["patch", "x", "y", "z"], zh.values),
                            stitched_x = (["patch", "x"], wofs.stitched_x.values),
                            stitched_y = (["patch", "y"], wofs.stitched_y.values),
                            predicted_no_tor = (["patch", "x", "y"], y_hat[:,:,:,0]),
                            predicted_tor = (["patch", "x", "y"], y_hat[:,:,:,1]),
                            n_convective_pixels = (["patch"], n_convective_pixels),
                            n_uh_pixels = (["patch"], n_uh_pixels),
                            lat = (["patch"], lat),
                            lon = (["patch"], lon),
                            time = (["patch"], time),
                            forecast_window = (["patch"], forecast_window)),
                    coords=dict(patch = range(y_hat.shape[0]),
                            x = range(32),
                            y = range(32),
                            z = range(1,13)))

    if not is_dry_run and not os.path.exists(predictions_outfile_name):
        # Save out the data, formatted as gridrad
        print("Saving the prediction output", predictions_outfile_name)
        ds_wofs_as_gridrad.to_netcdf(predictions_outfile_name)

    return ds_wofs_as_gridrad


def stitch_patches(one_time, datetime_time):

    # Find the total number of gridpoints in lat and lon
    total_in_lat = int(one_time.stitched_x.max().values + 1)
    total_in_lon = int(one_time.stitched_y.max().values + 1)

    # Find the minimum value of both lat and lon in the array
    lat_min = one_time.lat.min().values
    lon_min = one_time.lon.min().values

    # Because the grid is regular in lat/lon, reconstruct the lat/lon grid for the stitched data
    lats = np.linspace(lat_min, lat_min+(total_in_lat-1)/48, total_in_lat)
    lons = np.linspace(lon_min, lon_min+(total_in_lon-1)/48, total_in_lon)

    # Define empty arrays that will hold the stitched data
    tor_predictions_array = np.zeros((total_in_lat, total_in_lon))
    uh_array = np.zeros((total_in_lat, total_in_lon))
    zh_low_level = np.zeros((total_in_lat, total_in_lon))
    overlap_array = np.zeros((total_in_lat, total_in_lon))
        
    # Loop through all the patches
    for i_patch in range(one_time.patch.values.shape[0]):
        # Find the locations of the corner of this patch in the stitched grid
        min_x = int(one_time.isel(patch=i_patch).stitched_x.min().values)
        min_y = int(one_time.isel(patch=i_patch).stitched_y.min().values)
        max_x = int(one_time.isel(patch=i_patch).stitched_x.max().values + 1)
        max_y = int(one_time.isel(patch=i_patch).stitched_y.max().values + 1)
        
        
        # Reconstruct the stitched grid by taking the average value from all the patches at each grid point
        # At this step, we sum the data from all the patches
        tor_predictions_array[min_x:max_x, min_y:max_y]    += one_time.isel(patch=i_patch).predicted_tor.values      
        uh_array[min_x:max_x, min_y:max_y]          += one_time.isel(patch=i_patch).UH.values
        zh_low_level[min_x:max_x, min_y:max_y]      += one_time.isel(patch=i_patch, z=0).ZH.values*std_train_ZH + mean_train_ZH
        overlap_array[min_x:max_x, min_y:max_y]     += np.ones((patch_size,patch_size))

    
    # Take the average value at each patch by dividing by the total number of patches that contained each pixel
    tor_predictions_array = tor_predictions_array/overlap_array
    uh_array = uh_array/overlap_array
    zh_low_level = zh_low_level/overlap_array

    # Put the stitched grid into a dataset and return 
    predictions = xr.Dataset(data_vars=dict(UH = (["time", "lat", "lon"], uh_array.reshape(1,total_in_lat, total_in_lon)),
                            ZH_1km = (["time", "lat", "lon"], zh_low_level.reshape(1,total_in_lat, total_in_lon)),
                            predicted_tor = (["time", "lat", "lon"], tor_predictions_array.reshape(1,total_in_lat, total_in_lon))),
                    coords=dict(time = [datetime_time],
                            lon = lons,
                            lat = lats))
    predictions.ZH_1km.attrs['units'] = 'dBZ'
    predictions.UH.attrs['units'] = 'm^2/s^2'

    return predictions


def run_predictions_and_interpolation(patches_dirs):
    print('Processing:', patches_dirs)

    # Make the directory structure for the wofs predictions and define the output filepath
    outfile_path = make_directory_structure(patches_dirs)
    # Define output filename
    predictions_for_NCAR_outfile_name = os.path.join(outfile_path, 'wofs_stitched_light.nc')
    if is_dry_run: print("predictions_for_NCAR_outfile_name", predictions_for_NCAR_outfile_name)

    # Load in the patches from all the model times of this wofs run
    all_patches = glob.glob(patches_dirs + '/*')[:1] ## MOSHO - added [:1] for time/memory monitoring
    print("all_patches", all_patches)
    
    # If these files are already made, skip this model time
    if os.path.exists(predictions_for_NCAR_outfile_name):
        print(f"Already exists: {predictions_for_NCAR_outfile_name}")
        #return

    # Make the wofs predictions in the gridrad format
    ds_wofs_as_gridrad = make_wofs_predictions(all_patches, outfile_path)

    # We've made predictions for all the times, but we want to save out each time individually
    times = set(ds_wofs_as_gridrad.time.values)
    ds_stitched_list = []

    # Create process monitor
    proc = ProcessMonitor()

    if not is_dry_run: proc.start_timer()

    # Pull out and process all the data that we have for each time in the hour
    # We need to first stitch patches, then interpolate them back to the WoFS grid
    for time in times:

        # Format the time string correctly
        datetime_time = np.datetime64(netCDF4.num2date(time,'seconds since 2001-01-01'))
        # Isolate all the data from this one time
        one_time = ds_wofs_as_gridrad.where(ds_wofs_as_gridrad.time == time, drop = True)

        # Stitch the patches back together
        predictions = stitch_patches(one_time, datetime_time)
        
        # Add the stitched file to the list of stitched files     
        ds_stitched_list.append(predictions)
        

        # Pull out time, model run, and ensemble member information from the filepath
        path_components = patches_dirs.split('/')
        blen = len(path_to_patches)
        yyyy = path_components[-4]  #patches_dirs[blen+5:blen+9]
        yyyymmdd = path_components[-3]  #patches_dirs[blen+5:blen+13]
        model_initialization = path_components[-2]  #patches_dirs[blen+14:blen+18]
        ens_mem = path_components[-1][8:]  #[-2:]  #patches_dirs[-2:]
        
        mm = str(datetime_time)[5:7]
        dd = str(datetime_time)[8:10]
        hh = str(datetime_time)[11:13]
        minmin = str(datetime_time)[14:16]
        
        if is_dry_run:
            print("patches dirs", patches_dirs)
            print("path_to_patches", path_to_patches)
            print(path_components)
            print("yyyy", yyyy)
            print("yyyymmdd", yyyymmdd)
            print("model_initialization", model_initialization)
            print("ens_mem", ens_mem)

        # Open the original WoFS file to get the grid spacing
        original_wofs = xr.open_dataset(path_to_wofs + '/%s/%s/%s/ENS_MEM_%s/wrfwof_d01_%s-%s-%s_%s:%s:00' % (yyyy, yyyymmdd, model_initialization, ens_mem, yyyy, mm, dd, hh, minmin), engine='netcdf4')

        # Calculate how many gridpoints we need to add to each side to get to the full WoFS grid
        to_add_east_west = round((original_wofs.XLONG.max().values - predictions.lon.max().values + 360)*48 + 1)
        to_add_north_south = round((original_wofs.XLAT.max().values - predictions.lat.max().values)*48 + 1)

        # Add the proper padding to the top, bottom, left and right of the grid to get us back to the exact size of the original wofs file
        zeros_east_west = np.zeros((predictions.predicted_tor.isel(time=0).shape[0], to_add_east_west))
        padded_east_west = np.concatenate((zeros_east_west, predictions.predicted_tor.isel(time=0).values, zeros_east_west), axis=1)
        zeros_north_south = np.zeros((to_add_north_south, padded_east_west.shape[1]))
        padded_predictions = np.concatenate((zeros_north_south, padded_east_west, zeros_north_south), axis=0)

        # Calculate the lats and lons of this gridrad grid, which is now padded with 0s on each side
        padded_lats = np.linspace(predictions.lat.min().values - to_add_north_south/48, predictions.lat.max().values + to_add_north_south/48, padded_predictions.shape[0])
        padded_lons =  np.linspace(predictions.lon.min().values - to_add_east_west/48, predictions.lon.max().values + to_add_east_west/48, padded_predictions.shape[1])

                        
                        
        # Put the lats and lons into 2D arrays instead of 1D arrays
        gridrad_lats = np.tile(padded_lats, padded_lons.shape[0]).reshape(padded_lons.shape[0], padded_lats.shape[0]).T
        gridrad_lons = np.tile(padded_lons - 360, padded_lats.shape[0]).reshape(padded_lats.shape[0], padded_lons.shape[0])
        # Combine these 2 arrays into 1
        gridrad_lats_lons = np.stack((np.ravel(gridrad_lats), np.ravel(gridrad_lons))).T
        
        # Extract the lats and lons from the wofs grid which we are interpolating to
        wofs_lats_lons = np.stack((np.ravel(original_wofs.XLAT.values[0]), np.ravel(original_wofs.XLONG.values[0]))).T
        
        # Make the KD Tree
        tree = spatial.cKDTree(gridrad_lats_lons)

        # Query the tree at the desired points
        # For each point in wofs, find the gridpoint from gridrad that is the closest
        distances, points = tree.query(wofs_lats_lons, k=4)

        # Get the indices of the interpolation points in the new padded gridrad grid
        lat_points, lon_points = np.unravel_index(points, (padded_lats.shape[0], padded_lons.shape[0]))

        # Use inverse distance weighting to get the new grid
        predictions_not_idw = padded_predictions[lat_points, lon_points]
        predictions_idw = (np.sum(predictions_not_idw * 1/distances**2, axis=1)/np.sum(1/distances**2, axis=1)).reshape(original_wofs.XLAT.values.shape[1:])

        # Create a dataset that is formatted exactly like the raw wofs data
        wofs_like = xr.Dataset(data_vars=dict(ML_PREDICTED_TOR = (["Time", "south_north", "west_east"], predictions_idw.reshape((1,) + predictions_idw.shape))),
                        coords=dict(XLAT = (["Time", "south_north", "west_east"], original_wofs.XLAT.values),
                                XLONG = (["Time", "south_north", "west_east"], original_wofs.XLONG.values)))

        # Save out the interpolated file
        print("Save interpolated file", outfile_path + '/wrfwof_d01_%s-%s-%s_%s:%s:00' % (yyyy, mm, dd, hh, minmin))
        if not is_dry_run: wofs_like.to_netcdf(outfile_path + '/wrfwof_d01_%s-%s-%s_%s:%s:00' % (yyyy, mm, dd, hh, minmin))


    # Combine all the stitched files into one, sort the dataset and save it out
    ds_stitched = xr.concat(ds_stitched_list, dim='time')
    ds_stitched = ds_stitched.sortby('time')

    if not is_dry_run:
        proc.end_timer()
        day_idx = int(index_primer)
        basepath = os.path.join(outfile_path, f'process_monitoring_day{day_idx}_stitch_interpolate_')
        proc.write_performance_monitor(output_path=basepath + '_performance.csv')
        proc.plot_performance_monitor(attrs=None, ax=None, write=True,
                                        output_path=basepath + '_performance_plot.png', format='png')
        proc.print(write=True, output_path=basepath + '.csv')
        print("WoFS File", all_patches)
        
        if not os.path.exists(predictions_for_NCAR_outfile_name):
            print(f"Writing: {predictions_for_NCAR_outfile_name}")
            ds_stitched.to_netcdf(predictions_for_NCAR_outfile_name)
    print("Combined stitched files saved to", predictions_for_NCAR_outfile_name)

    # Clean up our script
    ds_stitched.close()
    ds_wofs_as_gridrad.close()
    del ds_wofs_as_gridrad
    #del uh
    #del zh
    #del model
    
    return all_patches




def main():

    # Get the inputs from the .sh file
    get_arguments()  

    # Make the beginning of the directory structure where the predictions will be saved
    print("outfile_dir", outfile_dir)
    if(not os.path.exists(outfile_dir)):
        print(f"Making top level predictions directory: {outfile_dir}")
        if not is_dry_run: os.mkdir(outfile_dir)
    outfile_path = os.path.join(outfile_dir, 'wofs_predictions/')
    if(not os.path.exists(outfile_path)):
        print(f"Making desired predictions directory: {outfile_path}")
        if not is_dry_run: os.mkdir(outfile_path)


    #load in the patches
    #all_patches_dirs = glob.glob(path_to_patches + '/2019/*/*/*')
    members_path = os.path.join(path_to_patches, '2019/*/*/*')
    all_patches_dirs = glob.glob(members_path)
    all_patches_dirs.sort()

    # Create process monitor
    #proc = ProcessMonitor()

    # We do in range 8 because there are ~8000 files. 
    # The task array in slurm only lets us do up to 1000, so we take care of the extra here.
    for i in range(1):
        #if i == 0 and not is_dry_run: proc.start_timer()
        run_predictions_and_interpolation(all_patches_dirs[int(index_primer) + i*1000])
        '''if i == 0 and not is_dry_run:
            proc.end_timer()
            day_idx = int(index_primer)
            basepath = os.path.join(outfile_path, f'process_monitoring_day{day_idx}_idx{i}_1_')
            proc.write_performance_monitor(output_path=basepath + '_performance.csv')
            proc.plot_performance_monitor(attrs=None, ax=None, write=True,
                                          output_path=basepath + '_performance_plot.png', format='png')
            proc.print(write=True, output_path=basepath + '.csv')
            print("WoFS File", all_patches_dirs[int(index_primer) + i*1000])
        '''


if __name__ == "__main__":
    main()

