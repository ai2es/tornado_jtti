from tensorflow import keras
print("keras version", keras.__version__)
import xarray as xr
print("xarray version", xr.__version__)
import numpy as np
print("numpy version", np.__version__)
import csv
import glob
import argparse
import tensorflow as tf
print("tensorflow version", tf.__version__)
import os
import sys
import pandas as pd
print("pandas version", pd.__version__)
import psutil, py3nvml
from time import time
from datetime import timedelta, datetime #fromtimestamp
sys.path.append("/home/lydiaks2/tornado_project")
from custom_losses import make_fractions_skill_score
from custom_metrics import MaxCriticalSuccessIndex
sys.path.append("../../process_monitoring")
from process_monitor import ProcessMonitor


def get_arguments():

    #Define the strings that explain what each input variable means
    INIT_TIME_HELP_STRING = 'i, where i indicates the ith file of the storm masks from which to make the patches.'
    PATCHES_DIR_HELP_STRING = 'The directory where the unet mask files are stored. This directory should start from the root directory \'/\'.'
    ML_MODEL_DIR_HELP_STRING = 'The directory where the trained model is stored. This directory should start from the root directory \'/\'.'
    PREDICTIONS_DIR_HELP_STRING = 'The directory where the predictions will be stored. This directory should start from the root directory \'/\'.'
    TRAINGING_METADATA_DIR_HELP_STRING = 'The netcdf file where the training model means and stds are for each variable field'

    #Tell python what arguments to expect from the .sh file
    INPUT_ARG_PARSER = argparse.ArgumentParser()
        
    INPUT_ARG_PARSER.add_argument(
        '--patch_day_idx', type=str, required=True,
        help=INIT_TIME_HELP_STRING)
        
    INPUT_ARG_PARSER.add_argument(
        '--input_patch_dir_name', type=str, required=True,
        help=PATCHES_DIR_HELP_STRING)
        
    INPUT_ARG_PARSER.add_argument(
        '--model_checkpoint_path', type=str, required=True,
        help=ML_MODEL_DIR_HELP_STRING)
        
    INPUT_ARG_PARSER.add_argument(
        '--output_predictions_path', type=str, required=True,
        help=PREDICTIONS_DIR_HELP_STRING)
        
    INPUT_ARG_PARSER.add_argument(
        '--training_data_metadata_path', type=str, required=True,
        help=TRAINGING_METADATA_DIR_HELP_STRING)

    #Pull out all of the input strings from the .sh file
    args = INPUT_ARG_PARSER.parse_args()
    #index of the day we are evaluating
    global index_primer
    index_primer = getattr(args, 'patch_day_idx')
    #path to patches is the location all the patch files are stored
    global path_to_patches
    path_to_patches = getattr(args, 'input_patch_dir_name')
    #checkpoint path is where the model will be stored
    global checkpoint_path
    checkpoint_path = getattr(args, 'model_checkpoint_path')
    #checkpoint path is where the predictions will be stored
    global output_predictions_path
    output_predictions_path = getattr(args, 'output_predictions_path')
    #file where the means and stds from the training data are stored (for normalizing the data)
    global training_data_metadata_path
    training_data_metadata_path = getattr(args, 'training_data_metadata_path')


def process_profile(attrs=None, show=True):
    '''
    Obtain information about time, memory, and cpu utilization
    @params: attrs, tuple of select attributes from psutil to 
             output
    @params: show, print process information to screen (1),
             file (2), or both (3)
    '''
    info = psutil.Process().as_dict(attrs=attrs)

    # Select Process attributes if any are specified
    #if not attrs is None:
    info_sel = dict(info)
    for key in info.keys(): # TODO: map/filter??
        details = info[key]
        if isinstance(details, list) and len(details) > 0:
            dets = [str(i) for i in details]
            info[key] = "; ".join(dets)
        elif key == 'create_time':
            info[key] = datetime.fromtimestamp(details).strftime("%A %B %d %Y %H:%M:%S %Z")
        elif key in ['io_counters', 'threads', 'cpu_times', 'memory_maps',
                    'memory_info', 'memory_full_info', 'open_files']: #todo named tuples
            dets = [str(i) for i in details._asdict().items()]
            units = ' sec' if key == 'cpu_times' else (' bytes' if '_info' in key else '') 
            sym = '\n' if show <= 1 else '; '
            info[key] = (units + sym).join(dets)
            #info[key] = f"{units}{sym}".join(dets)
        elif 'percent' in key:
            info[key] = "{}%".format(details)

    if show in [1, 3]:
        print('\nPROCESS INFO \n' + '\n'.join(['{0}: {1}'.format(key, value) for key, value in info.items()]))
    if show in [2, 3]:
        print("\nPROCESS EXECUTION PROFILE")
        df = pd.DataFrame(data=info, index=range(1)).T
        proc_info_fpath = os.path.join(output_predictions_path,
                            'predictions/process_execution_profile.csv')
        print("Writing {}".format(proc_info_fpath))
        df.to_csv(proc_info_fpath, header=False)
        print(df)
    
    return info    


def main():

    # Get the inputs from the .sh file
    get_arguments()  
    
    # Load in all the files from the testing dataset
    all_patches_dirs = glob.glob(path_to_patches + '/2019/*/*')

    # Make the directory structure for where we will save out the data
    if(not os.path.exists(output_predictions_path)):
        os.mkdir(output_predictions_path)
    if(not os.path.exists(output_predictions_path + '/predictions')):
        os.mkdir(output_predictions_path + '/predictions')

    # Define the output filenames
    predictions_outfile_name = output_predictions_path + '/predictions/test_predictions.nc'
    metadata_outfile_name = output_predictions_path + '/predictions/test_metadata.csv'
    
    # Create process monitor and start tracking process information
    proc = ProcessMonitor()
    proc.start_timer()
    
    #open all the testing data in one DataArray
    test = xr.open_mfdataset(all_patches_dirs, concat_dim='patch',combine='nested', parallel=True, engine='netcdf4')


    # Read in the mean and std for each variable field from the training set to normalize the test set
    training_metadata = xr.open_dataset(training_data_metadata_path)
    mean_train_ZH = float(training_metadata.ZH_mean.values)
    std_train_ZH = float(training_metadata.ZH_std.values)

    # Separate the data array into ml input, ml output and metadata
    # Pull out each data field & normalize based on training mean/std
    zh = (test.ZH.isel(z=[1,2,3,4,5,6,7,8,9,10,11,12]) - mean_train_ZH)/std_train_ZH
    labels = test.labels

    # TESTING 
    # Put back together into one ML input array
    input_array = zh #[:1]
    
    # Define the ml labels
    output_ds = labels
    output_array = output_ds.values #[:1]

    # Pull out the metadata
    n_convective_pixels = test.n_convective_pixels.values
    n_pixels_EF0_EF1 = test.n_pixels_EF0_EF1.values
    n_pixels_EF2_EF5 = test.n_pixels_EF2_EF5.values
    n_tornadic_pixels = test.n_tornadic_pixels.values
    lat = test.lat.values
    lon = test.lon.values
    time = test.time.values
    forecast_window = test.forecast_window.values
    
    #Save out the metadata
    with open(metadata_outfile_name, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['time'] + list(time))
        csvwriter.writerow(['lat'] + list(lat))
        csvwriter.writerow(['lon'] + list(lon))
        csvwriter.writerow(['forecast_window'] + list(forecast_window))
        csvwriter.writerow(['n_convective_pixels'] + list(n_convective_pixels))
        csvwriter.writerow(['n_pixels_EF0_EF1'] + list(n_pixels_EF0_EF1))
        csvwriter.writerow(['n_pixels_EF2_EF5'] + list(n_pixels_EF2_EF5))
        csvwriter.writerow(['n_tornadic_pixels'] + list(n_tornadic_pixels))


    # Read in the unet, with MaxCSI and fss defined
    fss = make_fractions_skill_score(2, 2, c=1.0, cutoff=0.5, want_hard_discretization=False)
    model = keras.models.load_model(checkpoint_path, 
                    custom_objects={'MaxCriticalSuccessIndex': MaxCriticalSuccessIndex,
                                    'fractions_skill_score': fss })
        
    #evaluate the unet on the testing data
    y_hat = model.predict(input_array)


    # Obtain memory and timing profile information
    '''process_profile(attrs=['name', 'cwd', 'cmdline', 'exe', 'pid',
                            'create_time', 'cpu_percent', 'cpu_times',
                            'memory_info', 'memory_full_info',
                            'memory_percent', 'num_threads'], show=3)
    '''
    proc.end_timer()
    proc.write_performance_monitor(output_path=None)
    proc.plot_performance_monitor(attrs=None, ax=None, output_path=None, write=True)
    proc.print()
    print()


    #make a dataset of the true and predicted patch data with the metadata
    ds_return = xr.Dataset(data_vars=dict(truth_labels = (["patch", "x", "y"], output_array),
                                ZH_composite = (["patch", "x", "y"], test.ZH.values.max(axis=3)),
                                ZH_1km = (["patch", "x", "y"], test.ZH.values[:,:,:,0]),
                                predicted_no_tor = (["patch", "x", "y"], y_hat[:,:,:,0]),
                                predicted_tor = (["patch", "x", "y"], y_hat[:,:,:,1]),
                                n_convective_pixels = (["patch"], n_convective_pixels),
                                n_pixels_EF0_EF1 = (["patch"], n_pixels_EF0_EF1),
                                n_pixels_EF2_EF5 = (["patch"], n_pixels_EF2_EF5),
                                n_tornadic_pixels = (["patch"], n_tornadic_pixels),
                                lat = (["patch"], lat),
                                lon = (["patch"], lon),
                                time = (["patch"], time),
                                forecast_window = (["patch"], forecast_window)),
                        coords=dict(patch = range(y_hat.shape[0]),
                                x = range(32),
                                y = range(32)))


    print(ds_return)

    #save out the prediction and truth values
    ds_return.to_netcdf(predictions_outfile_name)
    #"""


if __name__ == "__main__":
    main()

    '''    
    process_profile(attrs=['name', 'cwd', 'cmdline', 'exe', 'pid',
                            'create_time', 'cpu_percent', 'cpu_times',
                            'memory_info', 'memory_full_info',
                            'memory_percent', 'num_threads'], show=3)
    '''


