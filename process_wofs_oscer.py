import os, glob, json, time, argparse, traceback, datetime
# from azure.storage.queue import QueueClient, TextBase64EncodePolicy, TextBase64DecodePolicy
# from azure.storage.blob import BlobServiceClient
from multiprocessing.pool import Pool
from real_time_scripts import preds_to_msgpk_oscer
import pandas as pd
import warnings
import logging
import subprocess
warnings.filterwarnings("ignore")


def process_one_file(wofs_filepath, args):
    from real_time_scripts import download_file, wofs_to_preds_oscer
    # ncar_filepath = download_file.download_file(wofs_filepath, args)
    vm_filepath = wofs_to_preds_oscer.wofs_to_preds(wofs_filepath, args)
    print(vm_filepath)
    return vm_filepath

def parse_args():
    parser = argparse.ArgumentParser(description='Process single timestep from WoFS to msgpk')
    
    # download_file ___________________________________________________
    parser.add_argument('--account_url_wofs', type=str, required=True,
                        help='WoFS queue account url')
    parser.add_argument('--queue_name_wofs', type=str, required=True,
                        help='WoFS queue name for available files')    
    parser.add_argument('--blob_url_ncar', type=str, required=True,
                        help='NCAR path to storage blob')
    parser.add_argument('--hours_to_analyze', type=int, required=True,
                        help='Number of hours to analyze, starting from 0, for each runtime.')
    
    # wofs_to_preds ___________________________________________________
    parser.add_argument('--vm_datadrive', type=str, required=True,
                        help='NCAR VM path to datadrive')
    
    # relative directories for various files to be saved
    parser.add_argument('--dir_wofs', type=str, required=True,
                        help='Directory to store WoFS on NCAR VM')
    parser.add_argument('--dir_preds', type=str, required=True,
                        help='Directory to store the predictions. Prediction files are saved individually for each WoFS files. The prediction files are saved of the form: <WOFS_FILENAME>_predictions.nc')
    parser.add_argument('--dir_patches', type=str,
                        help='Directory to store the patches of the interpolated WoFS data. The files are saved of the form: <WOFS_FILENAME>_patched_<PATCH_SHAPE>.nc. This field is optional and mostly for testing')
    
    # Various other parameters
    parser.add_argument('--datetime_format', type=str, required=True, default="%Y-%m-%d_%H:%M:%S",
                        help='Date time format string used in the WoFS file name. See python datetime module format codes for more details (https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes). ')
    parser.add_argument('-p', '--patch_shape', type=tuple, default=(32,),
                        help='Shape of patches. Can be empty (), 2D (xy,), 2D (x, y), or 3D (x, y, h) tuple. If empty tuple, patching is not performed. If tuple length is 1, the x and y dimension are the same. Ex: (32) or (32, 32).')
    parser.add_argument('--with_nans', action='store_true', 
        help='Set flag such that data points with reflectivity=0 are stored as NaNs. Otherwise store as normal floats. It is recommended to set this flag')
    parser.add_argument('-Z', '--ZH_only', action='store_true',  
        help='Use flag to only extract the reflectivity (COMPOSITE_REFL_10CM and REFL_10CM), updraft (UP_HELI_MAX) and forecast time (Times) data fields, excluding divergence and vorticity fields. Additionally, do not compute divergence and vorticity. Regardless of the value of this flag, only reflectivity is used for training and prediction.')
    parser.add_argument('-f', '--fields', nargs='+', type=str,
        help='Space delimited list of additional WoFS fields to store. Regardless of whether these fields are specified, only reflectivity is used for training and prediction. Ex use: --fields U WSPD10MAX W_UP_MAX')
    
    # Preds to msgpk ___________________________________________________
    parser.add_argument('--dir_preds_msgpk', type=str, required=True,
        help='Directory to store the machine learning predictions in MessagePack format. The files are saved of the form: <wofs_sparse_prob_<DATETIME>.msgpk')
    parser.add_argument('-v', '--variables', nargs='+', type=str,
        help='List of string variables to save out from predictions. Ex use: --variables ML_PREDICTED_TOR REFL_10CM')
    parser.add_argument('-t', '--thresholds', nargs='+', type=float,
        help='Space delimited list of float thresholds. Ex use: --thresholds 0.08 0.07')

    # Model directories and files
    parser.add_argument('--loc_model', type=str, required=True,
        help='Trained model directory or file path (i.e. file descriptor)')
    parser.add_argument('--file_trainset_stats', type=str, required=True,
        help='Path to training set statistics file (i.e., training metadata in Lydias code) for normalizing test data. Contains the means and std computed from the training data for at least the reflectivity (i.e., ZH)')
    
    # Functionality parameters
    parser.add_argument('-w', '--write', type=int, default=0,
        help='Write/save data and/or figures. Set to 0 to save nothing, set to 1 to only save WoFS predictions file (.nc), set to 2 to only save all .nc data files, set to 3 to only save figures, and set to 4 to save all data files and all figures')
    parser.add_argument('-d', '--debug_on', action='store_true',
        help='For testing and debugging. Execute without running models or saving data and display output paths')
    
    # If loading model weights and using hyperparameters from_weights
    hyperparams_subparser = parser.add_subparsers(title='model_loading', dest='load_options', 
        help='optional, additional model loading options')
    hp_parser = hyperparams_subparser.add_parser('load_weights_hps', 
        help='Specifiy details to load model weights and UNetHyperModel hyperparameters')
    hp_parser.add_argument('--hp_path', type=str, required=True,
        help='Path to the csv containing the top hyperparameters')
    hp_parser.add_argument('--hp_idx', type=int, default=0,
        help='Index indicating the row to use within the csv of the top hyperparameters')
    
    args = parser.parse_args()
    return args

def append_to_available_dates_csv(new_rundatetime, args):
    run_date_dt = datetime.datetime.strptime(new_rundatetime, "%Y%m%d%H%M")
    if run_date_dt.hour < 4:
        run_date_dt = run_date_dt - datetime.timedelta(hours=24)
    run_date_dt_str = run_date_dt.strftime("%Y%m%d%H%M")
    conn_string = "DefaultEndpointsProtocol=https;AccountName=wofsdltornado;AccountKey=gS4rFYepIg7Rtw0bZcKjelcJ9TNoVEhKV5cZBGc1WEtRZ4eCn35DhDnaDqugDXtfq+aLnA/rD0Bc+ASt4erSzQ==;EndpointSuffix=core.windows.net"
    container = args.dir_preds_msgpk

    blob_service_client = BlobServiceClient.from_connection_string(conn_string)
    container_client = blob_service_client.get_container_client(container)

    filename = "available_dates.csv"
    blob_client = container_client.get_blob_client(filename)
    with open(filename, "wb") as new_blob:
        download_stream = blob_client.download_blob()
        new_blob.write(download_stream.readall())

    df = pd.read_csv(filename)
    df.loc[len(df.index)] = run_date_dt_str

    csv_string = df.to_csv(index=False)
    csv_bytes = csv_string.encode()
    blob_client.upload_blob(csv_bytes, overwrite=True)

    os.remove(filename)


if __name__ == '__main__':
    
    args = parse_args()

    if args.hours_to_analyze > 3:
        len_rundatetimes = 11*args.hours_to_analyze*12 + 10*3*12
    elif args.hours_to_analyze < 4:
        len_rundatetimes = 21*args.hours_to_analyze*12
    else:
        raise ValueError(f"Argument hours_to_analyze should be between 0 and 6 but was {args.hours_to_analyze}")

    wofs_fps = sorted(glob.glob("/ourdisk/hpc/ai2es/tornado/wofs-preds-2023-hgt/20230524/**/**/**.nc"))

    with Pool(9) as p:

        # begin processing
        try:
            result = p.starmap(process_one_file,
                               [(wofs_fp, args) for wofs_fp in wofs_fps],
                               )
            #result = process_one_file(wofs_fps[0], args)
            print(result)
            preds_to_msgpk_oscer.preds_to_msgpk(result, args)

        except Exception as e:
            print(traceback.format_exc())