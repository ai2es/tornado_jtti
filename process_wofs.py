import json, time, argparse, traceback, glob, os, datetime
from azure.storage.queue import QueueClient, TextBase64EncodePolicy, TextBase64DecodePolicy
from multiprocessing.pool import Pool
from real_time_scripts import preds_to_msgpk


def process_one_file(wofs_filepath, args):
    from real_time_scripts import download_file, wofs_to_preds
    ncar_filepath = download_file.download_file(wofs_filepath, args)
    vm_filepath = wofs_to_preds.wofs_to_preds(ncar_filepath, args)

def parse_args():
    parser = argparse.ArgumentParser(description='Process single timestep from WoFS to msgpk')
    
    # download_file ___________________________________________________
    parser.add_argument('--account_url_wofs', type=str, required=True,
                        help='WoFS queue account url')
    parser.add_argument('--queue_name_wofs', type=str, required=True,
                        help='WoFS queue name for available files')    
    parser.add_argument('--blob_url_ncar', type=str, required=True,
                        help='NCAR path to storage blob')    
    
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
    parser.add_argument('-f', '--fields', type=str, nargs='+',
        help='Space delimited list of additional WoFS fields to store. Regardless of whether these fields are specified, only reflectivity is used for training and prediction. Ex use: --fields U WSPD10MAX W_UP_MAX')

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
    
    # Preds to msgpk ___________________________________________________
    parser.add_argument('--dir_preds_msgpk', type=str, required=True,
        help='Directory to store the machine learning predictions in MessagePack format. The files are saved of the form: <wofs_sparse_prob_<DATETIME>.msgpk')
    parser.add_argument('--variable', type=str, required=True, 
        help='Variable to save out from predictions')
    parser.add_argument('--threshold', type=float, required=True, 
        help='If probability of tornado is greater than or equal to this threshold value, build tornado tracks')

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

if __name__ == '__main__':
    
    args = parse_args()
    
    queue_wofs = QueueClient(account_url=args.account_url_wofs,
                             queue_name=args.queue_name_wofs,
                             message_encode_policy=TextBase64EncodePolicy(),
                             message_decode_policy=TextBase64DecodePolicy())
    
    def preds_to_msgpk_callback(result):
        for item in result:
            print(f'DONE with wofs_to_preds for {item}', flush=True)
    
    with Pool(16, maxtasksperchild=1) as p:
        while True:
            msg = queue_wofs.receive_message(visibility_timeout=40)

            # check to see if queue is empty
            if msg == None:
                print('No message: sleeping.')
                time.sleep(10)
                continue

            msg_dict = json.loads(msg.content)

            # check to see if message has expired
            datetime_string = msg_dict["data"][0].split('se=')[1].split('%')[0]
            expiration_datetime = datetime.datetime.strptime(datetime_string, '%Y-%m-%dT%H')
            if expiration_datetime < datetime.datetime.now():
                print(f"EXPIRED: {msg_dict['jobId']} - {expiration_datetime}")
                queue_wofs.delete_message(msg)
                continue
            
            # begin processing
            try:
                p.starmap_async(process_one_file,
                                [(wofs_fp, args) for wofs_fp in msg_dict["data"]],
                                callback=preds_to_msgpk_callback)
            except Exception as e:
                print(traceback.format_exc())
                with open(f"./logs/{rundate}_msgs_errors.txt", 'a') as file:
                    file.write('\n')
                    file.write(msg.content)
                #raise e
            
            rundatetime = msg_dict["runtime"]
            filename = msg_dict["data"][0].split('?se')[0].rsplit('/')[-1] + "_predictions.nc"
            path_preds_timestep = f"/datadrive2/wofs-preds/{rundate[:4]}/{rundate}/{rundatetime}"
            
            if len(glob.glob(path_preds_timestep)) == 18:
                path_preds_timestep_msgpk = f"/datadrive2/wofs-preds-msgpk/{rundate[:4]}/{rundate}/{rundatetime}/"
                timestep = filename[11:30].replace('-', '').replace('_', '')
                preds_to_msgpk.preds_to_msgpk(path_preds_timestep,
                                              path_preds_timestep_msgpk,
                                              timestep,
                                              args)
            
                with open(f"./logs/{rundate}_msgs.txt", 'a') as file:
                    file.write('\n')
                    file.write(msg.content)
                queue_wofs.delete_message(msg)

        p.close()
        p.join()
