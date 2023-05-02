import json, time, argparse, traceback
from azure.storage.queue import QueueClient, TextBase64EncodePolicy, TextBase64DecodePolicy
from multiprocessing.pool import Pool, ThreadPool


def process_one_file(ncar_filepath, args):
    from real_time_scripts import wofs_to_preds
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
    # NCAR's queue urls and names and blob url
    parser.add_argument('--account_url_ncar', type=str, required=True,
                        help='NCAR queue account url')
    parser.add_argument('--queue_name_ncar_wofs_to_preds', type=str, required=True,
                        help='NCAR queue name for downloaded WoFS files')
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
        help='Use flag to only extract the reflectivity (COMPOSITE_REFL_10CM and REFL_10CM), updraft (UP_HELI_MAX) and forecast time (Times) data fields, excluding divergence and vorticity fields. Regardless of the value of this flag, only reflectivity is used for training and prediction.')
    parser.add_argument('-f', '--fields', type=str, nargs='+',
        help='Space delimited list of additional WoFS fields to store. Regardless of whether these fields are specified, only reflectivity is used for training and prediction. Ex use: --fields U WSPD10MAX W_UP_MAX')

    # Model directories and files
    parser.add_argument('--loc_model', type=str, required=True,
        help='Trained model directory or file path (i.e. file descriptor)')
    parser.add_argument('--file_trainset_stats', type=str, required=True,
        help='Path to training set statistics file (i.e., training metadata in Lydias code) for normalizing test data. Contains the means and std computed from the training data for at least the reflectivity (i.e., ZH)')
    
    # TODO: If loading model weights and using hyperparameters from_weights
    hyperparams_subparser = parser.add_subparsers(title='model_loading', dest='load_options', 
        help='optional, additional model loading options')
    hp_parser = hyperparams_subparser.add_parser('load_weights_hps', 
        help='Specifiy details to load model weights and UNetHyperModel hyperparameters')
    hp_parser.add_argument('--hp_path', type=str, required=True,
        help='Path to the csv containing the top hyperparameters')
    hp_parser.add_argument('--hp_idx', type=int, default=0,
        help='Index indicating the row to use within the csv of the top hyperparameters')
    
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
    
    args = parser.parse_args()
    
    return args

if __name__ == '__main__':
        
    args = parse_args()
    
    queue_wofs = QueueClient(account_url=args.account_url_ncar,
                             queue_name=args.queue_name_ncar_wofs_to_preds,
                             message_encode_policy=TextBase64EncodePolicy(),
                             message_decode_policy=TextBase64DecodePolicy())
    with Pool(4) as p:
        while True:
            messages = queue_wofs.receive_messages(messages_per_page=18, visibility_timeout=5*60)
            for msg_batch in messages.by_page():
                msg_batch_list = []
                for msg in msg_batch:
                    msg_batch_list.append(msg.content)
                    queue_wofs.delete_message(msg)
                    try:
                        p.starmap(process_one_file, [(ncar_filepath, args) for ncar_filepath in msg_batch_list])
                    except Exception as e:
                        print(traceback.format_exc())
                        raise e
        p.close()
        p.join()
