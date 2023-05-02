import json, time, argparse, traceback, glob
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
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = parse_args()
    
    queue_wofs = QueueClient(account_url=args.account_url_wofs,
                             queue_name=args.queue_name_wofs,
                             message_encode_policy=TextBase64EncodePolicy(),
                             message_decode_policy=TextBase64DecodePolicy())
    
    with Pool(4, maxtasksperchild=1) as p:
        while True:
            msg = queue_wofs.receive_message(visibility_timeout=5*60)

            if msg == None:
                print('No message: sleeping.')
                time.sleep(10)
                
                # start the msgpk process if enough files have been saved
                files = sorted(glob.glob(os.path.join(path_preds, f"ENS_MEM_**", "wrfwof_d01_*")))
                if path_preds[-2:] == "00":
                    if len(files) == 1314:
                        preds_to_msgpk.preds_to_msgpk(path_preds, args)
                    else:
                        continue
                if path_preds[-2:] == "30":
                    if len(files) == 666:
                        preds_to_msgpk.preds_to_msgpk(path_preds, args)
                    else:
                        continue
                continue

            files = json.loads(msg.content)["data"]
            try:
                #p.starmap_async(process_one_file, [(wofs_filepath, args) for wofs_filepath in files])
                p.starmap(process_one_file, [(wofs_filepath, args) for wofs_filepath in files])
            except Exception as e:
                print(traceback.format_exc())
                raise e
            
            path_preds = os.path.join("/datadrive2/wofs-preds/2023/", msg.content["runtime"][:8], msg.content["runtime"][-4:])
            with open('230502_messages.txt', 'a') as a_writer:
                a_writer.write(msg)
            queue_wofs.delete_message(msg)
        
        p.close()
        p.join()
