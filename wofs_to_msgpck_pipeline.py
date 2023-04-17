import unittest, datetime, sys, os.path, asyncio, json, time, subprocess
import yaml

from azure.storage.queue import (
    QueueClient, TextBase64EncodePolicy, TextBase64DecodePolicy
)

account_url = 'https://storwofstest003.queue.core.windows.net/?sv=2019-02-02&st=2023-03-09T20%3A39%3A15Z&se=2024-01-01T06%3A00%3A00Z&sp=rp&sig=biy6JZg2n4Wmg%2BLHF4QnVLQvt%2F4W8oYJhXMiaTkyj4U%3D'


def test_monitor_queue():
#    queue = QueueClient(account_url=account_url,
#                        queue_name='wofs-ucar',
#                        message_encode_policy=TextBase64EncodePolicy(),
#                        message_decode_policy=TextBase64DecodePolicy())
#    while True:
#        # Receive messages one-by-one
#        print('checking for messages...')

#        messages = queue.peek_messages()
        
#        if len(messages) == 0:
#            print('no message, sleeping')
#            continue
        
#        else: 
#            for message in messages:
#                body = json.loads(message.content)
#                print('Processing message:')
#                for file_string in body['data']:
                    # save file to temp location
        
        with open("config.yml") as config_file:
            config = yaml.safe_load(config_file)
            
        cmd = ["python",
               "wofs_raw_predictions_azure.py",
                f"--loc_wofs={config['loc_wofs']}",
                f"--datetime_format={config['datetime_format']}",
                f"--dir_preds={config['dir_preds']}",
                f"--dir_patches={config['dir_patches']}",
                f"--dir_figs={config['dir_figs']}",
                f"--with_nans={config['with_nans']}",
                f"--fields={config['fields']}",
                f"--loc_model={config['loc_model']}",
                f"--file_trainset_stats={config['file_trainset_stats']}",
                f"--write={config['write']}",
                f"--dry_run={config['dry_run']}"]
        
        # subprocess.Popen(cmd, shell=False, stdout="/dev/null",
        #                  stderr=subprocess.PIPE).communicate() # Popen will wait until the python process is finished
        subprocess.run(cmd, shell=False, stdout=False,
                       stderr=subprocess.PIPE, check=True).communicate()
    
            # add Monique's pipeline script here (WoFS to WoFS-grid preds and variables)
                    # Must save: predictions and radar reflectivity, updraft helcity (0-2),
                    # Must save: updraft helicity (2-5)
                    # Must save: archival tier storage blob
                    # Nice to have: 1) U and V windfields and 2) W if budget allows
                        # U and V are useful for adding divergence and vorticity later
                    # Retrieve later from summary files: Cape (mixed layer), Cin, USHR, VSHR
                                    
            # add a monitoring line to determine when Monique's script is finished
                # add Unet outputs to msgpack script here
                
            # transfer msgpack files to hot tier data blob
            
            # time.sleep(10)

if __name__ == '__main__':
    test_monitor_queue()
