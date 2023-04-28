import os, json, time, argparse, subprocess
from azure.storage.queue import (
    QueueClient, TextBase64EncodePolicy, TextBase64DecodePolicy
)


def test_monitor_queue():
    
    parser = argparse.ArgumentParser(description='Save WoFS to azure datablob')
    parser.add_argument('--account_url_wofs', type=str, required=True,
                        help='WoFS queue account url')
    parser.add_argument('--queue_name_wofs', type=str, required=True,
                        help='WoFS queue name for available files')
    parser.add_argument('--blob_url_ncar', type=str, required=True,
                        help='NCAR path to storage blob')
    parser.add_argument('--account_url_ncar', type=str, required=True,
                        help='NCAR queue account url')
    parser.add_argument('--queue_name_ncar_wofs_to_preds', type=str, required=True,
                        help='NCAR queue name for downloaded WoFS files')
    args = parser.parse_args()
    
    queue_wofs = QueueClient(account_url=args.account_url_wofs,
                        queue_name=args.queue_name_wofs,
                        message_encode_policy=TextBase64EncodePolicy(),
                        message_decode_policy=TextBase64DecodePolicy())
    
    queue_ncar_wofs_to_preds = QueueClient(account_url=args.account_url_ncar,
                                           queue_name=args.queue_name_ncar_wofs_to_preds,
                                           message_encode_policy=TextBase64EncodePolicy(),
                                           message_decode_policy=TextBase64DecodePolicy())
    
    while True:
        
        print('Checking for messages...')
        msg = queue_wofs.receive_message(visibility_timeout=120)
        
        if msg == None:
            print('No message: sleeping.')
            time.sleep(10)
            continue

        body = json.loads(msg.content)
        try: 
            print('Saving message content to storage blob:')
            print(f"__NEW__: {body}")
            for file_string in body["data"]:
                year = file_string.split("WOFSRun")[1][:4]
                date = file_string.split("WOFSRun")[1].split("-")[0]
                run_time = file_string.split("/fcst")[0][-4:]
                mem = file_string.split("fcst/mem")[1].split("/wrfwof")[0]
                filename = file_string.split("?se=")[0].rsplit('/', 1)[1]
                path = f"{args.blob_url_ncar}/wrf-wofs/{year}/{date}/{run_time}/ENS_MEM_{mem}/"
                
                subprocess.Popen(["azcopy",
                                  "copy",
                                  f"{file_string}",
                                  f"{path}{filename}"])
                
                queue_ncar_wofs_to_preds.send_message(f"{path}{filename}")
                
            queue_wofs.delete_message(msg)
            
        except Exception as e:
            print(f'ERROR: {e}')
        
if __name__ == '__main__':
    test_monitor_queue()
