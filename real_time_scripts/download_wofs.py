import json, argparse, subprocess
from azure.storage.queue import (
    QueueClient, TextBase64EncodePolicy, TextBase64DecodePolicy
)


def test_monitor_queue():
    
    parser = argparse.ArgumentParser(description='Save WoFS to azure datablob')
    parser.add_argument('--account_url', type=str, required=True,
                        help='Account url for WoFS file location')
    parser.add_argument('--queue_name', type=str, required=True,
                        help='Queue name for WoFS file location')
    args = parser.parse_args()
    
    queue = QueueClient(account_url=args.account_url,
                        queue_name=args.queue_name,
                        message_encode_policy=TextBase64EncodePolicy(),
                        message_decode_policy=TextBase64DecodePolicy())
    while True:
        
        print('Checking for messages...')
        msg = queue.receive_message(visibility_timeout=120)
        
        if msg == None:
            print('No message: sleeping.')
            time.sleep(10)
            continue

        with open('message_log.txt', 'a') as message_log:
            message_log.write(msg.content)
        try: 
            print('Saving message content to storage blob:')
            body = json.loads(msg.content)
            print(f"\t{body}")
            for file_string in body["data"]:
                year = file_string.split("WOFSRun")[1][:4]
                date = file_string.split("WOFSRun")[1].split("-")[0]
                run_time = file_string.split("/fcst")[0][-4:]
                mem = file_string.split("fcst/mem")[1].split("/wrfwof")[0]
                filename = file_string.split("?se=")[0].rsplit('/', 1)[1]
                path = f"/datadrive/wofs/{year}/{date}/{run_time}/ENS_MEM_{mem}/"
                
                os.makedirs(path, exist_ok = True)
                subprocess.run(["azcopy",
                                "copy",
                                f"{file_string}",
                                f"{path}{filename}"])
                
            time.sleep(10)
            queue.delete_message(msg)
            
        except Exception as e:
            print(f'ERROR: {e}')
        
if __name__ == '__main__':
    test_monitor_queue()
