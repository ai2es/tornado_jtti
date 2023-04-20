import json, argparse
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
        # Receive messages one-by-one
        print('checking for messages...')
        messages = queue.receive_messages()
        
        #if len(messages) == 0:
            #print('no message, sleeping')
            #continue
        
        #else: 
        with open('message_log.txt', 'a') as message_log:
            for message in messages:
                message_log.write(message.content)
                body = json.loads(message.content)
                print('Saving message to storage blob:')
                for file_string in body['data']:
                    print(file_string)
                    
        
if __name__ == '__main__':
    test_monitor_queue()
