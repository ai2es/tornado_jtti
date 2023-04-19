from azure.storage.queue import (
    QueueClient, TextBase64EncodePolicy, TextBase64DecodePolicy
)

account_url = 'https://storwofstest003.queue.core.windows.net/?sv=2019-02-02&st=2023-03-09T20%3A39%3A15Z&se=2024-01-01T06%3A00%3A00Z&sp=rp&sig=biy6JZg2n4Wmg%2BLHF4QnVLQvt%2F4W8oYJhXMiaTkyj4U%3D'


def test_monitor_queue():
    queue = QueueClient(account_url=account_url,
                        queue_name='wofs-ucar',
                        message_encode_policy=TextBase64EncodePolicy(),
                        message_decode_policy=TextBase64DecodePolicy())
    while True:
        # Receive messages one-by-one
        print('checking for messages...')

        messages = queue.peek_messages()
        
        if len(messages) == 0:
            print('no message, sleeping')
            continue
        
        else: 
            for message in messages:
                body = json.loads(message.content)
                print('Processing message:')
                for file_string in body['data']:
                    # save file to temp location
        
if __name__ == '__main__':
    test_monitor_queue()
