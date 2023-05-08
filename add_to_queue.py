import subprocess
from azure.storage.queue import QueueClient, TextBase64EncodePolicy, TextBase64DecodePolicy

queue_ncar = QueueClient(account_url='https://wofsdltornado.queue.core.windows.net/?sv=2021-12-02&ss=bfqt&srt=sco&sp=rwdlacupiytfx&se=2023-06-15T15:00:00Z&st=2023-04-26T15:00:00Z&spr=https&sig=H2JOkeMn0UuhOqyuifMc%2BCfoSXoN5ZRL7mCe9iGEjBM%3D',
                         queue_name='wrf-wofs-queue',
                         message_encode_policy=TextBase64EncodePolicy(),
                         message_decode_policy=TextBase64DecodePolicy())

files_txt = open("files.txt", "r")
files = files_txt.read().split('\n')[:-1]

for filename in files:
    try:
        fn = filename.split(': ')[1].split(';')[0]
    except:
        print(filename)
        continue
    queue_ncar.send_message(f"https://wofsdltornado.blob.core.windows.net/wrf-wofs/2023/{fn}")
