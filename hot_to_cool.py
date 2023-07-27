import argparse, datetime, subprocess, traceback
from multiprocessing.pool import Pool


def parse_args():
    parser = argparse.ArgumentParser(description='Update BlobAccessTier for specified date to specified tier')
    parser.add_argument('--date', type=int, required=True,
                        help="Date for which the data's BlobAccessTier should be changed. Format: YYYYMMDD")
    args = parser.parse_args()
    return args

def set_properties_directory(dt_dir):
    try:
        subprocess.run(["azcopy",
                        "set-properties",
                        f"https://wofsdltornado.blob.core.windows.net/wofs-dl-preds/{dt_dir}",
                        "--block-blob-tier=cool",
                        "--recursive=true"])
    except Exception as e:
        print(traceback.format_exc())
            
if __name__ == '__main__':
    
    args = parse_args()
    
    dt_start = datetime.datetime.strptime(f'{args.date}1700', '%Y%m%d%H%M')
    dts = [dt_start + datetime.timedelta(minutes=x) for x in range(0, 630, 30)]
    dts.sort()
    dirs = [dt.strftime('%Y%m%d%H%M') for dt in dts]

    with Pool(7) as p:
        p.map(set_properties_directory, dirs)
