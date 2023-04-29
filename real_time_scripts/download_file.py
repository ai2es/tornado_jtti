import subprocess


def download_file(filepath):

    year = filepath.split("WOFSRun")[1][:4]
    date = filepath.split("WOFSRun")[1].split("-")[0]
    run_time = filepath.split("/fcst")[0][-4:]
    mem = filepath.split("fcst/mem")[1].split("/wrfwof")[0]
    filename = filepath.split("?se=")[0].rsplit('/', 1)[1]
    path = f"{args.blob_url_ncar}/wrf-wofs/{year}/{date}/{run_time}/ENS_MEM_{mem}/"
                
    subprocess.run(["azcopy",
                    "copy",
                    f"{msg}",
                    f"{path}{filename}"])
    
    return f"{path}{filename}"
