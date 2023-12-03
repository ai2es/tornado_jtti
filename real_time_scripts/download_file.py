import os, time, subprocess


def download_file(filepath, args):

    year = filepath.split("WOFSRun")[1][:4]
    date = filepath.split("WOFSRun")[1].split("-")[0]
    run_time = filepath.split("/fcst")[0][-4:]
    mem = filepath.split("fcst/mem")[1].split("/wrfwof")[0]
    filename = filepath.split("?se=")[0].rsplit('/', 1)[1]
    path_vm = f"{args.vm_datadrive}/wrf-wofs/{year}/{date}/{run_time}/ENS_MEM_{mem}/"
    path_blob = f"{args.blob_url_ncar}/wrf-wofs/{year}/{date}/{run_time}/ENS_MEM_{mem}/"

    os.umask(0o002)
    os.makedirs(path_vm, exist_ok=True)
    subprocess.run(["azcopy",
                    "copy",
                    f"{filepath}",
                    f"{path_vm}{filename}"])
    
    return f"{path_vm}{filename}"

def download_init_file(filepath, vm_path):
    year = filepath.split("WOFSRun")[1][:4]
    date = filepath.split("WOFSRun")[1].split("-")[0]
    run_time = filepath.split("/")[-3]
    mem = filepath.split("/")[-1].split("?")[0].split(".")[1]
    filename = filepath.split("/")[-1].split("?")[0]
    path_vm = f"{vm_path}/wrf-wofs/{year}/{date}/{run_time}/ENS_MEM_{mem}/"
    os.umask(0o002)
    os.makedirs(path_vm, exist_ok=True)
    subprocess.run(["azcopy",
                    "copy",
                    f"{filepath}",
                    f"{path_vm}{filename}"])
    return f"{path_vm}{filename}"
