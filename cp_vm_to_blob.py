import glob, os


def cp_vm_to_blob():
    path_preds = "/datadrive2/wofs-preds"
    files = sorted(glob.glob(os.path.join(path_preds, "2023", "2023**", "**", "ENS_MEM_**", "wrfwof_d01_**")))
    
    file_count = 0
    blob_path_ncar = "https://wofsdltornado.blob.core.windows.net"
    for file in files:
        rel_path = file.split('/datadrive2')[1]
        file_blob = os.path.join(blob_path_ncar, rel_path)
        subprocess.run(["azcopy",
                        "copy",
                        f"{file}",
                        f"{file_blob}"])
        if len(files)%20 == 0:
            file_count += 20
            print(f"DONE - {file_count} of {len(files)}")
        os.remove(f"{file}")

