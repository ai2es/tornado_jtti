import os, glob
import xarray as xr


engine = 'netcdf4'
preds_dir = "/ourdisk/hpc/ai2es/tornado/wofs-preds-2023/"
init_dir = "/ourdisk/hpc/ai2es/tornado/wofs-preds-2023-init/2023/"

dates = [i[-8:] for i in glob.glob(preds_dir+'2023****', recursive=True)]

for date in dates:

    init_times = [i[-4:] for i in glob.glob(preds_dir+date+'/****', recursive=True)]
    wofs_init_times = [i[-12:] for i in glob.glob(init_dir+date+'/*', recursive=True)]

    for init_time in init_times:
        if init_time.startswith('0'):
            wofs_init_time = list(filter(lambda x: x.endswith(init_time), wofs_init_times))[0]
        else:
            wofs_init_time = date+init_time
        
        for ens in range(1, 19):
            filepath_wofs = init_dir + f"{date}/{wofs_init_time}/ENS_MEM_{ens}/wrfinput_d01_predictions.nc"
            print(filepath_wofs)
            wofs = xr.open_dataset(filepath_wofs, engine=engine, decode_times=False, decode_coords=True)
    
            files = glob.glob(preds_dir+f"{date}/{init_time}/ENS_MEM_{ens}/**.nc", recursive=True)
    
            for file in files:
                print(file)
                preds = xr.open_dataset(file, engine=engine, decode_times=False, decode_coords=True)
                preds = preds.merge(wofs[["HGT", "PH", "PHB"]], compat='override')
    
                savepath = file.replace('wofs-preds-2023', 'wofs-preds-2023-hgt')
                os.makedirs(savepath.split('wrfwof')[0], exist_ok=True)
    
                encoding_vars = [k for k in preds.variables.keys() if k not in ["Times", "Time"]]
                encoding = {var: {"zlib":True, "complevel":4, "least_significant_digit":2} for var in encoding_vars}
                if "ML_PREDICTED_TOR" in encoding.keys():
                    encoding["ML_PREDICTED_TOR"]['least_significant_digit'] = 3
                preds.to_netcdf(savepath, encoding=encoding)
