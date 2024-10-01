import os, glob, argparse
import xarray as xr
from multiprocessing.pool import Pool


engine = 'netcdf4'    

def parse_args():
    
    parser = argparse.ArgumentParser(description='Process single timestep from WoFS to msgpk')
    
    # download_file ___________________________________________________
    parser.add_argument('--preds_dir', type=str, required=True,
                        help='Directory with ML preds that do not include height')
    parser.add_argument('--init_dir', type=str, required=True,
                        help='WoFS initiation file used to back-fill preds_dir files with height')    
    parser.add_argument('--date', type=str, required=True,
                        help='Date to process in preds_dir. Leave blank to run through many dates')
    args = parser.parse_args()
    
    return args

def process_one_file(file, wofs):
    
    preds = xr.open_dataset(file, engine=engine, decode_times=False, decode_coords=True)
    preds = preds.merge(wofs[["HGT", "P", "PB", "PH", "PHB"]], compat='override', join='override')

    savepath = file.replace('wofs-preds-2023', 'wofs-preds-2023-hgt')
    os.makedirs(savepath.split('wrfwof')[0], exist_ok=True)

    encoding_vars = [k for k in preds.variables.keys() if k not in ["Times", "Time"]]
    encoding = {var: {"zlib":True, "complevel":4, "least_significant_digit":2} for var in encoding_vars}
    encoding["ML_PREDICTED_TOR"]['least_significant_digit'] = 3
    preds.to_netcdf(savepath, encoding=encoding)
    
    return savepath
    
if __name__ == '__main__':
    
    args = parse_args()
    
    ncpus = int(os.getenv("SLURM_CPUS_PER_TASK"))
    dates = sorted([i[-8:] for i in glob.glob(args.preds_dir+f'{args.date}', recursive=True)])
    print(dates)
    
    for date in dates:
    
        init_times = sorted([i[-4:] for i in glob.glob(args.preds_dir+date+'/****', recursive=True)])
        wofs_init_times = sorted([i[-12:] for i in glob.glob(args.init_dir+date+'/*', recursive=True)])
    
        for init_time in init_times:
            if init_time.startswith('0'):
                wofs_init_time = list(filter(lambda x: x.endswith(init_time), wofs_init_times))[0]
            else:
                wofs_init_time = date+init_time
            
            print(date, init_time, wofs_init_time)
            for ens in range(1, 19):
                filepath_wofs = args.init_dir + f"{date}/{wofs_init_time}/ENS_MEM_{ens}/wrfinput_d01_predictions.nc"
                wofs = xr.open_dataset(filepath_wofs, engine=engine, decode_times=False, decode_coords=True)
        
                files = sorted(glob.glob(args.preds_dir+f"{date}/{init_time}/ENS_MEM_{ens}/**.nc", recursive=True))        
                with Pool(ncpus) as p:
                    try:
                        result = p.starmap(process_one_file,
                                           [(f, wofs) for f in files])
                        print(result)

                    except Exception as e:
                        print(traceback.format_exc())
