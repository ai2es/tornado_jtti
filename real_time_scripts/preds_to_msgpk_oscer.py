import xarray as xr
import numpy as np
from scipy.sparse import csr_matrix
import os, glob, msgpack
import subprocess


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--root_path', type=str, required=True,
                        help='Root path to save data')
    parser.add_argument('--dir_preds_msgpk', type=str, required=True,
        help='Directory to store the machine learning predictions in MessagePack format. The files are saved of the form: <wofs_sparse_prob_<DATETIME>.msgpk')
    parser.add_argument('-v', '--variables', nargs='+', type=str,
        help='List of string variables to save out from predictions. Ex use: --variables ML_PREDICTED_TOR REFL_10CM')
    parser.add_argument('-t', '--thresholds', nargs='+', type=float,
        help='Space delimited list of float thresholds. Ex use: --thresholds 0.08 0.07')
    
    args = parser.parse_args()
    return args

def preds_to_msgpk(paths, args):

    def get_sparse_dict(ds, thresh, variable=None):
        """ Convert variable from xarray dataset to a compressed sparse row matrix using a specified threshold."""

        if variable:
            ds = ds[variable]

        if len(ds.values.shape) == 2 or len(ds.values.shape) == 3:
            x = ds.values.reshape(300, 300)
        elif len(ds.values.shape) == 4:
            x = ds.values.reshape(ds.values.shape[1], 300, 300)
        else:
            print("WARNING: unknown variable size", len(ds.values.shape), len(ds.values.shape))

        data = np.where(x <= thresh, 0, x)
        sparse_data = csr_matrix(data)
        rows, columns = sparse_data.nonzero()
        probs = data[rows, columns].astype('float16').tolist()

        return dict(rows=rows.astype('uint16').tolist(), columns=columns.astype('uint16').tolist(), values=probs)

    # path_save = paths[0].replace('-update/', '-update-msgpk/').rsplit('ENS_MEM_')[0]
    # path_save = path_save[:-6] + path_save[-5:]
    # os.umask(0o002)
    # os.makedirs(path_save, exist_ok=True)
    # datetime = paths[0].rsplit('_predictions.nc')[0].rsplit('/wrfwof_d01_')[1]
    # datetime = datetime.replace('-', '').replace('_', '')
    
    for variable, threshold in zip (args.variables, args.thresholds):
        ds_list = []
        dir = root_path+date+init_time
        for i in range(1, 19):
            files = sorted(glob.glob(dir+f'ENS_MEM_{i}/**'))
            ds_list.append(xr.open_mfdataset(files,
                                             concat_dim='Time',
                                             combine='nested',
                                             decode_times=False,
                                             decode_coords=True,
                                             engine='netcdf4')[variable])
        ds_all = xr.concat(ds_list, dim='member')
        ds_mean = ds_all.mean(dim='member').load()
        ds_median = ds_all.median(dim='member').load()
        ds_max = ds_all.max(dim='member').load()

        files = sorted(glob.glob(paths[0]))
        for timestep, f in enumerate(files):
            data = {}
            for i in range(1, 19)[:10]:
                mem_f = f.replace("MEM_1", f"MEM_{i}")
                ds = xr.open_dataset(mem_f, decode_times=False, decode_coords=True)
                sparse_dict = get_sparse_dict(ds, threshold, variable)
                data[f"MEM_{i}"] = sparse_dict

            mean_sparse_dict = get_sparse_dict(ds_mean.isel(Time=timestep), threshold)
            max_sparse_dict = get_sparse_dict(ds_max.isel(Time=timestep), threshold)
            median_sparse_dict = get_sparse_dict(ds_median.isel(Time=timestep), threshold)
            data["MEM_mean"] = mean_sparse_dict
            data["MEM_max"] = max_sparse_dict
            data["MEM_median"] = median_sparse_dict
            se_coords = [ds['XLONG'][0, 0, 0].values.tolist(), ds['XLAT'][0, 0, 0].values.tolist()]
            data['se_coords'] = se_coords
            out_file_path = os.path.join(path_save, f"wofs_sparse_prob_{datetime}_{variable}.msgpk")
            with open(out_file_path, 'wb') as outfile:
                packed = msgpack.packb(data)
                outfile.write(packed)
                del packed
            run_date = out_file_path.split("/")[-2]
            out_file_name = out_file_path.split("/")[-1]
            subprocess.run(["azcopy",
                            "copy",
                            out_file_path,
                            f"{args.root_path}/{args.dir_preds_msgpk}/{run_date}/{out_file_name}"])

if __name__ == '__main__':
    
    args = parse_args()
    

