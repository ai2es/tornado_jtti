import xarray as xr
import numpy as np
from scipy.sparse import csr_matrix
import os, glob, msgpack
import subprocess

def preds_to_msgpk(paths, args, engine='netcdf4',):

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
    
    path_save = paths[0].replace('wofs-preds-2023-update', 'wofs-preds-2023-update-msgpk/').rsplit('ENS_MEM_')[0]
    os.makedirs(path_save, exist_ok=True)
    datetime = paths[0].rsplit('_predictions.nc')[0].rsplit('/wrfwof_d01_')[1]
    datetime = datetime.replace('-', '').replace('_', '')
    
    for variable, threshold in zip (args.variables, args.thresholds):
        ds_list = []
        ds = xr.open_mfdataset(paths,
                               engine=engine,
                               concat_dim='Time',
                               combine='nested',
                               decode_times=False,
                               decode_coords=True)[variable]
        ds_mean = ds.mean(dim='Time').load()
        ds_median = ds.median(dim='Time').load()
        ds_max = ds.max(dim='Time').load()

        data = {}
        for f in paths:
            i = f.split("ENS_MEM_")[1].split("/")[0]
            ds = xr.open_dataset(f, engine=engine, decode_times=False, decode_coords=True)
            sparse_dict = get_sparse_dict(ds, threshold, variable)
            data[f"MEM_{i}"] = sparse_dict

        mean_sparse_dict = get_sparse_dict(ds_mean, threshold)
        max_sparse_dict = get_sparse_dict(ds_max, threshold)
        median_sparse_dict = get_sparse_dict(ds_median, threshold)
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
