import xarray as xr
import numpy as np
from scipy.sparse import csr_matrix
import msgpack
import os
import sys
import glob
import argparse
from azure.storage.queue import QueueClient, TextBase64EncodePolicy, TextBase64DecodePolicy


def preds_to_msgpk(path_preds, args):

    def get_sparse_dict(ds):
        """ Convert variable from xarray dataset to a compressed sparse row matrix using a specified threshold."""

        x = ds[args.variable].values.reshape(300, 300)
        data = np.where(x <= args.threshold, 0, x)
        sparse_data = csr_matrix(data)
        rows, columns = sparse_data.nonzero()
        probs = data[rows, columns].astype('float16').tolist()

        return dict(rows=rows.astype('uint16').tolist(), columns=columns.astype('uint16').tolist(), values=probs)
    
    path_preds_msgpk = path_preds.replace('wofs-preds', 'wofs-preds-msgpk')
    path_preds_msgpk = path_preds_msgpk[:-6] + path_preds_msgpk[-5:]
    os.makedirs(path_preds_msgpk, exist_ok=True)
    
    ds_list = []
    for i in range(1, 19):
        files = sorted(glob.glob(os.path.join(path_preds, f"ENS_MEM_{i}", "wrfwof_d01_*")))
        ds_list.append(xr.open_mfdataset(files, concat_dim='Time', combine='nested', decode_times=False)[args.variable])
    ds_all = xr.concat(ds_list, dim='member')
    ds_mean = ds_all.mean(dim='member').load()
    ds_median = ds_all.median(dim='member').load()
    ds_max = ds_all.max(dim='member').load()

    files = sorted(glob.glob(os.path.join(path_preds, f"ENS_MEM_1", "wrfwof_d01_*")))
    for timestep, f in enumerate(files):
        data = {}
        datetime = f[-19:-2].replace("-", "").replace(":", "").replace("_", "")
        for i in range(1, 19):
            mem_f = f.replace("MEM_1", f"MEM_{i}")
            ds = xr.open_dataset(mem_f)
            sparse_dict = get_sparse_dict(ds)
            data[f"MEM_{i}"] = sparse_dict

        mean_sparse_dict = get_sparse_dict(ds_mean.isel(Time=timestep), args.variable, args.threshold)
        max_sparse_dict = get_sparse_dict(ds_max.isel(Time=timestep), args.variable, args.threshold)
        median_sparse_dict = get_sparse_dict(ds_median.isel(Time=timestep), args.variable, args.threshold)
        data["MEM_mean"] = mean_sparse_dict
        data["MEM_max"] = max_sparse_dict
        data["MEM_median"] = median_sparse_dict
        se_coords = [ds['XLONG'][0, 0, 0].values.tolist(), ds['XLAT'][0, 0, 0].values.tolist()]
        data['se_coords'] = se_coords
        with open(os.path.join(path_preds_msgpk, f"wofs_sparse_prob_{datetime}.msgpk"), 'wb') as outfile:
            packed = msgpack.packb(data)
            outfile.write(packed)
            print(f"Saving {outfile}")
            del packed
