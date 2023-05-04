import xarray as xr
import numpy as np
from scipy.sparse import csr_matrix
import msgpack
import os
import sys
import glob
import argparse
from azure.storage.queue import QueueClient, TextBase64EncodePolicy, TextBase64DecodePolicy


def preds_to_msgpk(path_preds_timestep, path_preds_timestep_msgpk, timestep, args):

    def get_sparse_dict(ds):
        """ Convert variable from xarray dataset to a compressed sparse row matrix using a specified threshold."""

        x = ds[args.variable].values.reshape(300, 300)
        data = np.where(x <= args.threshold, 0, x)
        sparse_data = csr_matrix(data)
        rows, columns = sparse_data.nonzero()
        probs = data[rows, columns].astype('float16').tolist()

        return dict(rows=rows.astype('uint16').tolist(), columns=columns.astype('uint16').tolist(), values=probs)
    
    os.makedirs(path_preds_timestep_msgpk, exist_ok=True)
    
    files = sorted(glob.glob(path_preds_timestep))    
    data = {}
    for file in files:
        ds = xr.open_dataset(file)
        sparse_dict = get_sparse_dict(ds, args.variable, args.threshold)
        mem = file.split('ENS_MEM_')[1].split('/')[0]
        data[f"MEM_{mem}"] = sparse_dict
    se_coords = [ds['XLONG'][0, 0, 0].values.tolist(), ds['XLAT'][0, 0, 0].values.tolist()]
    data['se_coords'] = se_coords
    with open(os.path.join(path_preds_timestep_msgpk, f"wofs_sparse_prob_{timestep}.msgpk"), 'wb') as outfile:
        packed = msgpack.packb(data)
        outfile.write(packed)
        print(f"___Saving___ {outfile}")
        del packed
