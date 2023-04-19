import xarray as xr
import numpy as np
from scipy.sparse import csr_matrix
import msgpack
import os
import matplotlib.pyplot as plt
import pyproj
import glob
import json
import pandas as pd
import argparse


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Path to config file")
    args = parser.parse_args()
    with open(args.config) as config_file:
        config = yaml.safe_load(config_file)

    def get_sparse_dict(ds, variable, thresh):
        """ Convert variable from xarray dataset to a compressed sparse row matrix using a specified threshold."""

        x = ds[variable].values.reshape(300, 300)
        data = np.where(x <= thresh, 0, x)
        sparse_data = csr_matrix(data)
        rows, columns = sparse_data.nonzero()
        probs = data[rows, columns].astype('float16').tolist()

        return dict(rows=rows.astype('uint16').tolist(), columns=columns.astype('uint16').tolist(), values=probs)

    in_dir = config["wofs_dir"]
    out_dir = config["out_dir"]
    variable = config["variable"]
    threshold = config["threshold"]

    ds_list = []
    for i in range(1, 19):
        files = sorted(glob.glob(os.path.join(in_dir, f"ENS_MEM_{i}", "wrfwof_d01_*")))
        ds_list.append(xr.open_mfdataset(files, concat_dim='Time', combine='nested')[variable])
    ds_all = xr.concat(ds_list, dim='member')
    ds_mean = ds_all.mean(dim='member')[variable].load()
    ds_median = ds_all.median(dim='member')[variable].load()
    ds_max = ds_all.max(dim='member')[variable].load()

    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(in_dir, f"ENS_MEM_1", "wrfwof_d01_*")))
    for timestep, f in enumerate(files):
        data = {}
        datetime = f[-19:-2].replace("-", "").replace(":", "").replace("_", "")
        for i in range(1, 19):
            mem_f = f.replace("MEM_1", f"MEM_{i}")
            ds = xr.open_dataset(mem_f)
            sparse_dict = get_sparse_dict(ds, variable, threshold)
            data[f"MEM_{i}"] = sparse_dict

        mean_sparse_dict = get_sparse_dict(ds_mean.isel(Time=timestep), variable, threshold)
        max_sparse_dict = get_sparse_dict(ds_max.isel(Time=timestep), variable, threshold)
        median_sparse_dict = get_sparse_dict(ds_median.isel(Time=timestep), variable, threshold)
        data["MEM_mean"] = mean_sparse_dict
        data["MEM_max"] = max_sparse_dict
        data["MEM_median"] = median_sparse_dict
        se_coords = [ds['XLONG'][0, 0, 0].values.tolist(), ds['XLAT'][0, 0, 0].values.tolist()]
        data['se_coords'] = se_coords
        with open(os.path.join(out_dir, f"wofs_sparse_prob_{datetime}.msgpk"), 'wb') as outfile:
            packed = msgpack.packb(data)
            outfile.write(packed)
            del packed

if __name__ == "__main__":
    main()
