import xarray as xr
import numpy as np
from scipy.sparse import csr_matrix
import msgpack
import os
import sys
import glob
import argparse


def main():

    parser = argparse.ArgumentParser(description='Save ML predictions to messagepack')
    parser.add_argument('--dir_preds', type=str, required=True, 
        help='Location of the WoFS prediction file(s). Can be a path to a single file or a directory to several files')
    parser.add_argument('--dir_preds_msgpk', type=str, required=True, 
        help='Directory to store the machine learning predictions in MessagePack format. The files are saved of the form: <wofs_sparse_prob_<DATETIME>.msgpk')
    parser.add_argument('--variable', type=str, required=True, 
        help='TODO')
    parser.add_argument('--threshold', type=float, required=True, 
        help='If probability of tornado is greater than or equal to this threshold value, build tornado tracks')
    args = parser.parse_args()

    def get_sparse_dict(ds):
        """ Convert variable from xarray dataset to a compressed sparse row matrix using a specified threshold."""

        x = ds[args.variable].values.reshape(300, 300)
        data = np.where(x <= args.threshold, 0, x)
        sparse_data = csr_matrix(data)
        rows, columns = sparse_data.nonzero()
        probs = data[rows, columns].astype('float16').tolist()

        return dict(rows=rows.astype('uint16').tolist(), columns=columns.astype('uint16').tolist(), values=probs)

    ds_list = []
    for i in range(1, 19):
        os.makedirs(os.path.join(args.dir_preds_msgpk, f"ENS_MEM_{i}"), exist_ok=True)
        files = sorted(glob.glob(os.path.join(args.dir_preds, f"ENS_MEM_{i}", "wrfwof_d01_*")))
        ds_list.append(xr.open_mfdataset(files, concat_dim='Time', combine='nested')[args.variable])
    ds_all = xr.concat(ds_list, dim='member')
    ds_mean = ds_all.mean(dim='member').load()
    ds_median = ds_all.median(dim='member').load()
    ds_max = ds_all.max(dim='member').load()

    os.makedirs(args.dir_preds_msgpck, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.dir_preds, f"ENS_MEM_1", "wrfwof_d01_*")))
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
        with open(os.path.join(args.dir_preds_msgpk, f"wofs_sparse_prob_{datetime}.msgpk"), 'wb') as outfile:
            packed = msgpack.packb(data)
            outfile.write(packed)
            print(f"Saving {outfile}")
            del packed

if __name__ == "__main__":
    main()
