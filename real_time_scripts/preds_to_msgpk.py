import xarray as xr
import numpy as np
from scipy.sparse import csr_matrix
import os, glob, msgpack


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
    
    path_save = paths[0].replace('/2023/', '-msgpk/').rsplit('ENS_MEM_')[0]
    path_save = path_save[:-6] + path_save[-5:]
    os.umask(0o002)
    os.makedirs(path_save, exist_ok=True)
    datetime = paths[0].rsplit('_predictions.nc')[0].rsplit('/wrfwof_d01_')[1]
    datetime = datetime.replace('-', '').replace('_', '')
    
    for variable, threshold in zip (args.variables, args.thresholds):
        ds_list = []
        for i in range(1, 19):
            files = sorted(glob.glob(paths[0].split('ENS_MEM')[0] + f'ENS_MEM_{i}/' + paths[0].rsplit('/')[-1]))
            ds_list.append(xr.open_mfdataset(files,
                                             concat_dim='Time',
                                             combine='nested',
                                             decode_times=False,
                                             decode_coords=False)[variable])
        ds_all = xr.concat(ds_list, dim='member')
        ds_mean = ds_all.mean(dim='member').load()
        ds_median = ds_all.median(dim='member').load()
        ds_max = ds_all.max(dim='member').load()

        files = sorted(glob.glob(paths[0]))
        for timestep, f in enumerate(files):
            data = {}
            for i in range(1, 19):
                mem_f = f.replace("MEM_1", f"MEM_{i}")
                ds = xr.open_dataset(mem_f, decode_times=False, decode_coords=False)
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
            with open(os.path.join(path_save, f"wofs_sparse_prob_{datetime}_{variable}.msgpk"), 'wb') as outfile:
                packed = msgpack.packb(data)
                outfile.write(packed)
                del packed
