import time
import pandas as pd
import numpy as np
import xarray as xr
from os.path import exists, join
from os import listdir
from glob import glob
from datetime import datetime, timedelta
import pickle

def main():
    tor_path = "/glade/p/cisl/aiml/jtti_tornado/gridrad_storm_tracks/"
    all_dates = get_all_pickle_dates(tor_path)
    print(all_dates)
    for date in all_dates:
        storm_ds = load_patch_day(tor_path, date)
        if storm_ds is None:
            continue
        storm_poly = load_track_day(tor_path, date)
        poly_id = storm_poly["full_id_string"] + "_" + storm_poly["valid_time_unix_sec"].astype(str)
        ds_id = pd.Series(storm_ds["full_storm_id_strings"].values).str.decode("utf-8") + "_" + pd.Series(storm_ds["storm_times_unix_sec"].values).astype(str)
        overlap = np.intersect1d(poly_id, ds_id)
        print(date, poly_id.size, ds_id.size, overlap.size)
        storm_ds.close()
    return

def get_all_pickle_dates(tor_path):
    tor_years = sorted(listdir(tor_path))[:-1]
    print(f"Years in data: {tor_years}")
    all_dates = []
    for year in tor_years:
        all_dates.extend(sorted(listdir(join(tor_path, year))))
    return all_dates

def load_track_day(tor_path, date):
    object_files = sorted(glob(join(tor_path, date[:4], date, "scale_314159265m2", "*.p")))
    print(f"Number of files for {date}: {len(object_files)}")
    storm_frames = []
    for object_file in object_files:
        with open(object_file, "rb") as obj_file_r:
            df = pickle.load(obj_file_r)
            run_time = time.mktime(datetime.strptime(object_file.split('segmotion_')[1].split('.')[0], '%Y-%m-%d-%H%M%S').timetuple())
            df['run_time_unix_sec'] = [run_time] * df.shape[0]
            storm_frames.append(df)
    storms = pd.concat(storm_frames)
    storms = storms.sort_values(by=['run_time_unix_sec', 'valid_time_unix_sec'])
    storms = storms.reset_index(drop=True)
    return storms

def load_patch_day(tor_path, date, downsampled=False):
    if downsampled:
        down_dir = "downsampled_2012-2018"
    else:
        down_dir = "nondownsampled_2011_2015-2018"
    tor_file = join(tor_path, "tornado_occurrence", down_dir, "learning_examples", date[:4], f"input_examples_{date}.nc")
    tor_ds = None
    if exists(tor_file):
        tor_ds = xr.open_dataset(tor_file)
    return tor_ds

if __name__ == "__main__":
    main()
