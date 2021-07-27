"""Data provider for HWT_MODE application."""

import numpy as np
import pandas as pd
import datetime as dt
from bokeh.models import ColumnDataSource, CDSView, BooleanFilter
from bokeh.palettes import Turbo256 as palette
import glob
from os.path import exists, join
import pickle
from datetime import datetime, timedelta

# import dask.dataframe as dd

import config as cfg


class HRRRProvider(object):
    def __init__(self, source, cols, cols_view):
        self.source = source
        self.cols = cols
        self.cols_view = cols_view

        # Preparing containers
        self.data = pd.DataFrame(columns=self.cols_view)
        self.data_ds = ColumnDataSource(data={cl: [] for cl in self.cols_view})
        self.data_view = CDSView(filters=[], source=self.data_ds)

        # Calculating start time for inital data fetch
        self.fetch_data()

    def lon_to_web_mercator(self, lon):
        k = 6378137
        return lon * (k * np.pi / 180.0)

    def lat_to_web_mercator(self, lat):
        k = 6378137
        return np.log(np.tan((90 + lat) * np.pi / 360.0)) * k
    
    def fetch_data(self):
        
        # Loading data        
        filenames = [f for f in glob.glob(self.source + "**/*.p", recursive=True)]
        datetimes = [datetime.strptime(f.split('segmotion_')[1].split('.')[0], '%Y-%m-%d-%H%M%S') for f in filenames]
        datetime_min = datetime(year=datetimes[0].year, month=datetimes[0].month, day=datetimes[0].day, hour=12)
        filenames_24hr = [filenames[0].split('_2')[0] + "_" + (datetime_min + timedelta(minutes=5)*i).strftime("%Y-%m-%d-%H%M%S") + '.p' for i in range(24*60//5)]
        filenames_24hr = set(filenames).intersection(filenames_24hr)
        print(len(filenames_24hr))
        files = []
        for filename in filenames_24hr:
            with open(filename, 'rb') as f:
                df = pickle.load(f)
                run_time = datetime.strptime(filename.split('segmotion_')[1].split('.')[0], '%Y-%m-%d-%H%M%S')
                df['run_time'] = [run_time] * df.shape[0]
                files.append(df)
        data = pd.concat(files)
        data = data[self.cols]
        data = data.sort_values(by=['run_time', 'valid_time_unix_sec'])
        data = data.reset_index(drop=True)

        
        xs = []
        ys = []
        for i in data.index:
            x, y = data.loc[i, 'polygon_object_latlng_deg'].exterior.coords.xy
            xs.append(self.lon_to_web_mercator(np.array(x) - 360))
            ys.append(self.lat_to_web_mercator(np.array(y)))
        data['x'] = xs
        data['y'] = ys
        
        data['valid_time'] = pd.to_datetime(data['valid_time_unix_sec'], unit='s')
        colors = {k:v for k, v in list(zip(data['full_id_string'].unique().tolist(), list(palette)))}
        data["colors"] = data['full_id_string'].map(colors)
        data = data[self.cols_view]
        
        if data.valid_time.size > 0:
            
            # Saving data to internal containers
            self.data = data
            self.data_ds.stream(self.data.to_dict(orient="list"))
    
            # initialize select variables for data_view filtering
            self.valid_time_menu = self.get_valid_time_menu()
            self.update_valid_time_filter()
            
        else:
            self.data_ds.stream({cl: [] for cl in self.cols_view})
            
        # Calculating filters
        valid_time_filter = self.update_valid_time_filter()
        
        return self.data
    
    def get_valid_time_menu(self):
        menu = np.unique(self.data["valid_time"].values)
        self.min_valid_time = menu[0]
        self.max_valid_time = menu[-1]
        self.valid_time = menu[-1]
        menu = [pd.to_datetime(dt).strftime("%-m/%-d/%Y %H:%M") for dt in menu]
        return menu

    def set_valid_time(self, valid_time):
        valid_time = np.datetime64(pd.to_datetime(valid_time))
        self.valid_time = np.clip(valid_time, self.min_valid_time, self.max_valid_time)
        valid_time_filter = self.update_valid_time_filter()
        
    def update_valid_time_filter(self):
        idx_true = self.data.index[(self.data['valid_time'] == self.valid_time)]
        valid_time_filter = np.zeros(self.data.shape[0], dtype=bool)
        valid_time_filter[idx_true] = 1
        self.data_view.filters = [BooleanFilter(valid_time_filter)]
        return valid_time_filter
