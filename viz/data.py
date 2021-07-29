"""Data provider for HWT_MODE application."""

import numpy as np
import pandas as pd
import datetime as dt
from bokeh.models import ColumnDataSource, CDSView, BooleanFilter
from bokeh.palettes import Turbo256 as palette
import glob
import math
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
        self.run_date = False

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
        self.datetimes = datetimes
        
        if self.run_date == False:
            self.run_date_menu = self.get_run_date_menu()
        
        datetime_min = datetime(year=self.run_date.year, month=self.run_date.month, day=self.run_date.day, hour=12)
        filename_start = filenames[0].split("20")[0] + datetime_min.strftime("%Y%m%d") + "/scale_314159265m2/storm-tracking_segmotion_"

        filenames_24hr = [filename_start + (datetime_min + timedelta(minutes=5)*i).strftime("%Y-%m-%d-%H%M%S") + '.p' for i in range(24*60//5)]
        filenames_24hr = set(filenames).intersection(filenames_24hr)
        print("number of filenames", len(filenames_24hr))
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
        self.y_min = 10000
        self.y_max = 0
        self.x_min = 10000
        self.x_max = -10000
        for i in data.index:
            x, y = data.loc[i, 'polygon_object_latlng_deg'].exterior.coords.xy
            print("lon_min", (np.array(x) - 360).min())
            print("lon_max", (np.array(x) - 360).max())
            print("lat_min", np.array(y).min())
            print("lat_max", np.array(y).max())
            self.x_min = min(self.x_min, (np.array(x) - 360).min())
            self.x_max = max(self.x_max, (np.array(x) - 360).max())
            self.y_min = min(self.y_min, np.array(y).min())
            self.y_max = max(self.y_max, np.array(y).max())
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
            self.set_zoom()
            self.valid_time_menu = self.get_valid_time_menu()
            self.set_select_valid_time_values()
            self.update_valid_time_filter()

            
        else:
            self.data_ds.stream({cl: [] for cl in self.cols_view})
            
        # Calculating filters
        valid_time_filter = self.update_valid_time_filter()
        
        return self.data
    
    def set_zoom(self):
        self.x_min = self.lon_to_web_mercator(self.x_min)
        self.x_max = self.lon_to_web_mercator(self.x_max)   
        self.y_min = self.lat_to_web_mercator(self.y_min)
        self.y_max = self.lat_to_web_mercator(self.y_max)
#         self.x_max = math.ceil(xs.max() / factor) * factor
#         self.x_min = math.floor(xs.min() / factor) * factor
        
#         self.y_max = math.ceil(ys.max() / factor) * factor 
#         self.y_min = math.floor(ys.min() / factor) * factor
        
#         print("X MAX", xs.max(), self.x_max)
#         print("X MIN", xs.min(), self.x_min)
#         print("Y MAX", ys.max(), self.y_max)
#         print("Y MIN", ys.min(), self.y_min)     
        
    def get_run_date_menu(self):
        dates_12Z = []
        for d in self.datetimes:
            date = datetime(year=d.year, month=d.month, day=d.day, hour=12)
            if d.hour >= 12:
                dates_12Z.append(date)
            if d.hour < 12:
                dates_12Z.append(date - timedelta(days=1))
        menu = np.unique(dates_12Z)
        self.run_date = menu[-1]
        menu = [dt.strftime("%-m/%-d/%Y - %HZ") for dt in menu]
        return menu
    
    def set_run_date(self, run_date):
        self.run_date = datetime.strptime(run_date, "%m/%d/%Y - %HZ")
        self.fetch_data()
    
    def get_valid_time_menu(self):
        menu = np.unique(self.data["valid_time"].values)
        self.min_valid_time = menu[0]
        self.max_valid_time = menu[-1]
        self.valid_time = menu[0]
        menu = [pd.to_datetime(dt).strftime("%-m/%-d/%Y %-H:%M") for dt in menu]
        return menu
    
    def set_select_valid_time_values(self):
        
        start = pd.Timestamp(self.min_valid_time)
        self.start = datetime(start.year, start.month, start.day, start.hour, start.minute)
        
        end = pd.Timestamp(self.max_valid_time)
        self.end = datetime(end.year, end.month, end.day, end.hour, end.minute)
        
        value = pd.Timestamp(self.valid_time)
        self.value = datetime(value.year, value.month, value.day, value.hour, value.minute)

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
