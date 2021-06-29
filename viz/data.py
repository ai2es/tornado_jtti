"""Data provider for HWT_MODE application."""

import numpy as np
import pandas as pd
import datetime as dt
from bokeh.models import ColumnDataSource, CDSView, BooleanFilter
from bokeh.palettes import Set1 as palette
from glob import glob
from os.path import exists, join
import pickle5 as pickle
# import dask.dataframe as dd

import config as cfg


models = {"GMM": "mod_0_labels_*.pkl",
          "CNN": "cnn_test_000_labels_*.pkl"}

class HRRRProvider(object):
    def __init__(self, source, cols):
        self.source = source
        self.cols = cols

        # Preparing containers
        self.data = pd.DataFrame(columns=self.cols+["colors"]+["model"])
        self.data_ds = ColumnDataSource(data={cl: [] for cl in self.cols+["colors"]+["model"]})
        self.data_view = CDSView(filters=[], source=self.data_ds)

        # Calculating start time for inital data fetch
        self.fetch_data()

    def fetch_data(self):
        
#         data = dd.read_csv(self.source + 'mod_0_labels_*.pkl')

        # Loading data
        data = []
        for model in ["GMM", "CNN"]:
            filenames = glob(self.source + models[model])
            data_m = []
            for f in filenames:
                data_f = pickle.load(open(f, "rb"))
                data_m.append(data_f)
            data_m = pd.concat(data_m)
            data_m = data_m[self.cols]
            data_m["model"] = model
            data.append(data_m)
        data = pd.concat(data)
        data = data.set_index(pd.Index(range(data.shape[0])))
        
        data['run_date'] = pd.to_datetime(data['run_date'])
        colors = {0: palette[3][1], 1: palette[3][0], 2: palette[3][2]}
        data["colors"] = data['label_int'].map(colors)
        data.sort_values(by=['model', 'run_date', 'forecast_hour'])
        
        if data.run_date.size > 0:
            
            # Saving data to internal containers
            self.data = data
            self.data_ds.stream(self.data.to_dict(orient="list"))
    
            # initialize select variables for data_view filtering
            self.model = "CNN"
            self.update_model_filter()
            self.run_date_menu = self.get_run_date_menu()
            self.forecast_hour = 1
#             self.valid_hour = self.run_date + np.timedelta64(self.forecast_hour,'h')
            
        else:
            self.data_ds.stream({cl: [] for cl in self.cols+["colors"]})
            
        # Calculating filters
        model_filter = self.update_model_filter()
        run_date_filter = self.update_run_date_filter()
        forecast_hour_filter = self.update_forecast_hour_filter()
        
        return self.data
    
    def set_model(self, model):        
        self.model = model
        model_filter = self.update_model_filter()
        
    def update_model_filter(self):
        models = self.data["model"]
        model_filter = (models == self.model)
        self.data_view.filters = [BooleanFilter(model_filter)]
        return model_filter
    
    def get_run_date_menu(self):
        menu = np.unique(self.data.loc[self.data['model'] == self.model, "run_date"].values)
        self.min_run_date = menu[0]
        self.max_run_date = menu[-1]
        self.run_date = menu[-1]
        menu = [pd.to_datetime(dt).strftime("%-m/%-d/%Y - %HZ") for dt in menu]
        return menu

    def set_run_date(self, run_date):
#         run_date = pd.to_datetime(np.datetime64(run_date))
        self.run_date = np.clip(run_date, self.min_run_date, self.max_run_date)
        run_date_filter = self.update_run_date_filter()
        
    def update_run_date_filter(self):
        idx_true = self.data.index[(self.data['model'] == "GMM") &
                                   (self.data['run_date'] == self.run_date)]
        run_date_filter = np.zeros(self.data.shape[0], dtype=bool)
        run_date_filter[idx_true] = 1
        self.data_view.filters = [BooleanFilter(run_date_filter)]
        return run_date_filter
        
    def set_forecast_hour(self, forecast_hour):        
        self.forecast_hour = np.clip(forecast_hour, 1, 18)
        forecast_hour_filter = self.update_forecast_hour_filter()
    
    def update_forecast_hour_filter(self):        
        idx_true = self.data.index[(self.data['model'] == self.model) &
                                   (self.data['run_date'] == self.run_date) &
                                   (self.data['forecast_hour'] == self.forecast_hour)]
        forecast_hour_filter = np.zeros(self.data.shape[0], dtype=bool)
        forecast_hour_filter[idx_true] = 1
        self.data_view.filters = [BooleanFilter(forecast_hour_filter)]
        return forecast_hour_filter

    def set_valid_hour(self):        
        self.valid_hour = self.run_date + np.timedelta64(self.forecast_hour,'h')

