import glob
import pandas as pd
import netCDF4
from pyproj import Proj
import xarray as xr
from netCDF4 import Dataset
import h5py
import numpy as np
import tqdm
import os
import argparse
import shutil

#gewitter tools to open the pickel files
from gewittergefahr.gg_io import storm_tracking_io
from gewittergefahr.gg_utils import linkage

def get_arguments():
    #Define the strings that explain what each input variable means
    INIT_TIME_HELP_STRING = 'Model-initialization time (format "yyyymmdd").  This script will create storm masks for all hours from 1200 UTC on this date to 12 UTC on the following date for all hours that have radar data.'
    STORM_MASK_DIR_HELP_STRING = 'The directory where the storm masks will be saved. This directory should start from the root directory \'/\'.'
    TRACKING_DIR_HELP_STRING = 'The directory where the final storm tracking files are stored. This directory should start from the root directory \'/\'.'
    STORM_LABEL_DIR_HELP_STRING = 'The directory where the labeled storm files are stored. This directory should start from the root directory \'/\'.'
    INPUT_RADAR_DIR_HELP_STRING = 'The directory where the gridrad files are stored. This directory should start from the root directory \'/\'.'
    LINKED_TORNADO_DIR_HELP_STRING = 'The directory where the linked tornado-storm files are stored. This directory should start from the root directory \'/\'.'


    #Tell python what arguments to expect from the .sh file
    INPUT_ARG_PARSER = argparse.ArgumentParser()
    INPUT_ARG_PARSER.add_argument(
        '--this_spc_date_string_array_val', type=int, required=True,
        help=INIT_TIME_HELP_STRING)
        
    INPUT_ARG_PARSER.add_argument(
        '--output_storm_mask_dir_name', type=str, required=True,
        help=STORM_MASK_DIR_HELP_STRING)
        
    INPUT_ARG_PARSER.add_argument(
        '--input_tracking_dir_name', type=str, required=True,
        help=TRACKING_DIR_HELP_STRING)
        
    INPUT_ARG_PARSER.add_argument(
        '--output_labeled_storm_dir_name', type=str, required=True,
        help=STORM_LABEL_DIR_HELP_STRING)
        
    INPUT_ARG_PARSER.add_argument(
        '--input_radar_dir_name', type=str, required=True,
        help=INPUT_RADAR_DIR_HELP_STRING)
        
    INPUT_ARG_PARSER.add_argument(
        '--input_linked_tornado_dir_name', type=str, required=True,
        help=LINKED_TORNADO_DIR_HELP_STRING)
        
        
    #Pull out all of the input strings from the .sh file
    args = INPUT_ARG_PARSER.parse_args()
    #Index primer indicates the day of data that we are looking at in this particuar run of this code
    global index_primer
    index_primer = getattr(args, 'this_spc_date_string_array_val')
    global storm_mask_output_path
    storm_mask_output_path = getattr(args, 'output_storm_mask_dir_name')
    global final_tracking_directory
    final_tracking_directory = getattr(args, 'input_tracking_dir_name')
    global labeled_storm_dataframe_path
    labeled_storm_dataframe_path = getattr(args, 'output_labeled_storm_dir_name')
    global input_radar_directory_path
    input_radar_directory_path = getattr(args, 'input_radar_dir_name')
    global linkage_path
    linkage_path = getattr(args, 'input_linked_tornado_dir_name')

def main():

    #get the inputs from the .sh file
    get_arguments()

    #find all the days that we have data for
    all_gridrad_dirs = glob.glob(final_tracking_directory + '/*/*')
    all_gridrad_dirs.sort()


    #find which day corresponds to index_primer
    this_gridrad_dir = all_gridrad_dirs[index_primer]

    #pull out the date in YYYYMMDD format
    YYYYMMDD = this_gridrad_dir[-8:]
    print(YYYYMMDD)
    print(this_gridrad_dir)


    #convert to dtime to use fance time string functions 
    this_dtime = pd.to_datetime(YYYYMMDD)

    # If this file has already been made, skip to the next step
    if(not os.path.exists(labeled_storm_dataframe_path + this_dtime.strftime('%Y/%Y%m%d/labeled_storms_%Y%m%d_') + '05_min_10_km.csv')):
        
        #Find the linked storm-tornado file
        linkage_filepath = linkage_path + this_dtime.strftime('%Y/storm_to_tornadoes_%Y%m%d.p')
        
        #Pull out the data
        this_storm_to_events_table,_,this_tornado_table = linkage.read_linkage_file(linkage_filepath)
        
        #add dtime to read the data
        this_tornado_table['dates'] = pd.to_datetime(this_tornado_table.unix_time_sec,unit='s')
        this_tornado_table = this_tornado_table.set_index('dates')
        this_storm_to_events_table['dates'] = pd.to_datetime(this_storm_to_events_table.valid_time_unix_sec,unit='s')
        this_storm_to_events_table['date2'] = pd.to_datetime(this_storm_to_events_table.valid_time_unix_sec,unit='s')
        this_storm_to_events_table=this_storm_to_events_table.set_index('dates')

        #expand out the table so each tornado 'link' has a spot in the table 

        ##pre-allocate empty arrays because i dont know how big these will be beforehand
        #tor_array is the tornado id strings
        tor_array = np.array([])
        #time_diff is the relative time between the tor and storm 
        time_diff_array = np.array([])
        #distance_array is the distance between the tor and storm 
        distance_array = np.array([])
        #full_id_array is the full id string 
        full_id_array = np.array([])
        #date_array is the datetime associated with the storm 
        date_array = np.array([])
        #rating_array is the F or EF rating for each tor
        rating_array = np.array([])
        #this is a counter of how many entries are in the linked events
        l = np.array([])
        #this is a dummy index array to keep along for the ride 
        dummy_keeper = np.array([],dtype='int')


        #Loop through the storms in the linked file
        for i in tqdm.tqdm(np.arange(len(this_storm_to_events_table.index))):

            var_tmp = this_storm_to_events_table.iloc[i].event_latitudes_deg
            l_var_tmp = len(var_tmp)
            #store length
            l = np.append(l,l_var_tmp)
            
            #check if entry is empty
            if l_var_tmp > 0:
                #if it has data, expand it out 
                ts = this_storm_to_events_table.iloc[i].relative_event_times_sec
                ds = this_storm_to_events_table.iloc[i].linkage_distances_metres
                torids = this_storm_to_events_table.iloc[i].tornado_id_strings
                rating = this_storm_to_events_table.iloc[i].f_or_ef_scale_ratings
                
                #tile id so we can keep track of ids with each tor entry 
                ID = this_storm_to_events_table.iloc[i].full_id_string
                ID = np.tile(ID,(l_var_tmp))
                
                #same with datetimes 
                dt = this_storm_to_events_table.iloc[i].date2
                dt = np.tile(dt,(l_var_tmp))
                
                #same with dummy index 
                ii = np.tile(i,(l_var_tmp))
                
                #store it away 
                tor_array = np.append(tor_array,torids)
                time_diff_array =  np.append(time_diff_array,ts)
                rating_array = np.append(rating_array, rating)
                distance_array = np.append(distance_array,ds)
                full_id_array = np.append(full_id_array,ID)
                date_array = np.append(date_array,dt)
                dummy_keeper = np.append(dummy_keeper,ii)
            else:
                #if it doesnt have data, fill the spot with nans
                ID = this_storm_to_events_table.iloc[i].full_id_string
                dt = this_storm_to_events_table.iloc[i].date2
                tor_array = np.append(tor_array,np.nan)
                time_diff_array =  np.append(time_diff_array,np.nan)
                rating_array = np.append(rating_array, np.nan)
                distance_array = np.append(distance_array,np.nan)
                full_id_array = np.append(full_id_array,ID)
                date_array = np.append(date_array,dt)
                dummy_keeper = np.append(dummy_keeper,i)

        #fill new dataframe
        df_orig = pd.DataFrame({'full_id_string':full_id_array,'tor_id_string':tor_array,'f_or_ef_rating':rating_array,'time_diff':time_diff_array,'distance':distance_array,'date':date_array,'original_idx':dummy_keeper}).dropna(how='any')
        df_orig = df_orig.set_index('date')
        
        
        
        #check to see if the directory path is set up. If not make the proper directories
        if(not os.path.exists(labeled_storm_dataframe_path + this_dtime.strftime('%Y'))):
            os.mkdir(labeled_storm_dataframe_path + this_dtime.strftime('%Y'))
            
        if(not os.path.exists(labeled_storm_dataframe_path + this_dtime.strftime('%Y/%Y%m%d/'))):
            os.mkdir(labeled_storm_dataframe_path + this_dtime.strftime('%Y/%Y%m%d/'))
        

        # We only want tors within the next 5min and within 10 km of the storm 
        df_5 = df_orig.where((df_orig.time_diff <= 300) & (df_orig.distance <= 10000)).dropna(how='all')
        #combine two of the columns to drop duplicates 
        df_5['comb_str'] = df_5.index.strftime("%Y-%m-%d-%H%M%S")+df_5.full_id_string
        df_5 = df_5.drop_duplicates('comb_str')
        #drop that col. we dont need it anymore 
        df_5 = df_5.drop(columns='comb_str')
        df_5 = df_5.sort_index()
        #save it so you dont have to re-process 
        df_5.to_csv(labeled_storm_dataframe_path + this_dtime.strftime('%Y/%Y%m%d/labeled_storms_%Y%m%d_') + '05_min_10_km.csv')


    #all the radar data 
    rad_dir = this_dtime.strftime(input_radar_directory_path + '/%Y/%Y%m%d/')
    rad_files = glob.glob(rad_dir + '*')
    rad_files.sort()

    for i in range(len(rad_files)):

        #Pull the specific radar file
        idx = i
        rad_file = rad_files[idx]
        ds = xr.open_dataset(rad_file)

        #Get the time of this radar file
        radar_time = pd.to_datetime(np.asarray(netCDF4.num2date(ds.time.values[0],'seconds since 2001-01-01 00:00:00'),dtype='str'))
        
        #load the corresponding dataframes
        df_labeled  = pd.read_csv(labeled_storm_dataframe_path + this_dtime.strftime('%Y/%Y%m%d/labeled_storms_%Y%m%d_') + '05_min_10_km.csv')
        df_labeled  = df_labeled.set_index('date')
        
        #We need to collect the data and metadata - make empty lists for this data
        storm_mask_dss = []
        num_labeled_storms_list = []
        num_EF0_EF1_storms_list = []
        num_EF2_EF5_storms_list = []
        tornadic_pixels_EF0_EF1_list = []
        tornadic_pixels_EF2_EF5_list = []
        
        #reset all the metadata variables
        num_EF0_EF1_storms = 0
        num_EF2_EF5_storms = 0
        tornadic_pixels_EF0_EF1 = 0
        tornadic_pixels_EF2_EF5 = 0
        
        #Find how many total labeled storms occur at this time
        num_labeled_storms = len(pd.unique(df_labeled['full_id_string']))
    
        #extract all points with the current time 
        try:
            #see if there are any torandoes at this time - if not throw the exception
            df_labeled = df_labeled.loc[(radar_time).strftime('%Y-%m-%d %H:%M:%S')]
        except KeyError:
            print("No tornadoes at", (radar_time).strftime('%Y-%m-%d %H:%M:%S'))
            
            #make an empty storm mask for this time, since no tors
            storm_mask = np.zeros([ds.Latitude.shape[0],ds.Longitude.shape[0],])
            ds_mask = xr.Dataset({})
            ds_mask['storm_mask'] = ds.ZH[0,0].copy()
            ds_mask['storm_mask'].values = storm_mask
            storm_mask_dss.append(ds_mask)
            
            #these are all zeros if no tors
            num_labeled_storms_list.append(num_labeled_storms)
            num_EF0_EF1_storms_list.append(num_EF0_EF1_storms)
            num_EF2_EF5_storms_list.append(num_EF2_EF5_storms)
            tornadic_pixels_EF0_EF1_list.append(tornadic_pixels_EF0_EF1)
            tornadic_pixels_EF2_EF5_list.append(tornadic_pixels_EF2_EF5)
        else:
            #There is a tornado somewhere
    
            #point to the final_tracking dir 
            seg_dir = final_tracking_directory + this_dtime.strftime('%Y/%Y%m%d/')

            #load this tracking file 
            #grab the same time as the radar file 
            file_str = radar_time.strftime('scale_314159265m2/storm-tracking_segmotion_%Y-%m-%d-%H%M%S.p')
            #current tracking file path
            tracking_file = seg_dir + file_str

            #load the tracking dataframe
            df = storm_tracking_io.read_file(tracking_file)

            #make an all 0 matrix with the same shape as the original radar data
            storm_mask = np.zeros([ds.Latitude.shape[0],ds.Longitude.shape[0],])

            #loop over the storms in df_labeled 
            for i in tqdm.tqdm(np.arange(len(df_labeled))):
                if type(df_labeled)==pd.core.series.Series:
                    df_tmp = df_labeled
                else:
                    df_tmp = df_labeled.iloc[i]
                #drop all other storms 
                df_storm = df.where(df.full_id_string == df_tmp.full_id_string).dropna(how='all')
                df_storm = df_storm.iloc[0]

                #grab the rows and columns of the polygon
                r = ds.Latitude.shape[0]-df_storm.grid_point_rows -1 
                c = df_storm.grid_point_columns
                
                #calculate the metadata for each tornado - first is it a sig or non-sig tornado?
                if(df_tmp.f_or_ef_rating == 'EF0' or df_tmp.f_or_ef_rating == 'EF1'):
                    #increment the number of EF0s and EF1s
                    num_EF0_EF1_storms = num_EF0_EF1_storms + 1
                    #increment the number of EF0 and EF1 pixels
                    tornadic_pixels_EF0_EF1 = tornadic_pixels_EF0_EF1 + r.shape[0]
                    #label them 1 
                    storm_mask[r,c] = 1
                else:
                    #increment the number of EF2+ tornadoes
                    num_EF2_EF5_storms = num_EF2_EF5_storms + 1
                    #increment the number of EF2+ pixels
                    tornadic_pixels_EF2_EF5 = tornadic_pixels_EF2_EF5 + r.shape[0]
                    #label them 2 
                    storm_mask[r,c] = 2
            
            #Create a xarray dataset with the same metadata as the gridrad data so you can easily concat them
            ds_mask = xr.Dataset({})
            ds_mask['storm_mask'] = ds.ZH[0,0].copy()
            ds_mask['storm_mask'].values = storm_mask
            storm_mask_dss.append(ds_mask)
            
            #add the metadata
            num_labeled_storms_list.append(num_labeled_storms)
            num_EF0_EF1_storms_list.append(num_EF0_EF1_storms)
            num_EF2_EF5_storms_list.append(num_EF2_EF5_storms)
            tornadic_pixels_EF0_EF1_list.append(tornadic_pixels_EF0_EF1)
            tornadic_pixels_EF2_EF5_list.append(tornadic_pixels_EF2_EF5)

        #Concatenate the datasets with varied forecast windows
        combined_masks = xr.concat(storm_mask_dss, 'forecast_window')
        combined_masks = combined_masks.assign_coords(forecast_window= [5])#[60,45,30,15,0])

        #Add the metadata to the new dataset
        combined_masks['num_labeled_storms'] = xr.DataArray(num_labeled_storms_list, dims=dict(forecast_window=[5]))#[60,45,30,15,0]))
        combined_masks['tornadic_pixels_EF0_1'] = xr.DataArray(tornadic_pixels_EF0_EF1_list, dims=dict(forecast_window=[5]))#[60,45,30,15,0]))
        combined_masks['tornadic_pixels_EF2_5'] = xr.DataArray(tornadic_pixels_EF2_EF5_list, dims=dict(forecast_window=[5]))#[60,45,30,15,0]))
        combined_masks['num_EF0_1_storms'] = xr.DataArray(num_EF0_EF1_storms_list, dims=dict(forecast_window=[5]))#[60,45,30,15,0]))
        combined_masks['num_EF2_5_storms'] = xr.DataArray(num_EF2_EF5_storms_list, dims=dict(forecast_window=[5]))#[60,45,30,15,0]))
        
        #save it as a netCDF
        
        #check to see if the directory path is set up. If not make the proper directories
        if(not os.path.exists(storm_mask_output_path + this_dtime.strftime('%Y'))):
            os.mkdir(storm_mask_output_path + this_dtime.strftime('%Y'))
        if(not os.path.exists(storm_mask_output_path + this_dtime.strftime('%Y/%Y%m%d/'))):
            os.mkdir(storm_mask_output_path + this_dtime.strftime('%Y/%Y%m%d/'))
        if(not os.path.exists(storm_mask_output_path + this_dtime.strftime('%Y/%Y%m%d/') + '/5_min_window')):
            os.mkdir(storm_mask_output_path + this_dtime.strftime('%Y/%Y%m%d/') + '/5_min_window/')
        
        # Save out the storm masks
        combined_masks.to_netcdf(storm_mask_output_path + this_dtime.strftime('%Y/%Y%m%d/') + '/5_min_window/' + radar_time.strftime('storm_mask_%Y-%m-%d-%H%M%S.nc'))



if __name__ == "__main__":
    main()


