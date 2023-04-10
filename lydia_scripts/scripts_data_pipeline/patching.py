import glob
import pandas as pd
print(f"pd {pd.__version__}")
import netCDF4
print(f"netCDF4 {netCDF4.__version__}")
import xarray as xr
print(f"xr {xr.__version__}")
import numpy as np
print(f"np {np.__version__}")
import os
import argparse
import random
from datetime import timedelta
from time import perf_counter
    
# This function will return a dataset of patches from the given data
# xs and ys only include the gridpoints that will give us a tornado in the patch if this is for tornadic data
# otherwise xs and ys are all the gridpoints
def make_training_patches(radar, storm_mask, xs, ys, n, time, lats, lons):

    patches = []

    # Make n patches from this time
    for k in range(n):
        # Select a random patch from the available gridpoints
        i = random.randint(0, xs.shape[0]-1)
        xi = xs[i]
        yi = ys[i]

        to_add = xr.Dataset(data_vars=dict(
                           ZH=(["x", "y", "z"], radar.ZH.isel(Latitude=slice(xi, xi+size), Longitude=slice(yi, yi+size), time=0).values.swapaxes(0,2).swapaxes(1,0)), 
                           DIV=(["x", "y", "z"], radar.DIV.isel(Latitude=slice(xi, xi+size), Longitude=slice(yi, yi+size), time=0).values.swapaxes(0,2).swapaxes(1,0)),
                           VOR=(["x", "y", "z"], radar.VOR.isel(Latitude=slice(xi, xi+size), Longitude=slice(yi, yi+size), time=0).values.swapaxes(0,2).swapaxes(1,0)),
                           labels=(["x", "y"], storm_mask[xi:xi+size, yi:yi+size]),
                           n_tornadic_pixels=([], np.count_nonzero(storm_mask[xi:xi+size, yi:yi+size])),
                           n_pixels_EF0_EF1 = ([], np.count_nonzero(storm_mask[xi:xi+size, yi:yi+size] == 1)),
                           n_pixels_EF2_EF5 = ([], np.count_nonzero(storm_mask[xi:xi+size, yi:yi+size] == 2)),
                           n_convective_pixels = ([], np.count_nonzero(radar.ZH.isel(Latitude=slice(xi, xi+size), Longitude=slice(yi, yi+size), time=0).fillna(0).values.swapaxes(0,2).swapaxes(1,0).max(axis=(2)) >= 30)),
                           lat=([],lats[xi]),
                           lon=([],lons[yi]),
                           time=([],time),
                           forecast_window=([],5)),
                    coords=dict(x=(["x"], np.arange(size)), y=(["y"], np.arange(size)), z=(["z"], np.arange(29))))
        to_add = to_add.fillna(0)
        

        to_add['n_convective_pixels'] = ([], np.count_nonzero(to_add.ZH.max(axis=(2), skipna=True) >= 30))
        patches.append(to_add)
        
    
    output = xr.concat(patches, 'patch')
    
    return output
    
    
def get_arguments():

    #Define the strings that explain what each input variable means
    INIT_TIME_HELP_STRING = 'i, where i indicates the ith file of the storm masks from which to make the patches.'
    PATCHES_DIR_HELP_STRING = 'The directory where the patches will be saved. This directory should start from the root directory \'/\'.'
    UNET_MASK_DIR_HELP_STRING = 'The directory where the unet mask files are stored. This directory should start from the root directory \'/\'.'
    INPUT_RADAR_DIR_HELP_STRING = 'The directory where the gridrad files are stored. This directory should start from the root directory \'/\'.'
    PATCH_SIZE_HELP_STRING = 'Patch size is the size of the patches in each horizontal dimension.'
    PATCHES_NUMBER_HELP_STRING = 'n_patches is the number of patches to make of each type (nontor, tor, sigtor) from each time'

    #Tell python what arguments to expect from the .sh file
    INPUT_ARG_PARSER = argparse.ArgumentParser()
    INPUT_ARG_PARSER.add_argument(
        '--this_spc_date_string_array_val', type=int, required=True,
        help=INIT_TIME_HELP_STRING)
        
    INPUT_ARG_PARSER.add_argument(
        '--input_radar_dir_name', type=str, required=True,
        help=INPUT_RADAR_DIR_HELP_STRING)
        
    INPUT_ARG_PARSER.add_argument(
        '--input_storm_mask_dir_name', type=str, required=True,
        help=UNET_MASK_DIR_HELP_STRING)
        
    INPUT_ARG_PARSER.add_argument(
        '--output_patch_dir_name', type=str, required=True,
        help=PATCHES_DIR_HELP_STRING)
        
    INPUT_ARG_PARSER.add_argument(
        '--patch_size', type=int, required=True,
        help=PATCH_SIZE_HELP_STRING)
        
    INPUT_ARG_PARSER.add_argument(
        '--n_patches', type=int, required=True,
        help=PATCHES_NUMBER_HELP_STRING)

    INPUT_ARG_PARSER.add_argument(
        '--dry_run', action='store_true',
        help="Boolean flag whether. Use to perform a dry run and only display file paths. No files are actually saved.")

    #Pull out all of the input strings from the .sh file
    args = INPUT_ARG_PARSER.parse_args()
    #Index primer indicates the day of data that we are looking at in this particuar run of this code
    global index_primer
    index_primer = getattr(args, 'this_spc_date_string_array_val')
    #Storm mask path is the path to the storm mask files
    global storm_mask_path
    storm_mask_path = getattr(args, 'input_storm_mask_dir_name')
    # Patches path is the path to output the final patches files to
    global patches_path
    patches_path = getattr(args, 'output_patch_dir_name')
    # Input_radar_directory is the location of the gridrad data to patch
    global input_radar_directory_path
    input_radar_directory_path = getattr(args, 'input_radar_dir_name')
    # patch_size is the size of the patch in each horizontal dimension
    global size
    size = getattr(args, 'patch_size')
    # n_patches is the number of patches of each type (nontor, tor, sigtor) to make from each time
    global n_patches
    n_patches = getattr(args, 'n_patches')

    global dry_run
    dry_run = args.dry_run

    return args


# Make the directory structure for how we will save out the patches
# Also calculate the datetime of this day and collect all the storm mask files to patch
def make_directory_structure(this_storm_mask_dir):

    #pull out the date in YYYYMMDD format
    YYYYMMDD_path = this_storm_mask_dir[-21:-13]
    YYYY = YYYYMMDD_path[:4]

    # If this file hasn't already been made, make the proper folders
    base_path = patches_path
    '''if(not os.path.exists(base_path)):
        print(f"Make directory {base_path} [dry_run={dry_run}]")
        if not dry_run: os.mkdir(base_path)'''

    #print(patches_path + 'size_' + str(size))
    base_path += f'size_{size}'
    '''if not os.path.exists(base_path):
        print(f"Make directory {base_path} [dry_run={dry_run}]")
        #if not dry_run: os.mkdir(patches_path + '/size_' + str(size))
        if not dry_run: os.mkdir(base_path)'''

    #print(patches_path + 'size_' + str(size) + '/forecast_window_5')
    base_path += '/forecast_window_5'
    '''if not os.path.exists(base_path):
        print(f"Make directory {base_path} [dry_run={dry_run}]")
        #if not dry_run: os.mkdir(patches_path + '/size_' + str(size) + '/forecast_window_5')
        if not dry_run: os.mkdir(base_path)'''
        
    #print(patches_path + 'size_' + str(size) + '/forecast_window_5' + '/' + YYYY)
    base_path += f"/{YYYY}"
    '''if not os.path.exists(base_path):
        print(f"Make directory {base_path} [dry_run={dry_run}]")
        #if not dry_run: os.mkdir(patches_path + '/size_' + str(size) + '/forecast_window_5' + '/' + YYYY)
        if not dry_run: os.mkdir(base_path)'''
    
    #print(patches_path + 'size_' + str(size) + '/forecast_window_5' + '/' + YYYY + '/' + YYYYMMDD_path)
    base_path += f"/{YYYYMMDD_path}"
    '''if not os.path.exists(base_path):
        print(f"Make directory {base_path} [dry_run={dry_run}]")
        #if not dry_run: os.mkdir(patches_path + '/size_' + str(size) + '/forecast_window_5' + '/' + YYYY + '/' + YYYYMMDD_path)
        if not dry_run: os.mkdir(base_path)'''

    try:
        data_dirpath = os.path.join(patches_path, f'size_{size}', 'forecast_window_5', YYYY, YYYYMMDD_path)
        print(f"oMake directory {base_path} [dry_run={dry_run}]")
        print(f"nMake directory {data_dirpath} [dry_run={dry_run}]")
        if not dry_run: 
            os.makedirs()
    except OSError as oserr:
        print(oserr)
          


def main():

    #get the inputs from the .sh file
    args = get_arguments()

    #find all the days that we have data for
    storm_dirpath = storm_mask_path + '*/*/5_min_window/'
    print(f"1storm_dirpath {storm_dirpath}")
    print("old", storm_mask_path + '*/*/5_min_window/')
    all_storm_mask_dirs = glob.glob(storm_dirpath)
    all_storm_mask_dirs.sort()
    
    #find which day corresponds to index_primer
    this_storm_mask_dir = all_storm_mask_dirs[int(index_primer)]

    #pull out the date in YYYYMMDD format
    YYYYMMDD_path = this_storm_mask_dir[-21:-13]
    YYYY = YYYYMMDD_path[:4]

    # Find all the files to patch from this day
    storm_dirpath = os.path.join(storm_mask_path, YYYY, YYYYMMDD_path, '5_min_window/*')
    print(f"2storm_dirpath {storm_dirpath}")
    print("old", storm_mask_path + '/' + YYYY + '/' + YYYYMMDD_path + '/5_min_window/*')
    #all_masks_day = glob.glob(storm_mask_path + '/' + YYYY + '/' + YYYYMMDD_path + '/5_min_window/*')
    all_masks_day = glob.glob(storm_dirpath)[:2]
    all_masks_day.sort()
    print("all_masks_day", all_masks_day)

    #make the directory structure for the output files
    make_directory_structure(this_storm_mask_dir)
    
       
    # Initialize empty lists for the patches we will make
    # No tornado
    output_nt_list = []
    # Tornado
    output_t_list = []
    # Significant Tornado
    output_st_list = []
    
    # We want to output the patches by the hour, so '-1' flags that this is the start of the script
    last_hr = '-1'

    # Loop through each time and make patches
    for mask_time in all_masks_day:
        print(f">> {mask_time}")
        # Pull out the times from the file
        # Sometimes the YYYYMMDD in the filepath doesn't match the the YYYYMMDD in the filename
        # They are labeled YYYYMMDD_path and YYYYMMDD_day respectively
        #YYYYMMDD_day_with_dashes is formatted 'YYYY-MM-DD'
        YYYYMMDD_day_with_dashes = mask_time[-20:-10]
        HHMMSS = mask_time[-9:-3]
        HH = HHMMSS[:2]
        YYYYMMDD_day = YYYYMMDD_day_with_dashes[:4] + YYYYMMDD_day_with_dashes[5:7] + YYYYMMDD_day_with_dashes[8:]
        
        # Check to see if we have reached the end of the hour. If we have, output the data
        if(last_hr != HH and last_hr != '-1' and len(output_nt_list) != 0):
            print('Outputing with', len(output_nt_list), 'inputs')
            output_nt = xr.concat(output_nt_list, dim='patch')
            patches_fname = patches_path + '/size_' + str(size) + '/forecast_window_' + str(window) + '/' + YYYY + '/' + YYYYMMDD_path + '/patches_nontor_%s_%s.nc' % (YYYYMMDD_day, last_hr)
            print(f"Saving L0 {patches_fname} [dry_run={dry_run}]")
            if not dry_run: 
                output_nt.to_netcdf(patches_fname)
                output_nt.close()
            output_nt_list = []
            
            if(output_t_list != []):
                output_t = xr.concat(output_t_list, dim='patch')
                patches_fname = patches_path + '/size_' + str(size) + '/forecast_window_' + str(window) + '/' + YYYY + '/' + YYYYMMDD_path + '/patches_tor_%s_%s.nc' % (YYYYMMDD_day, last_hr)
                print(f"Saving L1 {patches_fname} [dry_run={dry_run}]")
                if not dry_run: 
                    output_t.to_netcdf(patches_fname)
                    output_t.close()
                output_t_list = []
                
                if(output_st_list != []):
                    output_st = xr.concat(output_st_list, dim='patch')
                    patches_fname = patches_path + '/size_' + str(size) + '/forecast_window_' + str(window) + '/' + YYYY + '/' + YYYYMMDD_path + '/patches_sigtor_%s_%s.nc' % (YYYYMMDD_day, last_hr)
                    print(f"Saving L2 {patches_fname} [dry_run={dry_run}]")
                    if not dry_run: 
                        output_st.to_netcdf(patches_fname)
                        output_st.close()
                    output_st_list = []
        
        # Update last_hr to be the current hour now that we are past the checkpoint
        last_hr = HH
        # If we have already made files for this day, skip
        if os.path.exists(patches_path + '/size_' + str(size) + '/forecast_window_' + str(window) + '/' + YYYY + '/' + YYYYMMDD_path + '/patches_nontor_%s_%s.nc' % (YYYYMMDD_day, HH)):
            print("Skipping")
            continue
        
        
        # Now we can finally make the patches       
        print('Making Patch', HHMMSS)
        
        # We only want storms matched in the most immediate time window to a tornado---within 5 minutes before or after
        window_idx = 0
        window = 5
        
        #Read in the storm mask
        storm_mask_and_metadata = xr.open_dataset(storm_mask_path + '/%s/%s/5_min_window/storm_mask_%s-%s.nc' % (YYYY, YYYYMMDD_path, YYYYMMDD_day_with_dashes, HHMMSS))
        # Open the corresponding radar file
        radar = xr.open_dataset(input_radar_directory_path + '/%s/%s/nexrad_3d_v4_2_%sT%sZ.nc' % (YYYY, YYYYMMDD_path, YYYYMMDD_day, HHMMSS))

        storm_mask = storm_mask_and_metadata.storm_mask[window_idx].values
        time = storm_mask_and_metadata.time.values
        lons = storm_mask_and_metadata.Longitude.values
        lats = storm_mask_and_metadata.Latitude.values
        
        #make non-tornadic patches
        possible_x = np.tile(np.arange(storm_mask.shape[0] - size), storm_mask.shape[1] - size).reshape(storm_mask.shape[1] - size, storm_mask.shape[0] - size).ravel()
        possible_y = np.tile(np.arange(storm_mask.shape[1] - size), storm_mask.shape[0] - size).reshape(storm_mask.shape[0] - size, storm_mask.shape[1] - size).swapaxes(0,1).ravel()

        # For the nontor dataset, we sample patches from anywhere whether or not there is a tornado
        output_nt_list.append(make_training_patches(radar, storm_mask, possible_x, possible_y, n_patches, time, lats, lons))
        

        #make tornadic patches if there are tornadoes present
        if(np.count_nonzero(storm_mask) > 0):
            
            #pull out the x,y indices of all tornadic images
            #make a list/array of all possible starting indices (top left corner of the patch) (+[0,size] in either direction of the tornadic pixels)
            possible_starting_indices = np.zeros(storm_mask[:-(size),:-(size)].shape)
            
            #Reduce the array size by 'size' in both dimensions
            possible_x_max, possible_y_max = (np.array(possible_starting_indices.shape) - np.ones(2)).astype(int)
            
            #pull out the indices of all torandic pixels
            tornado_x, tornado_y = np.nonzero(storm_mask)
            
            #find all the pixels such that if this pixel was in the corner of the patch, the patch would contain a tornadic pixel
            for x, y in zip(tornado_x, tornado_y):
                possible_starting_indices[max(0, x-size+1) : min(x, possible_x_max), max(0, y-size+1) : min(y, possible_y_max)] = 1
                
            #find all the indices where this is the case
            possible_x, possible_y = np.nonzero(possible_starting_indices)

            # Make the patches that include torandoes
            output_t_list.append(make_training_patches(radar, storm_mask, possible_x, possible_y, n_patches, time, lats, lons))
            
            
            #make significant tornado patches if there are sigtors present
            if(np.count_nonzero(storm_mask == 2) > 0):
                
                #find all the pixels such that if this pixel was in the corner of the patch, the patch would contain a sigtor pixel
                possible_starting_indices_sigtor = np.zeros(storm_mask[:-(size),:-(size)].shape)
                
                #Reduce the array size by 'size' in both dimensions
                possible_x_max, possible_y_max = (np.array(possible_starting_indices_sigtor.shape) - np.ones(2)).astype(int)
                
                #pull out the indices of all significant tornado pixels
                sigtornado_x, sigtornado_y = np.nonzero(storm_mask == 2)
                
                #find all the pixels such that if this pixel was in the corner of the patch, the patch would contain a sigtor pixel
                for x, y in zip(sigtornado_x, sigtornado_y):
                    possible_starting_indices_sigtor[max(0, x-size+1) : min(x, possible_x_max), max(0, y-size+1) : min(y, possible_y_max)] = 1

                #find all the indices where this is the case
                possible_x, possible_y = np.nonzero(possible_starting_indices_sigtor)

                output_st_list.append(make_training_patches(radar, storm_mask, possible_x, possible_y, n_patches, time, lats, lons))


if __name__ == "__main__":
    main()





