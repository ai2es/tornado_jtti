"""
Made by Lydia 2022
Modified by Monique Shotande Feb 2023

Convert raw WoFS (Warn on Forecast System) data from the
WoFS griding format to the GridRad griding format and 
generate patches.

Execution Instructions:
    Conda environment requires are in the environment.yml
    file. Execute:
        conda env create --name tornado --file environment.yml
    to create the conda environment necessary to run this script.
    Custom module requirements:
        The custom modules: custom_losses, custom_metrics, and process_monitoring
        are in the tornado_jtti project directory.
    Information about the required command line arguments are described in the 
    method get_arguments(). Run
        python wofs_to_gridrad_idw.py --h
"""

import xarray as xr
print("xr version", xr.__version__)
import numpy as np
print("np version", np.__version__)
import netCDF4
from netCDF4 import Dataset
import scipy.spatial
import wrf
import metpy
import metpy.calc
import os, sys
import glob
import datetime
import multiprocessing as mp
import argparse
import tqdm
#sys.path.append("/home/momoshog/Tornado/tornado_jtti/process_monitoring")
sys.path.append("process_monitoring")
from process_monitor import ProcessMonitor


def find_and_create_output_path(filepath):
    #Find the wofs date of the day we are processing
    yyyymmdd_path = filepath[29:37]
    #yyyy_path = yyyymmdd_path[:4]
    
    #Pull out the year, month, day and time from the input filename
    #yyyy_day = filepath[-19:-15]
    mm_day = filepath[-14:-12]
    dd_day = filepath[-11:-9]
    #yyyymmdd_day = yyyy_day + mm_day + dd_day
    hhmmss = filepath[-8:-6] + filepath[-5:-3] + filepath[-2:]

    # Find the time of the wofs run
    #ens_mem = filepath[51:53]
    #if ens_mem[1] == '/':
    #    ens_mem = ens_mem[0]
    model_init_hhhh = filepath[38:42]

    #Define the filepath fore the directory with all the regridded wofs data    
    #test if the filepath exists, and if not, create the necessary directories
    path_components = filepath.split('/')
    yyyy = path_components[-5]
    date = path_components[-4]
    init_time = path_components[-3]
    ensem = path_components[-2]

    dir_member = os.path.join(patches_path, *path_components[-5:-1])
    if is_dry_run:
        print(path_components)
        print("dir_member", dir_member)

    try:
        if not is_dry_run: os.makedirs(dir_member)
        print(f"Making predictions dir {dir_member} (dry_run={is_dry_run})")
    except OSError as err:
        print(f"[CAUGHT] {err} in {err.filename}")
    
    # Declare the final, newly created filepath and return
    #output_filepath = f"%s/%s/%s/%s/ENS_MEM_%s/wofs_patches_%s_%s.nc" % (patches_path, yyyy_path, yyyymmdd_path, model_init_hhhh, ens_mem, yyyymmdd_day, hhmmss)
    #output_filepath = "%s/wofs_patches_%s_%s.nc" % (dir_member, yyyymmdd_day, hhmmss)
    output_filepath = "%s/wofs_patches_%s_%s.nc" % (dir_member, date, hhmmss)
    
    #We need the gridrad time to be in seconds since 2001
    time = datetime.datetime(int(yyyy), int(mm_day), int(dd_day), int(hhmmss[0:2]), int(hhmmss[2:4]), int(hhmmss[4:]))
    time = netCDF4.date2num(time,'seconds since 2001-01-01')

    forecast_window = (datetime.timedelta(hours=int(hhmmss[:2]), minutes=int(hhmmss[2:4])) - datetime.timedelta(hours=int(model_init_hhhh[:2]))).total_seconds()/60

    return output_filepath, time, forecast_window


def calculate_output_lats_lons(wofs):

    # Find the range of the wofs grid in lat lon
    # This should be a rectangle in lat/lon coordinates, so we search for the extreme values of 
    # latitude on the most constrained longitude, and the extreme values of longitude on the most
    # constrained latitude
    min_lat_wofs = wofs.XLAT[0,:,int(wofs.XLAT.shape[2]/2)].values.min()
    max_lat_wofs = wofs.XLAT[0,:,0].values.max()
    min_lon_wofs = wofs.XLONG[0,0].values.min()
    max_lon_wofs = wofs.XLONG[0,0].values.max()
    
    # Create a grid with gridrad spacing that is contained within the given wofs grid
    # Gridrad files have grid spacings of 1/48th degrees lat/lon
    new_min_lat = int(min_lat_wofs * 48 + 1)/48
    new_max_lat = int(max_lat_wofs * 48 - 1)/48
    new_min_lon = int(min_lon_wofs * 48 + 1)/48
    new_max_lon = int(max_lon_wofs * 48 - 1)/48

    # Find the total number of lats and lons for this wofs grid
    num_lats = round((new_max_lat - new_min_lat)*48 + 1)
    num_lons = round((new_max_lon - new_min_lon)*48 + 1)

    # Make the new lats and lons grid. This new grid is just contained by the original wofs grid
    new_gridrad_lats = np.linspace(new_min_lat, new_max_lat, num_lats)
    new_gridrad_lons = np.linspace(new_min_lon, new_max_lon, num_lons)
    
    # Combine the individual lat and lon array into one array with both lat and lon:
    # Make 2D arrays of for both lats and lons
    gridrad_lats = np.zeros((new_gridrad_lats.shape[0], new_gridrad_lons.shape[0]))
    gridrad_lons = np.zeros((new_gridrad_lats.shape[0], new_gridrad_lons.shape[0]))

    # Put the 1D lats lons into 2D grids
    for i in range(new_gridrad_lons.shape[0]):
        gridrad_lons[:,i] = new_gridrad_lons[i]
    for j in range(new_gridrad_lats.shape[0]):
        gridrad_lats[j] = new_gridrad_lats[j]

    # Combine the Lats and Lons into an array of tuples, where each tuple is a point to interpolate the data to
    new_gridrad_lats_lons = np.stack((np.ravel(gridrad_lats), np.ravel(gridrad_lons))).T

    return new_gridrad_lats_lons, new_gridrad_lats, new_gridrad_lons


def extract_gridrad_data_fields(filepath, gridrad_heights, Z_only=True):

    # To run function from wrf-python, we need a netCDF Dataset, not xarray
    wrfin = Dataset(filepath)
    
    # Get the heights of the wofs file
    height = wrf.getvar(wrfin, "height_agl", units='m')

    # Pull out Reflectivity, and U and V winds from the the wofs file
    Z = wrf.getvar(wrfin, "REFL_10CM")
    if not Z_only:
        U = wrf.g_wind.get_u_destag(wrfin)
        V = wrf.g_wind.get_v_destag(wrfin)   
    
    
    # Interpolate the wofs data to the gridrad heights
    Z_agl = wrf.interplevel(Z,height,gridrad_heights*1000)
    if not Z_only:
        # Interpolate the wofs data to the gridrad heights
        U_agl = wrf.interplevel(U,height,gridrad_heights*1000)
        V_agl = wrf.interplevel(V,height,gridrad_heights*1000)
    
        # Add units to the winds so that we can use the metpy functions
        U = U_agl.values * (metpy.units.units.meter/metpy.units.units.second)
        V = V_agl.values * (metpy.units.units.meter/metpy.units.units.second)
    
        # Define the grid spacings - needed for div and vort
        dx = 3000 * (metpy.units.units.meter)
        dy = 3000 * (metpy.units.units.meter)

        # Calculate divergence and vorticity
        div = metpy.calc.divergence(U, V, dx=dx, dy=dy)
        vort = metpy.calc.vorticity(U, V, dx=dx, dy=dy)
    
        #lets strip out just the data, get rid of the metpy.units stuff and xarray.Dataset stuff 
        div = np.asarray(div)
        vort = np.asarray(vort)
    
    # Remove metpy.units stuff and xarray.Dataset stuff
    Z_agl = Z_agl.values 
    uh = wrf.getvar(wrfin, 'UP_HELI_MAX')
    uh = uh.values

    if not Z_only:
        return Z_agl, div, vort, uh
    else: 
        return Z_agl, uh

#Function to turn a wofs file to the gridrad format
#where filepath is the location of the wofs file, outfile_path is the location of directory containing 
#all of the regridded wofs files ex: /ourdisk/hpc/ai2es/tornado/wofs_gridradlike/
#with_nans is whether we want to change gridpoints with reflectivity = 0 to nan values
def to_gridrad(filepath, Z_only=True):

    # There is a difference sometimes between the day in the filepath and the day in the filename
    # All filepath dates have '_path' and all filename dates have '_day'
    
    # Create the directory structure to save out the data and return the output filepath for this file
    # Create the time object corresponding to the data
    # Calculate the forecast window for this file
    output_filepath, time, forecast_window = find_and_create_output_path(filepath)
    print(f"Output path for TIME {time} and window {forecast_window}: {output_filepath}")    

    # Create process monitor for recording run times and memory usage
    proc = ProcessMonitor()
    proc.start_timer()
      
    # Open the wofs file
    print(f"Loading {filepath}")
    wofs = xr.open_dataset(filepath, engine='netcdf4')
    
    # Find the lat/lon points that will make up our regridded gridpoints
    new_gridrad_lats_lons, new_gridrad_lats, new_gridrad_lons = calculate_output_lats_lons(wofs)
    
    # Define the height values we want to interpolate to
    gridrad_heights = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

    # Pull out the desired data from the wofs file
    if Z_only:
        Z_agl, uh = extract_gridrad_data_fields(filepath, gridrad_heights, Z_only=Z_only)
    else:
        Z_agl, div, vort, uh = extract_gridrad_data_fields(filepath, gridrad_heights, Z_only=Z_only)

    # Make a list of all the lat/lon gridpoints from the original wofs grid
    wofs_lats_lons = np.stack((np.ravel(wofs.XLAT.values[0]), np.ravel(wofs.XLONG.values[0]))).T
    
    # Make a KD Tree with the wofs gridpoints
    tree = scipy.spatial.cKDTree(wofs_lats_lons)
    
    # For each point in gridrad, find the 4 gridpoints from wofs that are the closest
    distances, points = tree.query(new_gridrad_lats_lons, k=4)
    
    # For the points found above, identify the value of each wofs coordinate (bottom_top is constant)
    time_points, south_north_points, west_east_points = np.unravel_index(points, (1,300,300))
    
    #need to repeat the x,y indices 29 times so we can select all the data in one step, no loop
    south_north_points_3d = np.tile(np.reshape(south_north_points, (south_north_points.shape[0], 1, south_north_points.shape[1])), (1,gridrad_heights.shape[0],1))
    west_east_points_3d = np.tile(np.reshape(west_east_points, (west_east_points.shape[0], 1, west_east_points.shape[1])), (1,gridrad_heights.shape[0],1))
    distances_3d = np.tile(np.reshape(distances, (distances.shape[0], 1, distances.shape[1])), (1,gridrad_heights.shape[0], 1))
    
    #need to make the z coordinate indices. This might be able to be improved, but works fine.  
    for i in np.arange(0,gridrad_heights.shape[0]):
        if i == 0:
            bottom_top_3d = np.zeros((south_north_points.shape[0],south_north_points.shape[1],1),dtype=int) 
        else:
            bottom_top_3d = np.append(bottom_top_3d,np.ones((south_north_points.shape[0], south_north_points.shape[1], 1),dtype=int)*int(i), axis=2)
    bottom_top_3d = np.swapaxes(bottom_top_3d, 1, 2)

    #now selecting the data should be fast 
    if not Z_only:
        vorts = vort[bottom_top_3d,south_north_points_3d,west_east_points_3d]
        divs = div[bottom_top_3d,south_north_points_3d,west_east_points_3d]
    refls = Z_agl[bottom_top_3d,south_north_points_3d,west_east_points_3d]
    uhs = uh[south_north_points,west_east_points]

    # Perform the IDW interpolation and reformat the data so that it has shape (Time, Altitude, Latitude, Longitude)
    if not Z_only:
        vort_final = np.swapaxes(np.swapaxes((np.sum(vorts * 1/distances_3d**2, axis=2)/np.sum(1/distances_3d**2, axis=2)).reshape(1,new_gridrad_lats.shape[0],new_gridrad_lons.shape[0],vort.shape[0]), 1, 3), 2, 3).astype(np.float32)
        div_final = np.swapaxes(np.swapaxes((np.sum(divs * 1/distances_3d**2, axis=2)/np.sum(1/distances_3d**2, axis=2)).reshape(1,new_gridrad_lats.shape[0],new_gridrad_lons.shape[0],vort.shape[0]), 1, 3), 2, 3).astype(np.float32)
    REFL_10CM_final = np.swapaxes(np.swapaxes((np.sum(refls * 1/distances_3d**2, axis=2)/np.sum(1/distances_3d**2, axis=2)).reshape(1,new_gridrad_lats.shape[0],new_gridrad_lons.shape[0],vort.shape[0]), 1, 3), 2, 3).astype(np.float32)
    uh_final = (np.sum(uhs * 1/distances**2, axis=1)/np.sum(1/distances**2, axis=1)).reshape(1,new_gridrad_lats.shape[0],new_gridrad_lons.shape[0]).astype(np.float32)
    
    # Put the data into xarray DataArrays that have the same dimensions, coordinates, and variable fields as gridrad
    if not Z_only:
        wofs_regridded_vort = xr.DataArray(
            data=vort_final,
            dims=("time", "Altitude", "Latitude", "Longitude"),
            coords={"time": [time], "Altitude": gridrad_heights, "Latitude": new_gridrad_lats, "Longitude": new_gridrad_lons + 360},
        )
        wofs_regridded_div = xr.DataArray(
            data=div_final,
            dims=("time", "Altitude", "Latitude", "Longitude"),
            coords={"time": [time], "Altitude": gridrad_heights, "Latitude": new_gridrad_lats, "Longitude": new_gridrad_lons + 360},
        )
    wofs_regridded_refc = xr.DataArray(
        data=REFL_10CM_final,
        dims=("time", "Altitude", "Latitude", "Longitude"),
        coords={"time": [time], "Altitude": gridrad_heights, "Latitude": new_gridrad_lats, "Longitude": new_gridrad_lons + 360},
    )
    wofs_regridded_uh = xr.DataArray(
        data=uh_final,
        dims=("time", "Latitude", "Longitude"),
        coords={"time": [time], "Latitude": new_gridrad_lats, "Longitude": new_gridrad_lons + 360},
    )
    
    # Combine the DataArrays into one Dataset
    wofs_regridded = xr.Dataset(coords={"time": [time], "Longitude": new_gridrad_lons + 360, "Latitude": new_gridrad_lats, "Altitude": gridrad_heights})
    wofs_regridded["ZH"] = wofs_regridded_refc
    wofs_regridded["UH"] = wofs_regridded_uh
    if not Z_only:
        wofs_regridded["VOR"] = wofs_regridded_vort
        wofs_regridded["DIV"] = wofs_regridded_div
    
    #Where there is no reflectivity, change 0s to nans for all data variables
    if(with_nans):
        wofs_regridded = wofs_regridded.where(wofs_regridded.ZH > 0)
    
     
    #make validation patches
    output = make_validation_patches(wofs_regridded, size, forecast_window, filepath, Z_only)
    wofs_regridded.close()
    
    # Stop the process timer
    proc.end_timer()
    basepath = os.path.join(patches_path, f'process_monitoring_idx{index_primer}_{time}')
    print('pm basepath', basepath)
    if not is_dry_run:
        proc.write_performance_monitor(output_path=basepath + '_performance.csv')
        proc.plot_performance_monitor(attrs=None, ax=None, write=True,
                                        output_path=basepath + '_performance_plot.png', format='png')
        proc.print(write=True, output_path=basepath + '.csv')
    else: proc.print(write=False)
    
    
    #save out the data
    print(f"Saving validation patches: {output_filepath}")
    if not is_dry_run: 
        output.to_netcdf(output_filepath)
    output.close()

    del output
    del wofs_regridded
    del wofs
    #del wrfin
    
    
def make_validation_patches(radar, size, window, filepath, Z_only=True):

    #initialize an empty array for the patches
    patches = []
    
    #Loop through Latitude and Longitude and pull out every size-th pixel. This will run for every (xi, yi) = multiples of size, and will give us a normal array
    for xi in range(0, radar.Latitude.shape[0], size-4):
        for yi in range(0, radar.Longitude.shape[0], size-4):  
        
            # Account for the case that the patch goes outside of the domain
            # If part of the patch is outside the domain, move over, so the patch edge lines up with the domain edge
            if xi >= radar.Latitude.shape[0]-size:
                xi = radar.Latitude.shape[0]-size - 1
            if yi >= radar.Longitude.shape[0]-size:
                yi = radar.Longitude.shape[0]-size - 1

            # Create the patch
            if not Z_only:
                to_add = xr.Dataset(data_vars=dict(
                           ZH=(["x", "y", "z"], radar.ZH.isel(Latitude=slice(xi, xi+size), Longitude=slice(yi, yi+size), time=0).values.swapaxes(0,2).swapaxes(1,0)), 
                           DIV=(["x", "y", "z"], radar.DIV.isel(Latitude=slice(xi, xi+size), Longitude=slice(yi, yi+size), time=0).values.swapaxes(0,2).swapaxes(1,0)),
                           VOR=(["x", "y", "z"], radar.VOR.isel(Latitude=slice(xi, xi+size), Longitude=slice(yi, yi+size), time=0).values.swapaxes(0,2).swapaxes(1,0)),
                           UH=(["x", "y"], radar.UH.isel(Latitude=slice(xi, xi+size), Longitude=slice(yi, yi+size), time=0, Altitude=0).values),
                           stitched_x=(["x"], range(xi, xi+size)),
                           stitched_y=(["x"], range(yi, yi+size)),
                           n_convective_pixels = ([], np.count_nonzero(radar.ZH.isel(Latitude=slice(xi, xi+size), Longitude=slice(yi, yi+size), time=0).fillna(0).values.swapaxes(0,2).swapaxes(1,0).max(axis=(2)) >= 30)),
                           n_uh_pixels = ([], np.count_nonzero(radar.UH.isel(Latitude=slice(xi, xi+size), Longitude=slice(yi, yi+size), time=0).fillna(0).values > 0)),
                           lat=([],radar.Latitude.values[xi]),
                           lon=([],radar.Longitude.values[yi]),
                           time=([],radar.time.values[0]),
                           forecast_window=([],window)),
                    coords=dict(x=(["x"], np.arange(size)), y=(["y"], np.arange(size)), z=(["z"], np.arange(1,radar.Altitude.values.shape[0]+1))))
            else:
                to_add = xr.Dataset(data_vars=dict(
                           ZH=(["x", "y", "z"], radar.ZH.isel(Latitude=slice(xi, xi+size), Longitude=slice(yi, yi+size), time=0).values.swapaxes(0,2).swapaxes(1,0)), 
                           UH=(["x", "y"], radar.UH.isel(Latitude=slice(xi, xi+size), Longitude=slice(yi, yi+size), time=0, Altitude=0).values),
                           stitched_x=(["x"], range(xi, xi+size)),
                           stitched_y=(["x"], range(yi, yi+size)),
                           n_convective_pixels = ([], np.count_nonzero(radar.ZH.isel(Latitude=slice(xi, xi+size), Longitude=slice(yi, yi+size), time=0).fillna(0).values.swapaxes(0,2).swapaxes(1,0).max(axis=(2)) >= 30)),
                           n_uh_pixels = ([], np.count_nonzero(radar.UH.isel(Latitude=slice(xi, xi+size), Longitude=slice(yi, yi+size), time=0).fillna(0).values > 0)),
                           lat=([],radar.Latitude.values[xi]),
                           lon=([],radar.Longitude.values[yi]),
                           time=([],radar.time.values[0]),
                           forecast_window=([],window)),
                    coords=dict(x=(["x"], np.arange(size)), y=(["y"], np.arange(size)), z=(["z"], np.arange(1,radar.Altitude.values.shape[0]+1))))
            to_add = to_add.fillna(0)

            # Add this patch to the total patches list
            patches.append(to_add)

    # Combine all patches into one dataset
    output = xr.concat(patches, 'patch')
    
    return output


def get_arguments():
    #Define the strings that explain what each input variable means
    INDEX_PRIMER_HELP_STRING = 'i, where i indicates the ith day of wofs to process. The task array should go from 0 to the total number of WoFS days to process.'
    WOFS_DIR_HELP_STRING = 'The directory where the raw WoFS files can be found. This directory should start from the root directory \'/\', and should end with wildcards which include all the wofs days to process.'
    PATCHES_DIR_HELP_STRING = 'The directory where the regridded wofs patches will be stored. This directory should start from the root directory \'/\'.'
    PATCHES_SIZE_HELP_STRING = 'The size of the patch in each horizontal dimension. Ex: patch_size=32 would make patches of shape (32,32,12).'
    WITH_NANS_HELP_STRING = 'If with_nans=1, datapoints with reflectivity=0 will be stored as nans. If with_nans=0, those points will be stored as normal float values.'

    #Tell python what arguments to expect from the .sh file
    INPUT_ARG_PARSER = argparse.ArgumentParser()
    INPUT_ARG_PARSER.add_argument('--array_index', type=str, required=True, help=INDEX_PRIMER_HELP_STRING)
    INPUT_ARG_PARSER.add_argument('--path_to_raw_wofs', type=str, required=True, help=WOFS_DIR_HELP_STRING)
    INPUT_ARG_PARSER.add_argument('--output_patches_path', type=str, required=True, help=PATCHES_DIR_HELP_STRING)
    INPUT_ARG_PARSER.add_argument('--patch_size', type=int, required=True, help=PATCHES_SIZE_HELP_STRING)
    INPUT_ARG_PARSER.add_argument('--with_nans', type=int, required=True, help=WITH_NANS_HELP_STRING)

    INPUT_ARG_PARSER.add_argument('-Z', '--Z_only', action='store_true',
        help='Use flag to only extraact the reflectivity and updraft data. Exclude U and V')
    INPUT_ARG_PARSER.add_argument('-d', '--dry_run', action='store_true',
        help='For testing. Execute without running or saving data and verify output paths')

    #Pull out all of the input strings from the .sh file
    args = INPUT_ARG_PARSER.parse_args()
    #Index primer indicates the day of data that we are looking at in this particuar run of this code
    global index_primer 
    index_primer = getattr(args, 'array_index')
    #path_to_wofs is the path to the raw wofs file that we are going to re-grid
    global path_to_wofs 
    path_to_wofs = getattr(args, 'path_to_raw_wofs')
    #patches_path is the patch that the output will be saved at
    global patches_path 
    patches_path = getattr(args, 'output_patches_path')
    #size is the patch size
    global size 
    size = getattr(args, 'patch_size')
    #with_nans is the patch size
    global with_nans 
    with_nans = getattr(args, 'with_nans')
    global is_dry_run
    is_dry_run = args.dry_run
    
    print(args)
    #vars(args).items()
    return args


def find_wofs_date(all_wofs_days, idx):

    #find the filepath of the date we will process
    date_path = all_wofs_days[idx]

    #pull out the year, month, and day
    yyyy = date_path[29:33]
    mm = date_path[33:35]
    dd = date_path[35:37]

    return yyyy, mm, dd


def main():    

    #get the inputs from the .sh file
    args = get_arguments()

    #Glob all the days that we have data
    model_runs = glob.glob(path_to_wofs)
    print("Model runs", model_runs)

    #Isolate the date of the data we are processing
    yyyy, mm, dd = find_wofs_date(model_runs, int(index_primer))
    print('Date to process', yyyy , mm , dd)

    #Make a list of all the filepaths for this day
    #They have the form /ourdisk/hpc/ai2es/wofs/2019/20190430/0000/ENS_MEM_1/wrfwof_d01_*
    filenames = glob.glob(f"/ourdisk/hpc/ai2es/wofs/%s/%s%s%s/*/ENS_MEM_*/wrfwof_d01_*" % (yyyy,yyyy,mm,dd))[:1] ## MOSHO [:1]
    filenames.sort()
    print("filenames", filenames)

    for file in filenames:
        to_gridrad(file, Z_only=args.Z_only)
    #with mp.Pool(processes=1) as p: #20
    #    tqdm.tqdm(p.map(to_gridrad, filenames), total=len(filenames))    


if __name__ == "__main__":
    main()
