import xarray as xr
import numpy as np
from netCDF4 import Dataset
import scipy.spatial
import wrf
import metpy
import metpy.calc
import os
import glob
import datetime
import netCDF4
import multiprocessing as mp
import argparse
import tqdm

def find_and_create_output_path(filepath):

    # Find the ensemble member
    ens_mem = filepath.split('/')[-2]
    date_fcst = filepath[-19:-9]
    time_fcst = filepath[-8:]

    #Define the filepath fore the directory with all the regridded wofs data    
    #test if the filepath exists, and if not, create the necessary directories
    os.makedirs(os.path.join(patches_path, model_init_date[:4]), exist_ok=True)
    os.makedirs(os.path.join(patches_path, model_init_date[:4], model_init_date), exist_ok=True)
    os.makedirs(os.path.join(patches_path, model_init_date[:4], model_init_date, model_init_time), exist_ok=True)
    os.makedirs(os.path.join(patches_path, model_init_date[:4], model_init_date, model_init_time, ens_mem), exist_ok=True)

    # Declare the final, newly created filepath and return
    output_filepath = os.path.join(patches_path, model_init_date[:4], model_init_date, model_init_time, ens_mem,
                                   f"wofs_patches_{date_fcst}_{time_fcst}.nc")
    
    #We need the gridrad time to be in seconds since 2001
    time = datetime.datetime(int(date_fcst[:4]), int(date_fcst[5:7]), int(date_fcst[8:]),
                             int(time_fcst[:2]), int(time_fcst[3:5]), int(time_fcst[6:]))
    time = netCDF4.date2num(time,'seconds since 2001-01-01')

    forecast_window = datetime.timedelta(days=int(date_fcst[8:]) - int(model_init_date[6:]),
                                         hours=int(time_fcst[:2]) - int(model_init_time[:2]),
                                         minutes=int(time_fcst[3:5]) - int(model_init_time[2:])).total_seconds()/60
    
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

def extract_gridrad_data_fields(filepath, gridrad_heights):

    # To run function from wrf-python, we need a netCDF Dataset, not xarray
    wrfin = Dataset(filepath)
    
    # Get the heights of the wofs file
    height = wrf.getvar(wrfin, "height_agl", units='m')

    # Pull out Reflectivity, and U and V winds from the the wofs file
    Z = wrf.getvar(wrfin, "REFL_10CM")
    U = wrf.g_wind.get_u_destag(wrfin)
    V = wrf.g_wind.get_v_destag(wrfin)   
    
    # Interpolate the wofs data to the gridrad heights
    Z_agl = wrf.interplevel(Z,height,gridrad_heights*1000)
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
    uh = wrf.getvar(wrfin, 'UP_HELI_MAX')
    
    #lets strip out just the data, get rid of the metpy.units stuff and xarray.Dataset stuff 
    div = np.asarray(div)
    vort = np.asarray(vort)
    Z_agl = Z_agl.values 
    uh = uh.values
    del wrfin

    return Z_agl, div, vort, uh

#Function to turn a wofs file to the gridrad format
#where filepath is the location of the wofs file, outfile_path is the location of directory containing 
#all of the regridded wofs files ex: /ourdisk/hpc/ai2es/tornado/wofs_gridradlike/
#with_nans is whether we want to change gridpoints with reflectivity = 0 to nan values
def to_gridrad(filepath):

    # There is a difference sometimes between the day in the filepath and the day in the filename
    # All filepath dates have '_path' and all filename dates have '_day'
    
    # Create the directory structure to save out the data and return the output filepath for this file
    # Create the time object corresponding to the data
    # Calculate the forecast window for this file
    output_filepath, time, forecast_window = find_and_create_output_path(filepath)
      
    # Open the wofs file
    wofs = xr.open_dataset(filepath)
    
    # Find the lat/lon points that will make up our regridded gridpoints
    new_gridrad_lats_lons, new_gridrad_lats, new_gridrad_lons = calculate_output_lats_lons(wofs)
    
    # Define the height values we want to interpolate to
    gridrad_heights = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

    # Pull out the desired data from the wofs file
    Z_agl, div, vort, uh = extract_gridrad_data_fields(filepath, gridrad_heights)



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
    vorts = vort[bottom_top_3d,south_north_points_3d,west_east_points_3d]
    divs = div[bottom_top_3d,south_north_points_3d,west_east_points_3d]
    refls = Z_agl[bottom_top_3d,south_north_points_3d,west_east_points_3d]
    uhs = uh[south_north_points,west_east_points]

    # Perform the IDW interpolation and reformat the data so that it has shape (Time, Altitude, Latitude, Longitude)
    vort_final = np.swapaxes(np.swapaxes((np.sum(vorts * 1/distances_3d**2, axis=2)/np.sum(1/distances_3d**2, axis=2)).reshape(1,new_gridrad_lats.shape[0],new_gridrad_lons.shape[0],vort.shape[0]), 1, 3), 2, 3).astype(np.float32)
    div_final = np.swapaxes(np.swapaxes((np.sum(divs * 1/distances_3d**2, axis=2)/np.sum(1/distances_3d**2, axis=2)).reshape(1,new_gridrad_lats.shape[0],new_gridrad_lons.shape[0],vort.shape[0]), 1, 3), 2, 3).astype(np.float32)
    REFL_10CM_final = np.swapaxes(np.swapaxes((np.sum(refls * 1/distances_3d**2, axis=2)/np.sum(1/distances_3d**2, axis=2)).reshape(1,new_gridrad_lats.shape[0],new_gridrad_lons.shape[0],vort.shape[0]), 1, 3), 2, 3).astype(np.float32)
    uh_final = (np.sum(uhs * 1/distances**2, axis=1)/np.sum(1/distances**2, axis=1)).reshape(1,new_gridrad_lats.shape[0],new_gridrad_lons.shape[0]).astype(np.float32)
    
    
    
    
    # Put the data into xarray DataArrays that have the same dimensions, coordinates, and variable fields as gridrad
    wofs_regridded_refc = xr.DataArray(
        data=REFL_10CM_final,
        dims=("time", "Altitude", "Latitude", "Longitude"),
        coords={"time": [time], "Altitude": gridrad_heights, "Latitude": new_gridrad_lats, "Longitude": new_gridrad_lons + 360},
    )
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
    wofs_regridded_uh = xr.DataArray(
        data=uh_final,
        dims=("time", "Latitude", "Longitude"),
        coords={"time": [time], "Latitude": new_gridrad_lats, "Longitude": new_gridrad_lons + 360},
    )
    
    # Combine the DataArrays into one Dataset
    wofs_regridded = xr.Dataset(coords={"time": [time], "Longitude": new_gridrad_lons + 360, "Latitude": new_gridrad_lats, "Altitude": gridrad_heights})
    wofs_regridded["ZH"] = wofs_regridded_refc
    wofs_regridded["VOR"] = wofs_regridded_vort
    wofs_regridded["DIV"] = wofs_regridded_div
    wofs_regridded["UH"] = wofs_regridded_uh
    
    #Where there is no reflectivity, change 0s to nans for all data variables
    if(with_nans):
        wofs_regridded = wofs_regridded.where(wofs_regridded.ZH > 0)
    
    
     
    #make validation patches
    output = make_validation_patches(wofs_regridded, size, forecast_window, filepath)
    wofs_regridded.close()
    del wofs_regridded
    
    #save out the data            
    output.to_netcdf(output_filepath)
    output.close()
    del output
    del wofs

def make_validation_patches(radar, size, window, filepath):

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
            to_add = to_add.fillna(0)

            # Add this patch to the total patches list
            patches.append(to_add)

    # Combine all patches into one dataset
    output = xr.concat(patches, 'patch')
    
    return output

def get_arguments():
    #Define the strings that explain what each input variable means
    WOFS_DIR_HELP_STRING = 'The directory where the raw WoFS files can be found. This directory should start from the root directory \'/\', and should end with wildcards which include all the wofs days to process.'
    MODEL_INIT_DATE_HELP_STRING = 'The model initialization date for which the raw WoFS files will be processed. The date should be in the following format: yyyymmdd.'
    MODEL_INIT_TIME_HELP_STRING = 'The model initialization time for which the raw WoFS files will be processed. The time should be in the following format: hhmm.'
    PATCHES_DIR_HELP_STRING = 'The directory where the regridded wofs patches will be stored. This directory should start from the root directory \'/\'.'
    PATCHES_SIZE_HELP_STRING = 'The size of the patch in each horizontal dimension. Ex: patch_size=32 would make patches of shape (32,32,12).'
    WITH_NANS_HELP_STRING = 'If with_nans=1, datapoints with reflectivity=0 will be stored as nans. If with_nans=0, those points will be stored as normal float values.'

    #Tell python what arguments to expect from the .sh file
    INPUT_ARG_PARSER = argparse.ArgumentParser()
    INPUT_ARG_PARSER.add_argument('--path_to_raw_wofs', type=str, required=True, help=WOFS_DIR_HELP_STRING)
    INPUT_ARG_PARSER.add_argument('--model_init_date', type=str, required=True, help=MODEL_INIT_DATE_HELP_STRING)
    INPUT_ARG_PARSER.add_argument('--model_init_time', type=str, required=True, help=MODEL_INIT_TIME_HELP_STRING)
    INPUT_ARG_PARSER.add_argument('--output_patches_path', type=str, required=True, help=PATCHES_DIR_HELP_STRING)
    INPUT_ARG_PARSER.add_argument('--patch_size', type=int, required=True, help=PATCHES_SIZE_HELP_STRING)
    INPUT_ARG_PARSER.add_argument('--with_nans', type=int, required=True, help=WITH_NANS_HELP_STRING)

    #Pull out all of the input strings from the .sh file
    args = INPUT_ARG_PARSER.parse_args()
    #path_to_wofs is the path to the raw wofs file that we are going to re-grid
    global path_to_wofs 
    path_to_wofs = getattr(args, 'path_to_raw_wofs')
    #model_init_date is the date of the directory that we are going to re-grid
    global model_init_date 
    model_init_date = getattr(args, 'model_init_date')
    #model_init_time is the model initialization time of the directory that we are going to re-grid
    global model_init_time 
    model_init_time = getattr(args, 'model_init_time')
    #patches_path is the patch that the output will be saved at
    global patches_path 
    patches_path = getattr(args, 'output_patches_path')
    #size is the patch size
    global size 
    size = getattr(args, 'patch_size')
    #with_nans is the patch size
    global with_nans 
    with_nans = getattr(args, 'with_nans')

def main():    

    #get the inputs from the .sh file
    get_arguments()

    #Glob all the days that we have data
    model_runs = glob.glob(path_to_wofs)

    #Make a list of all the filepaths for this date and model_init_time
    filenames = glob.glob(f"/glade/p/cisl/aiml/jtti_tornado/wofs/{model_init_date[:4]}/{model_init_date}/{model_init_time}/ENS_MEM_*/wrfwof_d01_*")
    filenames.sort()
    print(filenames[2])
    print(filenames[-2])

    with mp.Pool(processes=2) as p:
        tqdm.tqdm(p.map(to_gridrad, filenames), total=len(filenames))

if __name__ == "__main__":
    main()