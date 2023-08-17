''' This script is designed to house the methods used to run the validation of the WoFS ensemble and the ML outputs
The functions will be grouped based on their functionality:
0. Imports
1. Constants
2. Graphing
3. Analysis
4. Misc
5. Functions that use the previous to generate results
'''
################################################################################################################################################
''' IMPORTS '''
################################################################################################################################################
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import scipy.ndimage as ndimage
#from netCDF4 import Dataset
import netCDF4 as nf
import xarray as xr
import metpy.calc as mpcalc
from metpy.units import units as mpunits
import pandas as pd
import skimage.measure as ski
import math
import wrf
from pyproj import Proj
from bs4 import BeautifulSoup
import requests
import datetime
import openpyxl
import tqdm

# IMPORTS FOR IDENTIFICATION
# NOTE: NEED TO FIX THIS WHEN I PUSH THIS FILE TO THE SUPERCOMPUTER (SPECIFICALLY PUSH THAT FILE AND CHANGE THE PATH)

import sys, os
sys.path.append(os.path.dirname(os.getcwd()))

sys.path.append(os.path.abspath('/ourdisk/hpc/ai2es/alexnozka/tools/MontePython/'))
from monte_python import *
import monte_python

################################################################################################################################################
''' CONSTANTS '''
################################################################################################################################################

# Plotting constants
crs = ccrs.LambertConformal(central_longitude=-100.0)
# Identification/Classification constants SUBJECT TO CHANGE
reflectivity_threshold = 40.
uh_threshold = 40.
updraft_threshold = 5.
echo_height_min_threshold = 4000.
echo_height_max_threshold = 15000.
### TORNADIC_CRITERION = PERCENT NEEDED TO BE CONSIDERED TO HAVE FORECASTED A TORNADO: SUBJECT TO CHANGE, is varied through the CSI analysis
tornadic_criterion = 0.3
tor_link_distance = 10000 # Meters away from the storm 
# Evaluation graphs constants
FREQ_BIAS_COLOUR = np.full(3, 152. / 255)
FREQ_BIAS_WIDTH = 2.
FREQ_BIAS_STRING_FORMAT = '%.2f'
FREQ_BIAS_PADDING = 5

################################################################################################################################################
''' PLOTTING '''
################################################################################################################################################

# Function used to create the map subplot backgrounds over a specified subdomain
def plot_subset_background(ax, lonmin, lonmax, latmin, latmax):
    ax.set_extent([lonmin, lonmax, latmin, latmax])
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    return ax

# Placing labels in the projection:
def label_center(x, y, ax, object_props, storm_modes=None, converter=None):
    """Place object label on object's centroid"""
    for region in object_props:
        x_cent, y_cent = region.centroid
        x_cent = int(x_cent)
        y_cent = int(y_cent)
        try:
            xx, yy = x[x_cent, y_cent], y[x_cent, y_cent]
        except(IndexError):
            xx, yy = x[x_cent], y[y_cent]

        if storm_modes is None:
            fontsize = 6.5 if region.label >= 10 else 8
            txt = region.label
        else:
            fontsize = 4
            coords = region.coords
            ind = int(np.max(storm_modes[coords[:, 0], coords[:, 1]]))
            txt = converter[ind]

        ax.text(xx, yy,
                txt,
                fontsize=fontsize,
                transform=ccrs.PlateCarree(),
                ha='center',
                va='center',
                color='k'
                )

# Plotting the storm objects on a specified domain
def plot_storms_subset(x, y, labels, lonmin_in, lonmax_in, latmin_in, latmax_in, ax=None, storm_props=None, alpha=1.0):
    '''
    params:
        x: storm object x coordinates
        y: storm object y coordinates
        labels: storm labels (0 or 1)
        lon/lat min/max _in: the bounds of the subsetted domain (may be the full domain if desired)
        ax = None: (matplotlib_Axes), axes given to plot the storm, None will create a new axes to plot the storms onto
        storm_props = None: List of storm ID numbers to plot, will plot when storm_props != 1
    returns:
        ax: (matplotlib_Axes) plotted with the storm objects
    '''
    if ax is None:
        f, ax = plt.subplots(figsize=(5, 4), dpi=150,
                             facecolor='w', edgecolor='k')

    labels = np.ma.masked_where(labels == 0, labels)
    c = ax.pcolormesh(x, y, labels, cmap='tab20', vmin=1, vmax=np.max(
        labels), alpha=alpha, transform=ccrs.PlateCarree())
    #ax.tick_params(axis='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    plot_subset_background(ax, lonmin_in, lonmax_in, latmin_in, latmax_in)

    ### LABEL PROPS NOT WORKING ###
    if storm_props != None:
        #sub_label_props = find_subset_labels(dataset,data_column, label_props, lonmin_in,lonmax_in,latmin_in,latmax_in)
        label_center(x, y, ax, storm_props)

    return(ax)

# Returns an array of the storm domain using the tornado reports
### RETURN FORMATTED [lonmin,lonmax,latmin,latmax]
def find_storm_domain(dataset, ML = False):
    storm_bounds = [None] * 4
    storm_bounds[0] = dataset['XLONG'].min()
    storm_bounds[1] = dataset['XLONG'].max()
    storm_bounds[2] = dataset['XLAT'].min()
    storm_bounds[3] = dataset['XLAT'].max()
    return storm_bounds

# Returns an array of the storm domain using the tornado reports
### RETURN FORMATTED [lonmin,lonmax,latmin,latmax]
def find_tornado_domain(filtered_tor_rpts):
    storm_bounds = [None] * 4
    storm_bounds[0] = int(filtered_tor_rpts['Lon'].min()) -1
    storm_bounds[1] = int(filtered_tor_rpts['Lon'].max()) + 1
    storm_bounds[2] = int(filtered_tor_rpts['Lat'].min()) -1
    storm_bounds[3] = int(filtered_tor_rpts['Lat'].max()) + 1
    return storm_bounds

################################################################################################################################################
''' ANALYSIS '''
################################################################################################################################################

"""ECHO TOP HEIGHTS"""
def get_echotop(filename,threshold=40):

    #load wofs file 
    wrfin = nf.Dataset(filename)

    #get altitudes of model levels
    height = wrf.getvar(wrfin, "height_agl",units='m')
    
    #get Z
    Z = wrf.getvar(wrfin, "REFL_10CM")
    
    #interpolate it to common vertical grid
    gridrad_levels = np.arange(0.5,22,0.1)*1000
    Z_agl = wrf.interplevel(Z,height, gridrad_levels)
    
    #swap dim so we can have channels last 
    Z_agl = np.moveaxis(Z_agl.values,[0],[-1])
    
    #get ids where Z_agl >= the threshold
    idx = np.where(Z_agl >= threshold)

    #build DataFrame to use groupby 
    df = pd.DataFrame({'x':idx[0],'y':idx[1],'z':idx[2]})
    #take the maximum altitude index of Z
    zidx = df.groupby(['x','y']).max()['z']

    #define empty array 
    echo_top = np.zeros([Z_agl.shape[0],Z_agl.shape[1]])
    
    #fill empty array with echotop heights 
    for i,(x,y) in enumerate(zidx.index):
        echo_top[x,y] = gridrad_levels[zidx.iloc[i]]
        
    return echo_top

"""Table building functions"""

def link_WoFS_tornadoes(storm_objects, dataset, tornado_reports):

    """This function links the closest storm to the tornado from the WoFS output
    params:
        storm_object: type: numpy array of skimage.regiongroups objects
            the objects within the array represent the different storms identified by Reflectivity
        dataset: type: xarray 
            the dataset should be the WoFS or ML data at a particular timestep (subject to change)
        tornado_reports: type: DataFrame
            the DataFrame will consist of all the tornado reports within the timestep (subject to change)
    returns:
        cells_dist: type: numpy array
            The list of cells that are linked to tornadoes during this timestep
        OB_tornado: type: numpy array
            The list of tornadoes that occured that were not sufficiently close to a storm (Missed tornadoes)
    """
    distances = []
    cells_dist = []
    OB_tornado = []

    for i in range(0, len(tornado_reports)):
        tor = Proj(proj='aeqd', ellps='WGS84', datum='WGS84',
                   lat_0=tornado_reports['Lat'][i], lon_0=tornado_reports['Lon'][i])
        
        # LOOPING/SELECTING EACH STORM IDENTIFIED
        for j in storm_objects:
            if j.label < 0:
                continue
            
            # VALIDATION BASED ON CLOSEST PIXEL
            # Grabbing the pixel coordinates (row,col) within the objects
            coords = j.coords
            # LOOPING THROUGH EACH PIXEL WITHIN EACH STORM
            mindist = 0
            for k in range(0, len(coords)):
                # This has to be where the error is
                lat = float(dataset['XLAT'].isel(
                    Time=0, west_east=coords[k][1], south_north=coords[k][0]))
                lon = float(dataset['XLONG'].isel(
                    Time=0, west_east=coords[k][1], south_north=coords[k][0]))
                #print("Cell: (" + str(lat) + "," + str(lon) + ")" )
                distx, disty = tor(float(lon), float(lat))
                dist = np.sqrt(distx**2+disty**2)
                if k == 0:
                    mindist = dist
                else:
                    if dist < mindist:
                        midist = dist
            
            # Validation based on storm center
            cent_coords = j.centroid
            # Finding the integer pixel center
            lonint = float(dataset['XLONG'].isel(Time=0,west_east=int(cent_coords[1]),south_north=int(cent_coords[0])))
            latint = float(dataset['XLAT'].isel(Time=0,west_east=int(cent_coords[1]),south_north=int(cent_coords[0])))

            # Creating the linked tornado object
            linked_obj = {'Tornado': i,
                            'Tor coords': [tornado_reports['Lon'][i], tornado_reports['Lat'][i]],
                            ### For storm center
                            #'Distance': dist,
                            #'Closest cell': np.array([j.label, centlon, centlat])}
                            ### For closest storm pixel
                            'Distance': mindist,
                            'Closest cell': j.label}
            distances.append(linked_obj)
            
        linked = False
        for cells in distances: # Filtering such that the storm object is only counted once
            if cells['Distance'] <= tor_link_distance:
                cells_dist.append(cells)
                linked = True
                break
        if linked == False: # Adding missed tornadoes
            OB_tornado.append(i)

    cells_dist = np.array(cells_dist)
    OB_tornado = np.array(OB_tornado)

    return cells_dist, OB_tornado


#Function compiles a unique list of cells that are close to tornado reports
def unique_cells_list(linked_tornado):
    unique_list = []
    for tornado in linked_tornado:
        # If using the Central lon/lat are included in the Closest cell field
        '''
        if tornado['Closest cell'][0] not in unique_list:
            unique_list.append(tornado['Closest cell'][0])
        '''
        if tornado['Closest cell'] not in unique_list:
            unique_list.append(tornado['Closest cell'])
    return unique_list

#Function has identical process as the previous function, except returns the whole dictionary rather than just the cell label
def unique_tornadoes(linked_tornado):
    unique_list = []
    for tornado in linked_tornado:
        if tornado['Closest cell'][0] not in unique_list:
            unique_list.append(tornado)
    return unique_list


def find_storm_properties(storm_objects, dataset, model_refl, model_uh, echo_top, tornado_reports, ML=False):
    """
    params:
        storm_object: type: numpy array of skimage.regiongroups objects
            the objects within the array represent the different storms identified by Reflectivity
        dataset: type: xarray 
            the dataset should be the WoFS or ML data at a particular timestep (subject to change)
        tornado_reports: type: DataFrame
            the DataFrame will consist of all the tornado reports within the timestep (subject to change)
    return:
        stornm_properties: type: Pandas DataFrame
            The DataFrame will consist of the storm label, centroid coordinates, max UH and DBZ,
            and the storm linkage to a tornado. This labled as 'stat', which refers to if the tornado was forecasted and/or observed a tornado.
                columns: [Storm ID, Central Coordinate, Max Reflectivity, Max Updraft Helicity, Max Echotop height, stat, ML max probability, ML Stat]
                    ML columns only if ML files are present, otherwise evaluates WoFS only
                Stat line: Describing the tornadic properties of the storm 
                    0: True negative (not forecasted nor had a tornado)
                    1: False positive (False Alarm) (forecasted but no tornado)
                    2: False negative (Miss) (not forecasted but observed tornado)
                    3: True positive (Hit) (forecasted and observed tornado)
    """
    if(ML == False):
        storm_properties = pd.DataFrame(index=range(0, len(storm_objects)),
                                    columns=['StormID', 'CentCoords', 'MaxDBZ', 'MaxUH', 'EchoMaxZ',
                                             'Stat'])
    else:
        storm_properties = pd.DataFrame(index=range(0, len(storm_objects)),
                                    columns=['StormID', 'CentCoords', 'MaxDBZ', 'MaxUH', 'EchoMaxZ',
                                             'Stat', 'ML_Max_Prob', 'ML_Stat'])
    tor_confirmation, OB_tornado = link_WoFS_tornadoes(storm_objects, dataset, tornado_reports)
    confirmed_cell_list = unique_cells_list(tor_confirmation)
    # Looping through each identified storm object
    for storm in (range(0, len(storm_objects))):
        storm_properties['StormID'][storm] = storm_objects[storm].label
        
        # Finding the integer pixel center
        cent_coords = storm_objects[storm].centroid
        lonint = float(dataset['XLONG'].isel(Time=0,west_east=int(cent_coords[1]),south_north=int(cent_coords[0])))
        latint = float(dataset['XLAT'].isel(Time=0,west_east=int(cent_coords[1]),south_north=int(cent_coords[0])))
        storm_properties['CentCoords'][storm] = (lonint,latint)
        
        # Finding the list of pixels in the storm object
        storm_coords = storm_objects[storm].coords
        """The next section goes through pixel by pixel, so this will be used for the following
        1. MaxDBZ within the storm
        2. MaxUH within the storm
        3. Maximum ML probability within the storm object (if applicable)"""
        max_dbz = 0
        max_uh = 0
        max_prob = 0
        # looping through each pixel
        for pixel in storm_coords:
            # [row,col] = [lat,lon] = [y,x]
            dbz = model_refl.isel(
                west_east=pixel[1], south_north=pixel[0]).values
            uh = model_uh.isel(
                west_east=pixel[1], south_north=pixel[0]).values
            if ML == True:
                #prob = dataset['predicted_tor'].isel(lon=pixel[1], lat=pixel[0]).values
                #prob = dataset['ML_PREDICTED_TOR'].isel(west_east=pixel[1], south_north=pixel[0]).values
                prob = float(dataset['ML_PREDICTED_TOR'].isel(west_east=pixel[1], south_north=pixel[0]).values)
                if prob > max_prob:
                    max_prob = prob
            if dbz > max_dbz:
                max_dbz = dbz
            if uh > max_uh:
                max_uh = uh
        # Assigning the max values to each column after looping through the pixels
        storm_properties['MaxDBZ'][storm] = max_dbz
        storm_properties['MaxUH'][storm] = max_uh
        storm_properties['EchoMaxZ'][storm] = storm_objects[storm].intensity_max # Using the intensity_max function of the storm object to find this value
        if ML == True:
            storm_properties['ML_Max_Prob'][storm] = max_prob

        # Classification
        """Stat line:
        0: True negative
        1: False positive (False Alarm)
        2: False negative (Miss)
        3: True positive (Hit)
        """
        if storm_properties['MaxUH'][storm] >= uh_threshold and storm_properties['StormID'][storm] in confirmed_cell_list:
            storm_properties['Stat'][storm] = 3
        elif storm_properties['MaxUH'][storm] >= uh_threshold and storm_properties['StormID'][storm] not in confirmed_cell_list:
            storm_properties['Stat'][storm] = 1
        elif storm_properties['MaxUH'][storm] <= uh_threshold and storm_properties['StormID'][storm] in confirmed_cell_list:
            storm_properties['Stat'][storm] = 2
        else:
            storm_properties['Stat'][storm] = 0

        if ML == True:
            if storm_properties['ML_Max_Prob'][storm] >= tornadic_criterion and storm_properties['StormID'][storm] in confirmed_cell_list:
                storm_properties['ML_Stat'][storm] = 3
            elif storm_properties['ML_Max_Prob'][storm] >= tornadic_criterion and storm_properties['StormID'][storm] not in confirmed_cell_list:
                storm_properties['ML_Stat'][storm] = 1
            elif storm_properties['ML_Max_Prob'][storm] <= tornadic_criterion and storm_properties['StormID'][storm] in confirmed_cell_list:
                storm_properties['ML_Stat'][storm] = 2
            else:
                storm_properties['ML_Stat'][storm] = 0

    if len(tornado_reports) > len(confirmed_cell_list):
        for i in range (0, (len(tornado_reports) - len(confirmed_cell_list))):
            if ML == True:
                missed_tornado = {'StormID':(i-999),'CentCoords':(tornado_reports['Lon'][i],tornado_reports['Lat'][i]),
                            'MaxDBZ':0,'MaxUH':0,'Stat':2,'ML_Max_Prob':float(0),'ML_Stat':2}
            else:
                missed_tornado = {'StormID':(i-999),'CentCoords':(tornado_reports['Lon'][i],tornado_reports['Lat'][i]),
                            'MaxDBZ':0,'MaxUH':0,'Stat':2}
            storm_properties = storm_properties.append(missed_tornado,ignore_index=True)
    return storm_properties

# Building a function to filter the proper tornado reports
def filter_tor_reports(tor_rpts, date):
    # Finding the tornados that happened at this time and cropping in the domain to that time
    times = []
    # Selecting the reports around the time of the timestep
    timemin = date
    timemax = date+5
    tornado = []
    for i in range(0, len(tor_rpts)):
        if(tor_rpts['Time'][i] >= timemin and tor_rpts['Time'][i] <= timemax):
            tornado.append(tor_rpts.iloc[i])
    tornado = np.array(tornado)
    #print("Tor Reports: " + str(len(tornado)))
    if(len(tornado) > 0):
        tornado = pd.DataFrame({'Time': tornado[:, 0], 'F_Scale': tornado[:, 1],
                            'Location': tornado[:, 2], 'County': tornado[:, 3],
                            'State': tornado[:, 4], 'Lat': tornado[:, 5],
                            'Lon': tornado[:, 6], 'Comments': tornado[:, 7]})
    else:
        print("There were no tornadoes within this timestep")
        tornado = pd.DataFrame()
    return tornado


################################################################################################################################################
''' MISC '''
################################################################################################################################################

# Formatting and extracting time information

# Forming the date string from the WoFS files
def date_time_toString(file):
    string_parts = file.split('-')
    date_string = string_parts[0].split(
        '_')[2] + string_parts[1] + string_parts[2].split('_')[0] + ' at ' + string_parts[2].split('_')[1]
    return date_string

# Forming the date string from the WoFS files made in the 2023 runs
def date_time_toString2023(file):
    string_parts = file.split('-')
    date_string = string_parts[0].split(
        '_')[2] + string_parts[1] + string_parts[2].split('_')[0] + ' at ' + string_parts[2].split('_')[1] + string_parts[2].split('_')[2]
    return date_string

# Functions to parse the time from the datetime ftimestr, returns in the same format as the date_time_toString
def date_time_stringParse(date_string):
    string_halves = date_string.split(',')
    string_parts = string_halves[0].split('/')
    date_string = string_parts[2] + string_parts[0] + string_parts[1] + ' at ' + string_halves[1][1:]
    return date_string

# Using the above method to extract the yy/mm/dd into a string for pulling the tornado reports
def yy_mm_dd_toString(date_string):
    yymmdd = date_string.split(' ')
    yymmdd = yymmdd[0][2:]
    return yymmdd

# This function is used to turn the time into an integer that can be used to find the relevant tornado reports
def time_to_UTC_int(date_string):
    time_strings = date_string.split(' ')[2].split(":")
    time_int = time_strings[0]+time_strings[1]
    time_int = int(time_int)
    return time_int

# This function is used to turn the time into an integer that can be used to find the relevant tornado reports
def time_to_UTC_int2023(date_string):
    time_int = date_string.split(' ')[2]
    time_int = int(time_int)
    return time_int

# This function extracts the run from the file path
def time_to_UTC_from_path(path):
    path_parts = path.split('/')
    time = ''
    for i in range (0, len(path_parts)):
        if "ENS" in path_parts[i]:
            time = path_parts[i-1]
    time_int = int(time)
    return time_int

# Extract the name of the ensemble from the file name
def extract_ensemble_from_file(file):
    file_parts = file.split('_')
    ensemble = ''
    for i in range(0,len(file_parts)):
        if "ENS" in file_parts[i]:
            ensemble = file_parts[i] + '_' + file_parts[i+1] + '_' + file_parts[i+2]
            break
    return ensemble

# Extracting the Ensemble from the file path
def extract_ensemble_from_path(path):
    path_parts = path.split('/')
    ensemble = ''
    for i in range (0, len(path_parts)):
        if "ENS" in path_parts[i]:
            ensemble = path_parts[i]
    return ensemble


# Function for automating/auto-downloading the SPC tornado reports CSV 

# This function downloads the csv file for a given day formatted yymmdd
def get_tor_reports(daystring, destination_path):
    rootURL = 'https://www.spc.noaa.gov/climo/reports/'
    url = rootURL + daystring + '_rpts.html'
    # Requests URL and get response object
    response = requests.get(url)
    # Parse text obtained
    soup = BeautifulSoup(response.text, 'html.parser')
    links = soup.find_all('a')
    for link in links:
        if (daystring + '_rpts_filtered_torn.csv' in link.get('href', [])):
            # Get response object for link
            response = requests.get(rootURL + link.get('href'))
            # Write content
            csv = open(destination_path + daystring + "_rpts_filtered_torn.csv", 'wb')
            csv.write(response.content)
            csv.close()

################################################################################################################################################
''' RESULTS '''
################################################################################################################################################

""" GRAPHING """
def plot_objects(dataset, echo_tops, tor_reports, storm_labels, object_props, date_string, ensemble, destination_path, plot_dist = False):
    # Plotting some results
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 15),constrained_layout=True,
                          subplot_kw={'projection': crs})
    # First plot: Composite Reflectivity
    plot_subset_background(axes[0][0],dataset['XLONG'].min(),dataset['XLONG'].max(),dataset['XLAT'].min(),dataset['XLAT'].max())
    cf1 = dataset['COMPOSITE_REFL_10CM'].isel(Time=0).plot(ax=axes[0][0], x='XLONG', y='XLAT', 
                                                    cmap='Spectral_r', transform=ccrs.PlateCarree(), zorder = 0)
    # Second plot: Echo Top Objects
    plot_storms_subset(dataset['XLONG'].isel(Time=0),dataset['XLAT'].isel(Time=0) , dataset, storm_labels, 
                    dataset['XLONG'].min(),dataset['XLONG'].max(),dataset['XLAT'].min(),dataset['XLAT'].max(), 
                    ax=axes[0][1], storm_props=object_props)
    # Third plot: Echo Tops
    plot_subset_background(axes[1][0],dataset['XLONG'].min(),dataset['XLONG'].max(),dataset['XLAT'].min(),dataset['XLAT'].max())
    cf2 = axes[1][0].pcolormesh(dataset['XLONG'].isel(Time=0).values,dataset['XLAT'].isel(Time=0).values, 
                                echo_tops, cmap='Spectral_r', transform=ccrs.PlateCarree())
    fig.colorbar(cf2, ax = axes[1][0], label = 'Echo top heights (ft)')
    # Fourth plot: Predicted tor 
    plot_subset_background(axes[1][1],dataset['XLONG'].min(),dataset['XLONG'].max(),dataset['XLAT'].min(),dataset['XLAT'].max())
    cf1 = dataset['ML_PREDICTED_TOR'].plot(ax=axes[1][1], x='XLONG', y='XLAT', cmap='Spectral_r', transform=ccrs.PlateCarree(), zorder = 0)
    # Plotting the tornado reports if there are any at the particular timestep
    if len(tor_reports) > 0:
        axes[0][0].scatter(tor_reports['Lon'], tor_reports['Lat'], s=48,
                            marker = 'v', color = 'red', edgecolor = 'black', transform = ccrs.PlateCarree(),zorder=2)
        axes[0][1].scatter(tor_reports['Lon'], tor_reports['Lat'], s=48,
                            marker = 'v', color = 'red', edgecolor = 'black', transform = ccrs.PlateCarree(),zorder=2)
        axes[1][0].scatter(tor_reports['Lon'], tor_reports['Lat'], s=48,
                            marker = 'v', color = 'red', edgecolor = 'black', transform = ccrs.PlateCarree(),zorder=2)
        axes[1][1].scatter(tor_reports['Lon'], tor_reports['Lat'], s=48,
                            marker = 'v', color = 'red', edgecolor = 'black', transform = ccrs.PlateCarree(),zorder=2)
        # Plotting the distances between the tornado and the storm
        if plot_dist == True:
            linked_tornadoes = unique_tornadoes(linked_tornadoes)
            unlinked_tornadoes = unique_tornadoes(unlinked_tornadoes)
            print(linked_tornadoes)
            print(unlinked_tornadoes)
            linked_tornadoes = linked_tornadoes.append(unlinked_tornadoes)
            print(linked_tornadoes)
            for i in range(0,len(linked_tornadoes)):
                ccbar = 'black'
                if linked_tornadoes[i]['Distance'] < 10000:
                    ccbar = 'green'
                else:
                    ccbar = 'red'
                axes[0].plot([linked_tornadoes[i]['Tor coords'][0],linked_tornadoes[i]['Closest cell'][1]], 
                            [linked_tornadoes[i]['Tor coords'][1],linked_tornadoes[i]['Closest cell'][2]], linewidth = 5, 
                            linestyle = '-',color = ccbar, transform = ccrs.PlateCarree(), zorder = 3)
                axes[1].plot([linked_tornadoes[i]['Tor coords'][0],linked_tornadoes[i]['Closest cell'][1]], 
                            [linked_tornadoes[i]['Tor coords'][1],linked_tornadoes[i]['Closest cell'][2]], linewidth = 5,
                            linestyle = '-',color = ccbar, transform = ccrs.PlateCarree(), zorder = 3)
                axes[0].text((linked_tornadoes[i]['Tor coords'][0]+linked_tornadoes[i]['Closest cell'][1])/2,
                            (linked_tornadoes[i]['Tor coords'][1]+linked_tornadoes[i]['Closest cell'][2])/2,
                            str(linked_tornadoes[i]['Distance']),fontsize=9,color='k',transform = ccrs.PlateCarree())
                axes[1].text((linked_tornadoes[i]['Tor coords'][0]+linked_tornadoes[i]['Closest cell'][1])/2,
                            (linked_tornadoes[i]['Tor coords'][1]+linked_tornadoes[i]['Closest cell'][2])/2,
                            str(linked_tornadoes[i]['Distance']),fontsize=9,color='k',transform = ccrs.PlateCarree())
    
    yymmdd = yy_mm_dd_toString(date_string)
    fig.suptitle(date_string + " WoFS Updraft identified objects on " + ensemble , fontsize = 16)
    axes[0][0].set_title("Composite Reflectivity on WoFS")
    axes[0][1].set_title("Echo top storm objects")
    axes[1][0].set_title("Echo tops")
    axes[1][1].set_title("ML tor probabilities")
    fig.savefig(destination_path + '_Storm_objects.png')


""" TABLE BUILDING """

# This function takes the directory and builds the tables
"""
params:
    file_path: str
        This is the path of the directory that the method will loop through
    destination_path (Optional): str
        This is the path of the directory that the tables will be saved to
    plot (Optional): bool
        This determines whether or not to plot the reflectivity and objects of the current 
    plot_destination_path (Optional): str 
        This is the path of the directory that the plots will be saved to
"""
def build_tables(wofs_file_path, ml_file_path, destination_path = None, plot = False, plot_destination_path = None):
    ml_data = xr.Dataset()
    wofs_data = xr.Dataset()
    tor_reports = pd.DataFrame()
    date_string = ''
    if destination_path == None:
        destination_path = wofs_file_path
    # Looping through the files of the given directory
    for file in os.listdir(wofs_file_path):
        if file.startswith('wrfwof'):
            #print(wofs_file_path + file)
            ml_merger = False
            wofs_data = xr.open_dataset(wofs_file_path + file, decode_times = False, engine = 'netcdf4')
            #date_string = date_time_toString(file)
            date_string = date_time_toString2023(file)
            print(date_string)
            yymmdd = yy_mm_dd_toString(date_string)
            #time_int = time_to_UTC_int(date_string)
            time_int = time_to_UTC_int2023(date_string)
            if(time_int - 500 < 0):
                yymmdd = str(int(yymmdd) - 1)
            #print(yymmdd)
            run_time = time_to_UTC_from_path(wofs_file_path)
            
            # This adjusts the time beyond 00z to be greater than 23z
            if time_int < 1000:
                time_int += 2400
            # This is to filter the files analyzes such that only ONE HOUR after the run time is analyzed
            if(time_int - run_time) < 100 and (time_int - run_time) > 0:
                if time_int > 2400:
                    time_int -= 2400
                # Printing/Checking the file and time and stuff
                print("File: " + file + "\n" + "YYMMDD: " + str(yymmdd) + "\nTime int: " + str(time_int))
                try:
                    tor_reports = pd.read_csv(destination_path + yymmdd + '_rpts_filtered_torn.csv')
                except(FileNotFoundError):
                    # Downloading the torando reports for the given day
                    get_tor_reports(yymmdd,destination_path)
                    tor_reports = pd.read_csv(destination_path + yymmdd + '_rpts_filtered_torn.csv')
                # Filtering the tornado reports for the current timestep of the file
                tor_reports = filter_tor_reports(tor_reports, time_int)
                # Retrieving the ML prediction data for the given day/time/ensemble (if applicable)
                try:
                    ml_data = xr.open_dataset(ml_file_path + file, engine = 'netcdf4')
                    wofs_data = wofs_data.merge(ml_data)
                    ml_merger = True
                except(FileNotFoundError):
                    pass
                # Going through object identification
                #print(wofs_data)
                echo_tops = get_echotop(wofs_file_path + file)
                '''
                # Storm identification based on composite reflectivity
                WoFS_storm_labels, WoFS_object_props = monte_python.label(input_data = wofs_data['COMPOSITE_REFL_10CM'].isel(Time=0), 
                        method ='watershed', 
                        return_object_properties=True, 
                        params = {'min_thresh':reflectivity_threshold,
                                    'max_thresh':80,
                                    'data_increment':5,
                                    'area_threshold': 200,
                                    'dist_btw_objects': 50})
                
                # Storm identification based on vertical updraft speed
                WoFS_storm_labels, WoFS_object_props = monte_python.label(  input_data = wofs_data['W_UP_MAX'].isel(Time=0), 
                        method ='watershed', 
                        return_object_properties=True, 
                        params = {'min_thresh':updraft_threshold,
                                    'max_thresh':50,
                                    'data_increment':1,
                                    'area_threshold': 20,
                                    'dist_btw_objects': 20})
                '''
                # Storm identification based on echo top heights
                WoFS_storm_labels, WoFS_object_props = monte_python.label(  input_data = echo_tops, 
                        method ='watershed', 
                        return_object_properties=True, 
                        params = {'min_thresh':echo_height_min_threshold,
                                    'max_thresh':echo_height_max_threshold,
                                    'data_increment':50,
                                    'area_threshold': 100,
                                    'dist_btw_objects': 50})
                # Generating the tables (See method for more details)
                if ml_merger == True:
                    storm_properties = find_storm_properties(WoFS_object_props, wofs_data, 
                                wofs_data['COMPOSITE_REFL_10CM'].isel(Time=0), wofs_data['UP_HELI_MAX'].isel(Time=0), 
                                echo_tops, tor_reports, ML=True)
                else:
                    storm_properties = find_storm_properties(WoFS_object_props, wofs_data, 
                                wofs_data['COMPOSITE_REFL_10CM'].isel(Time=0), wofs_data['UP_HELI_MAX'].isel(Time=0), 
                                echo_tops, tor_reports)

                ens_mem = extract_ensemble_from_path(wofs_file_path)
                #print(ens_mem)
                #print(storm_properties)
                # Saving the tables to excel spreadsheets to a given file
                try:
                    with pd.ExcelWriter(destination_path + yymmdd + '_' + 
                                str(time_int).zfill(4) + '_properties.xlsx',mode="a", engine="openpyxl") as writer:                    
                        storm_properties.to_excel(writer, sheet_name=ens_mem, index=False)
                except(FileNotFoundError):
                    with pd.ExcelWriter(destination_path + yymmdd + '_' + 
                                str(time_int).zfill(4) + '_properties.xlsx',mode="w", engine="openpyxl") as writer:
                        storm_properties.to_excel(writer, sheet_name=ens_mem, index=False)
                except(ValueError):
                    # Should override/rewrite the current table if already exists 
                    with pd.ExcelWriter(destination_path + yymmdd + '_' + 
                                str(time_int).zfill(4) + '_properties.xlsx',mode="w", engine="openpyxl") as writer:
                        storm_properties.to_excel(writer, sheet_name=ens_mem, index=False)
                    #pass
                if plot == True:
                    if plot_destination_path != None:
                        save_fig = str(plot_destination_path + yymmdd + '_' + ens_mem +'_' + str(time_int).zfill(4))
                    else:
                        save_fig = str(destination_path + yymmdd + '_' + ens_mem +'_' + str(time_int).zfill(4))
                    plot_objects(wofs_data, echo_tops, tor_reports, WoFS_storm_labels, WoFS_object_props, date_string, ens_mem, save_fig)

def test_function(file_path):
    for file in os.listdir(file_path):
        print("We access the file")
        print(file)
        if file.startswith('wrfwof'):
            print("This actually works " + file)

# Runs the build_tables logic on a specified time
### time: int - The desired time (int UTC) 
def build_specified_time(wofs_file_path, ml_file_path, time, destination_path = None, plot = False, plot_destination_path = None):
    for file in os.listdir(wofs_file_path):
        if file.startswith('wrfwof'):
            ml_merger = False
            wofs_data = xr.open_dataset(wofs_file_path + file)
            date_string = date_time_toString(file)
            #print(date_string)
            yymmdd = yy_mm_dd_toString(date_string)
            time_int = time_to_UTC_int(date_string)
            if(time_int - 500 < 0):
                yymmdd = str(int(yymmdd) - 1)
            if(time_int == time):
                get_tor_reports(yymmdd,destination_path)
                tor_reports = pd.read_csv(destination_path + yymmdd + '_rpts_filtered_torn.csv')
                # Filtering the tornado reports for the current timestep of the file
                tor_reports = filter_tor_reports(tor_reports, time_int)
            # Retrieving the ML prediction data for the given day/time/ensemble (if applicable)
            try:
                ml_data = xr.open_dataset(ml_file_path + file)
                wofs_data = wofs_data.merge(ml_data)
                ml_merger = True
            except(FileNotFoundError):
                pass
            # Computing the echo tops
            echo_tops = get_echotop(wofs_file_path + file)
            # Performing the storm object identification
            WoFS_storm_labels, WoFS_object_props = monte_python.label(  input_data = echo_tops, 
                    method ='watershed', 
                    return_object_properties=True, 
                    params = {'min_thresh':echo_height_min_threshold,
                                'max_thresh':echo_height_max_threshold,
                                'data_increment':50,
                                'area_threshold': 100,
                                'dist_btw_objects': 50})
            # Generating the tables (See method for more details)
            if ml_merger == True:
                storm_properties = find_storm_properties(WoFS_object_props, wofs_data, 
                            wofs_data['COMPOSITE_REFL_10CM'].isel(Time=0), wofs_data['UP_HELI_MAX'].isel(Time=0), 
                            wofs_data['W_UP_MAX'].isel(Time=0), tor_reports, ML=True)
            else:
                storm_properties = find_storm_properties(WoFS_object_props, wofs_data, 
                            wofs_data['COMPOSITE_REFL_10CM'].isel(Time=0), wofs_data['UP_HELI_MAX'].isel(Time=0), 
                            wofs_data['W_UP_MAX'].isel(Time=0), tor_reports)

            ens_mem = extract_ensemble_from_path(wofs_file_path)
            #print(ens_mem)
            #print(storm_properties)
            # Saving the tables to excel spreadsheets to a given file
            try:
                with pd.ExcelWriter(destination_path + yymmdd + '_' + 
                            str(time_int).zfill(4) + '_properties.xlsx',mode="a", engine="openpyxl") as writer:                    
                    storm_properties.to_excel(writer, sheet_name=ens_mem, index=False)
            except(FileNotFoundError):
                with pd.ExcelWriter(destination_path + yymmdd + '_' + 
                            str(time_int).zfill(4) + '_properties.xlsx',mode="w", engine="openpyxl") as writer:
                    storm_properties.to_excel(writer, sheet_name=ens_mem, index=False)
            except(ValueError):
                # Should override/rewrite the current table if already exists? 
                with pd.ExcelWriter(destination_path + yymmdd + '_' + 
                            str(time_int).zfill(4) + '_properties.xlsx',mode="w", engine="openpyxl") as writer:
                    storm_properties.to_excel(writer, sheet_name=ens_mem, index=False)
                #pass
            if plot == True:
                save_fig = str(destination_path + yymmdd + '_' + ens_mem +'_' + str(time_int).zfill(4))
                plot_objects(wofs_data, tor_reports, WoFS_storm_labels, WoFS_object_props, date_string, ens_mem, save_fig)
