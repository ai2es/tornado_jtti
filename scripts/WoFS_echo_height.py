# WoFS_echo_height.py
# Python 3 Script
# Author: Nate Snook (CAPS) -- 12 Jan. 2021
# Description:  This script will read data from a single WoFS file and output a listing of variables.
#               It then plots a 2D slice of one field using Cartopy and matplotlib.

#--------------------------------------------#
#   Import from standard libraries           
#--------------------------------------------#
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import numpy as np
import sys

#--------------------------------------------#
#   Input sources
#--------------------------------------------#
WoFS_file = '/ai-hail/data/WoFS/raw_wrf_output/ENS_MEM_13/wrfwof_d01_2019-05-25_03:00:00'

#--------------------------------------------#
#   Output filenames
#--------------------------------------------#
img_filename = 'sample_echotop.png'

#--------------------------------------------#
#   Read WoFS data and list info.
#--------------------------------------------#
#Open WoFS File
print("WoFS file is: " + str(WoFS_file))
print("   ...reading data...")
WoFS_data = xr.open_dataset(WoFS_file)
print(WoFS_data.keys())
print("---------------------------------------------------------")
print("Showing data for reflectivity (REFL_10CM):")
print(WoFS_data['REFL_10CM'])
print("Shape of REFL_10CM array: " + str(np.shape(WoFS_data['REFL_10CM'])))
print("Minimum value: " + str(np.min(WoFS_data['REFL_10CM'])) + "  |  Maximum value: " + str(np.max(WoFS_data['REFL_10CM'])))

#--------------------------------------------#
#   Determine k-coordinate of echo-top
#--------------------------------------------#
#Note:  If no echoes >= 40 dBZ are present, echo top will be set to k = np.NaN
#       If echoes >= 40 dBZ are present, echo top will be the highest vertical level with echo >= 40 dBZ.

#Define nz, ny, and nx
nz = np.shape(WoFS_data['REFL_10CM'])[1]  #Number of gridpoints in the y-direction (north-south)
ny = np.shape(WoFS_data['REFL_10CM'])[2]  #Number of gridpoints in the y-direction (north-south)
nx = np.shape(WoFS_data['REFL_10CM'])[3]  #Number of gridpoints in the x-direction (east-west)

#Define an array to store echo top level (vertical level number)
echotop = np.zeros((ny, nx))  

#Create an array from the REFL_10CM array where the value in each grid cell is:
#  0 if reflectivity is < 40 dBZ
#  1 if reflectivity is >= 40 dBZ
refl40_mask = np.where(WoFS_data['REFL_10CM'][0,:,:,:] >= 40.0, 1, 0)

print("refl40_mask dimensions: " + str(refl40_mask.shape))
print("Minimum value: " + str(np.min(refl40_mask)) + "  |  Maximum value: " + str(np.max(refl40_mask)))
print("Average value: " + str(np.average(refl40_mask)))

#Loop through each column to find echo top height
for j in np.arange(0, ny, 1):
   for i in np.arange(0, nx, 1):
      column = refl40_mask[:,j,i] #select current column
#      print(str(column))
      if (np.sum(column) == 0):     #if there are no values of reflectivity above 40 dBZ...
         echotop[j, i] = np.NaN   #...then set echotop to the no-echo value (np.NaN)...
#         print('Found empty column at [' + str(j) + ',' + str(i) + ']')
         continue                 #...and move on to the next column.
      for k in np.arange(nz-1, 1, -1):  #Loop DOWNWARD over the vertical (k) dimension
         if (column[k] > 0):        #find the highest level at which revlectivity >= 40 dBZ...
            echotop[j, i] = k     #...set echotop to the index of that level...
#            print('Found echotop of ' + str(k) + ' at [' + str(j) + ',' + str(i) + ']')
            break                 #...and move on to the next column

print("...Echo Top (40dBZ) calculated successfully!")
print("Minimum value: " + str(np.nanmin(echotop)) + "  |  Maximum value: " + str(np.nanmax(echotop)))

#Put echotop data into an xarray DataArray for convenient plotting
echotop = xr.DataArray(echotop, 
                       dims=['y', 'x'],
                       coords=dict(lat=(['y', 'x'], WoFS_data["XLAT"][0]), lon=(['y', 'x'], WoFS_data["XLONG"][0])), 
                       attrs=dict(description="echo-top height", units="vertical model level")
                       )

#--------------------------------------------#
#   Plot echo-top model level using Cartopy
#--------------------------------------------#
fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-97.5))
#In the following lines, colors are defined using a six-digit hexcode (same as for HTML)
ax.add_feature(cfeature.STATES, edgecolor='#333333', linewidth=1.5)
ax.add_feature(cfeature.BORDERS, edgecolor='#000000', linewidth=2.5)
ax.add_feature(cfeature.COASTLINE, edgecolor='#000000', linewidth=2.5)
ax.set_extent([-105, -95, 29, 39], crs=ccrs.PlateCarree())

#Define which data to plot
plot_data = echotop
print('Dimensions of data to plot are: ' + str(plot_data.shape))
print('Minimum: ' + str(np.min(plot_data)) + '  |  Maximum: ' + str(np.max(plot_data)))
PX = xr.plot.pcolormesh(echotop, x='lon', y='lat', ax=ax, transform=ccrs.PlateCarree(), colors='gist_ncar', levels=50)
plt.title('Echo-top vertical model index')

#Save figure to a file
print('Saving image to ' + str(img_filename))
plt.savefig(img_filename, dpi=200)
