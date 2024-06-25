# WoFS_demo.py
# Python 3 Script
# Author: Nate Snook (CAPS) -- 9 Dec. 2020
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

#--------------------------------------------#
#   Input sources
#--------------------------------------------#
WoFS_file = '/ai-hail/data/WoFS/raw_wrf_output/ENS_MEM_13/wrfwof_d01_2019-05-25_03:00:00'

#--------------------------------------------#
#   Output filenames
#--------------------------------------------#
img_filename = 'sample_WoFS.png'

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

#--------------------------------------------#
#   Plot reflectivity data using Cartopy
#--------------------------------------------#
fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-97.5))
#In the following lines, colors are defined using a six-digit hexcode (same as for HTML)
ax.add_feature(cfeature.STATES, edgecolor='#333333', linewidth=1.5)
ax.add_feature(cfeature.BORDERS, edgecolor='#000000', linewidth=2.5)
ax.add_feature(cfeature.COASTLINE, edgecolor='#000000', linewidth=2.5)
ax.set_extent([-105, -95, 29, 39], crs=ccrs.PlateCarree())

#Define which data to plot
plot_data = WoFS_data['REFL_10CM'][0,3,:,:]
print('Dimensions of data to plot are: ' + str(plot_data.shape))
print('Minimum: ' + str(np.min(plot_data)) + '  |  Maximum: ' + str(np.max(plot_data)))
PX = xr.plot.pcolormesh(plot_data, x='XLONG', y='XLAT', ax=ax, transform=ccrs.PlateCarree(), colors='gist_ncar', levels=100)

print('Saving image to ' + str(img_filename))
plt.savefig(img_filename, dpi=200)
