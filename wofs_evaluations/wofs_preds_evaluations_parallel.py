"""
author Monique Shotande

Parallelized Pixel-wise Evaluation of WoFS predictions.
Generates a pandas dataframe of all the files for convenience
when selecting all ensembles for specific forecast time, 
initialization time, and date.

Mean and median UH and ML probabilities can be computed across the 
ensembles. Results for UH can be optionally computed with the 
argument flag, --uh_compute.

Outputs performance diagram and csv files with the confusion/contingency
matrix and other useful metrics for each selected threshold

Example execution in `wofs_preds_evaluations.sh`

          cooresponding index range
min00-20 :  0,  5
min20-60 :  5, 13
min60-90 : 13, 19
min90-180: 19, 37

"""

import re, os, sys, stat, errno, glob, argparse
from datetime import timedelta, datetime, date, time
from dateutil.parser import parse as parse_date
from time import perf_counter

import xarray as xr
print("xr version", xr.__version__)
import numpy as np
print("np version", np.__version__)
import pandas as pd
print("pd version", pd.__version__)
from scipy import spatial, stats
from scipy.ndimage import grey_dilation
from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn.metrics import confusion_matrix

#import netCDF4
#print("netCDF4 version", netCDF4.__version__)
from netCDF4 import num2date #, Dataset, date2num

import threading
from concurrent.futures import (ThreadPoolExecutor, ProcessPoolExecutor, wait, 
                                as_completed, ALL_COMPLETED, FIRST_COMPLETED,
                                FIRST_EXCEPTION)
from multiprocessing import (Manager, Lock, shared_memory, Pipe, get_context)
from tqdm.auto import tqdm

import gc


# Working directory expected to be tornado_jtti/
sys.path.append("lydia_scripts")
#from wofs_raw_predictions import load_wofs_file
from wofs_ensemble_predictions import extract_datetime, select_files
from scripts_tensorboard.unet_hypermodel import plot_reliabilty_curve, plot_csi
from scripts_tensorboard.unet_hypermodel import compute_csi, compute_sr, compute_pod
from scripts_tensorboard.unet_hypermodel import contingency_curves, plot_roc, plot_prc


import matplotlib.pyplot as plt
import matplotlib as mpl
plt.rcParams.update({"text.usetex": True})
#https://matplotlib.org/stable/tutorials/introductory/customizing.html
plt.rcParams.update({"text.usetex": True})
mpl.rcParams['font.size'] = 18 # default font size
mpl.rcParams['axes.labelsize'] = 18
#mpl.rcParams['axes.labelpad'] = 5
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
#figure.titlesize figure.labelsize axes.titlesize font.size
cb_args = {'rotation': 270, 'labelpad': 25}




def create_argsparser(args_list=None):
    ''' 
    Create command line arguments parser

    @param args_list: list of strings with command line arguments to override
            any received arguments. default value is None and args are not 
            overridden. not used in production. mostly used for unit testing.
            Element at index 0 should either be the empty string or the name of
            the python script

    @return: the argument parser object
    '''
    if not args_list is None:
        sys.argv = args_list
        
    parser = argparse.ArgumentParser(description='WoFS ensemble evals', epilog='AI2ES')

    # TODO: mutually exclusive args
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument('--dir_wofs_preds', type=str, #required=True, 
        help='Directory location of WoFS predictions, with subdirectories for \
        the initialization times, which each contain subdirectories for all \
        ensembles, which contains the predictions for each forecast time. \
        Example directory: wofs_preds/2023/20230602. Example directory \
        structure: wofs_preds/2023/20230602/1900/ENS_MEM_13/wrfwof_d01_2023-06-02_20_30_00_predictions.nc')
    grp.add_argument('--loc_files_list', type=str, #required=True, 
        help='Path to the csv file that lists all the predictions files')

    parser.add_argument('--loc_storm_report', type=str, required=True, 
        help='List of paths to the csv file that lists the storm reports')

    parser.add_argument('--out_dir', type=str, required=True, 
        help='Directory to store the results. By default, stored in loc_init directory')
    
    parser.add_argument('--year', type=str, required=True,
                        help='Year to evaluate the model on')
    parser.add_argument('--date0', type=str, required=True,
                       help='Start date, of the format YYYY-mm-dd, in the corresponding storm report file. ex: 2019-03-30')
    parser.add_argument('--date1', type=str, required=True,
                       help='End date, of the format YYYY-mm-dd, in the corresponding storm report file. ex: 2019-06-03')
    
    parser.add_argument('--forecast_duration', type=int, nargs='+', default=[36],
                       help='List of at most 2 integers. The indices of the \
                       forecast range to evaluate on. For example, for the \
                       forecast range 0 to 20 min, the indices are 0 and 5. ')

    parser.add_argument('--uh_compute', action='store_true',
        help='Whether to compute performance results for the UH')
    parser.add_argument('--uh_thres_list', type=float, nargs='+',
        help='List of UH threshold values')

    # Skip forecast time vs skip day
    parser.add_argument('--skip_clearday', action='store_true',
        help='Whether to skip days where there are no tornadoes')
    parser.add_argument('--skip_cleartime', action='store_true',
        help='Whether to skip forecast times where there are no tornadoes')
    
    parser.add_argument('--gridsize', type=float, default=3,
        help='WoFS gridsize in kilometers. Default is 3')
    
    parser.add_argument('--thres_dist', type=float, default=10,
        help='Threshold distance, in kilometers, to use for the storm mask construction')
    
    parser.add_argument('--thres_time', type=float, default=5,
        help='Threshold time, in minutes, to use for the storm mask construction')
    
    parser.add_argument('--stat', type=str, default='median',
        help='Stat to use for encorporating the ensembles into the evaluations. median (default) | mean')
    
    parser.add_argument('--ml_probabs_norm', action='store_true',
        help='Max-normalize ML probabilities ')
    parser.add_argument('--ml_probabs_dilation', type=int, default=0,
        help='Dilate ML probabilities to the given number of pixels') 
    parser.add_argument('--nthresholds', type=int, default=41,
        help='Number of probability thresholds to use for the performance diagram') #
    
    parser.add_argument('--model_name', type=str, default='',
        help='(optional) Name of model predictions came from. Name is prepended to the output files')

    parser.add_argument('-w', '--write', type=int, default=0,
        help='Write/save data and/or figures. Set to 0 to save nothing')
    parser.add_argument('--write_storm_masks', action='store_true',
        help='Write/save the masks generated from the storm reports')
    parser.add_argument('-d', '--dry_run', action='store_true',
        help='For testing and debugging')
    parser.add_argument('-t', '--test', action='store_true',
        help='Use test args')

    return parser

def parse_args(args_list=None):
    '''
    Create and parse the command line args parser

    @param args_list: list of strings with command line arguments to override
            any received arguments. default value is None and args are not 
            overridden. not used in production. mostly used for unit testing.
            Element at index 0 should either be the empty string or the name of
            the python script

    @return: the parsed arguments object
    '''
    parser = create_argsparser(args_list=args_list)
    args = parser.parse_args()
    return args


def _add_int_to_date(times_list, date, DB=False):
    ''' 
    Helper method. add int times in the list (of the form HHMM) to the date
    Example: 
        time = 1930
        date = datetime.datetime(year=2019, month=4, day=30)
        datetime = date + time --> datetime.datetime(year=2019, month=4, 
                                                    day=30, hr=19, min=30)
    
    Parameters
    -------------
    times_list: list ints of the 4-digit times to convert into dates
    date: datetime. date to add the times to

    Returns
    -------------
    _datetimes: 
    '''
    # Convert the time of day into a datetime
    _datetimes = np.empty(times_list.shape, dtype=object)

    for i, t in enumerate(times_list):
        hrs = t // 100
        mins = t % 100

        if DB: print(t, hrs, mins)

        dtime = timedelta(hours=hrs, minutes=mins)
        _datetimes[i] = date + dtime

    return _datetimes

def load_storm_report(fnpath, year, month, day):
    '''
    Load the storm report file for the specified date
    Parameters
    ------------
    fnpath: str path to storm report file
    year: int
    month: int
    day: int

    Returns
    ------------
    df_storm_report: pd.DataFrame with the storm reports
    date_storm: datetime object for the storm report
    '''
    df_storm_report = pd.read_csv(fnpath)

    df_storm_report.sort_values(by=['Time'], inplace=True, ignore_index=True)

    date_storm = datetime(year=year, month=month, day=day)
    _time = df_storm_report['Time']
    df_storm_report['DateTime'] = _add_int_to_date(_time, date_storm)

    # reorder columns such that DateTime is the 2nd col. # TODO: might be a better way to do this
    cols = list(df_storm_report.columns)
    cols.remove('DateTime')
    cols.insert(1, 'DateTime') # insert column at index 1
    df_storm_report = df_storm_report[cols]

    return df_storm_report, date_storm

def load_storm_report_large(fnpath, date0, date1):
    '''
    Load the storm report file for the specified date range
    Parameters
    ------------
    fnpath: str path to storm report file
    date0: str of the format YYYY-mm-dd. ex: 2019-03-30
    date1: str of the format YYYY-mm-dd. ex: 2019-06-01

    Returns
    ------------
    df_storm_report: pd.DataFrame with the storm reports
    times_storms: numpy list of python datetime objects for each storm reported
    '''
    df_storm_report = pd.read_csv(fnpath)
    df_storm_report.sort_values(by=['DateTime'], inplace=True, ignore_index=True)

    df_storm_report.set_index('DateTime', inplace=True)
    df_storm_report = df_storm_report[date0:date1]
    
    # Convert npdate64 values into datetime objects
    # TODO (maybe): speed up option using existing npdate64 values
    times_storms = np.array([d.to_pydatetime() for d in pd.to_datetime(df_storm_report.index.values)])

    # TODO: Insert rows for times in range of start_datetime_sec to end_datetime_sec

    return df_storm_report, times_storms

def load_storm_report_large_v0(fnpath, date0, date1):
    ''' OLD DEPREICATED
    for storm reports file /ourdisk/hpc/ai2es/tornado/stormreports/processed/tornado_reports_2019.csv
    Load the storm report file for the specified date range
    Parameters
    ------------
    fnpath: str path to storm report file
    date0: str of the format YYYY-mm-dd. ex: 2019-03-30
    date1: str of the format YYYY-mm-dd. ex: 2019-06-01

    Returns
    ------------
    df_storm_report: pd.DataFrame with the storm reports
    times_storms: numpy list of python datetime objects for each storm reported
    times_storms_end
    None date_storm: datetime object for the storm report
    '''
    df_storm_report = pd.read_csv(fnpath)
    df_storm_report.sort_values(by=['start_time_unix_sec'], inplace=True, ignore_index=True)

    # Rename columns for convenience
    df_storm_report.rename(columns={#'start_time_unix_sec': 'DateTime', 
                                    'start_latitude_deg': 'Lat',
                                    'start_longitude_deg': 'Lon'}, inplace=True)

    # Convert unix time to datetime
    df_storm_report['DateTime'] = pd.to_datetime(df_storm_report['start_time_unix_sec'], unit='s')
    df_storm_report['DateTime_end'] = pd.to_datetime(df_storm_report['end_time_unix_sec'], unit='s')

    df_storm_report.set_index('DateTime', inplace=True)
    df_storm_report = df_storm_report[date0:date1]
    
    # Convert npdate64 values into datetime objects
    # TODO (maybe): speed up option using existing npdate64 values
    times_storms = np.array([d.to_pydatetime() for d in pd.to_datetime(df_storm_report.index.values)])
    times_storms_end = np.array([d.to_pydatetime() for d in pd.to_datetime(df_storm_report['DateTime_end'].values)])

    # TODO: Insert rows for times in range of start_datetime_sec to end_datetime_sec

    return df_storm_report, times_storms, times_storms_end


def load_preds(dir_root, year='*', date='*', init_time='*', ens='*', 
               fnpath='', **kwargs):
    """
    Load WoFS ml predictions from directory 
    Parameters
    ----------
    dir_root: top level directory
    year: (optional) string or int for the year directory. if excluded all 
            years are loaded.
    date: (optional) string or int for the date directory. if excluded all 
            dates are loaded
    init_time: (optional) string or int for the initialization time  
            directory. if excluded all init_times are loaded
    ens: (optional) string for the ensemble directory. if excluded 
            all ensembles are loaded
    fnpath: (optional) str full path and filename to a specific file. all other 
            arguments are ignored if this argument is not the empty string.
            or can be a list of strings of filepaths
    **kwargs: dictonary of keyword arguments for xr.load_dataset

    Returns
    ----------
    xr.Dataset
    """
    if fnpath == '':
        dirpath = os.path.join(dir_root, str(year), str(date), str(init_time), ens)
        #files_list = os.listdir(dirpath)
        files_list = sorted(glob.glob(dirpath))
        ds = xr.open_mfdataset(files_list, **kwargs)
    elif isinstance(fnpath, str):
        ds = xr.load_dataset(fnpath, **kwargs)
    elif isinstance(fnpath, (list, tuple)):
        ds = xr.open_mfdataset(fnpath, **kwargs)
    else: raise ValueError(f'fnpath should either be the empty string, a string or a list of strings. but was {fnpath}')
    return ds

def generate_ensemble_files_list(dir_preds, date, itime, ens, save_path=None, 
                                 df_index=False, **kwargs):
    ''' 
    Generate pd.DataFrame of files for the single ensemble member
    column headers: 'run_date', 'init_time', 'ensemble_member', 
                    'forecast_time', 'filename_path'
    Parameters:
    -------------
    dir_preds: str top level directory containing the files containing the 
            subdirectories for the initialization times
    date: str date 
    itime: str for the initialization time
    ens: string for the ensemble directory
    save_path: str filename path to store the csv containing the list of
            wofs preds file names and meta details
    df_index: bool whether to save the csv with the index
    **kwargs: keyword args for wofs_ensemble_predictions.extract_datetime()

    Returns:
    --------------
    df_all_files: pandas dataframe with the list of all the files. with the
            run_date (date of the run), init_time (initialization time), 
            ensemble_member, forecast_time (time of the wofs forecast), 
            filename_path
    '''
    all_files = [] 

    ens_files = sorted(glob.glob(os.path.join(dir_preds, date, itime, ens, '*')))

    for fnpath in ens_files:
        #fnpath_parts = fnpath.split('/')

        forecast_time, dt_obj = extract_datetime(fnpath, **kwargs)
        if dt_obj is None: continue
        #method='re', fmt_dt=fmt_dt, fmt_re=fmt_re)
        #run_date = datetime.strptime(fnpath, '%Y%m%d')

        entry = {'run_date': date, 'init_time': itime, #fnpath_parts[-1]
                 'ensemble_member': ens, 'forecast_time': forecast_time,
                 'filename_path': fnpath}
        all_files.append(entry)

    df_all_files = pd.DataFrame(all_files)
    df_all_files.sort_values(['run_date', 'init_time', 'ensemble_member', 
                              'forecast_time', 'filename_path'], inplace=True)
    
    if isinstance(save_path, str):
        df_all_files.to_csv(save_path, index=df_index)
    return df_all_files

def generate_ensemble_files_list_all(dir_preds, date, itime, ens_names=[],
                                     save_path=None, df_index=False,
                                     **kwargs):
    '''
    Generate pd.DataFrame listing all file names and paths for ALL ensembles
        column headers: 'run_date', 'init_time', 'ensemble_member', 
                        'forecast_time', 'filename_path'
    Parameters:
    -------------
    dir_preds: str top level directory containing the files containing the 
            subdirectories for the initialization times
    date: str date 
    itime: str for the initialization time
    ens_names: (optional) list of strings of the ensemble directory names
    save_path: str filename path to store the csv containing the list of
            wofs preds file names and meta details
    df_index: bool whether to save the csv with the index
    **kwargs: keyword args for wofs_ensemble_predictions.extract_datetime()

    Returns:
    --------------
    df_all_ens: pandas dataframe with the list of all the files, for all 
            ensembles. with the run_date (date of the run), init_time 
            (initialization time), ensemble_member, forecast_time (time of the 
            wofs forecast), filename_path
    '''
    if ens_names == []:
        dir_ens = os.path.join(dir_preds, str(date), str(itime))
        ens_names = sorted(os.listdir(dir_ens))

    df_all_ens = pd.DataFrame([], columns=['run_date', 'init_time', 
                                           'ensemble_member', 'forecast_time', 
                                           'filename_path'])
    for ens_ in ens_names:
        df_ = generate_ensemble_files_list(dir_preds, date, str(itime), ens_, 
                                           save_path=save_path, 
                                           df_index=df_index, **kwargs)
        df_all_ens = pd.concat([df_all_ens, df_], ignore_index=True, 
                               verify_integrity=True)

    df_all_ens.sort_values(['run_date', 'init_time', 'ensemble_member', 
                            'forecast_time', 'filename_path'], inplace=True)
    return df_all_ens

def generate_init_times_files_list_all(dir_preds, date, save_path=None, 
                                       df_index=False, **kwargs):
    '''
    Generate pd.DataFrame of all files for all init times, ensembles, forecast
    times for the date. This makes it easier to select files based on
    combinations of init times, ensemble members, and foredast times
    Parameters:
    -------------
    dir_preds: str top level directory containing the files containing the 
            subdirectories for the initialization times
    save_path: str filename path to store the csv containing the list of
            wofs preds file names and meta details
    df_index: bool whether to save the csv with the index
    **kwargs: keyword args for wofs_ensemble_predictions.extract_datetime()

    Returns:
    --------------
    df_all_inits: pandas dataframe with the list of all the files, for all 
            ensembles, for all the init times. with the run_date (date of the run), 
            init_time (initialization time), ensemble_member, forecast_time (time 
            of the wofs forecast), filename_path
    '''
    dir_date = os.path.join(dir_preds, str(date))
    init_times = sorted(os.listdir(dir_date))
    init_times = [re.fullmatch(r'\d{4}', _dirname)[0] for _dirname in init_times if re.fullmatch(r'\d{4}', _dirname) is not None]
    
    df_all_inits = pd.DataFrame([], columns=['run_date', 'init_time', 
                                             'ensemble_member', 'forecast_time', 
                                             'filename_path'])
    for itime in init_times:
        df_ = generate_ensemble_files_list_all(dir_preds, date, itime, 
                                               save_path=save_path, 
                                               df_index=df_index, **kwargs)
        df_all_inits = pd.concat([df_all_inits, df_], ignore_index=True, 
                                 verify_integrity=True)

    df_all_inits.sort_values(['run_date', 'ensemble_member', 'forecast_time', 
                              'init_time', 'filename_path'], inplace=True)
    #0['run_date', 'init_time', 'ensemble_member', 'forecast_time', 'filename_path']
    #x['run_date', 'forecast_time', 'init_time', 'ensemble_member', 'filename_path']

    # For instances where there are wrfout and wrfwof files, drop the first (i.e. the wrfout)
    df_all_inits.drop_duplicates(subset=['run_date', 'ensemble_member', 
                                         'forecast_time', 'init_time'], 
                                         inplace=True, keep='last', ignore_index=True)

    return df_all_inits


"""
Generating Labels
"""
def create_storm_mask(lats, lons, dtime, lats_storms, lons_storms, times_storms, 
                      gridsize, thres_dist=10, thres_time=5, eps=1e-3,
                      use_ball=True, sort_time=False, tree=None, DB=False, 
                      kdworkers=-1, times_storms_end=None):
    ''' 
    Uses kD tree to perform nearest neighbor search to determine wofs grid cells
    the storm reports reside. 

    Parameters
    -----------
    lats: full domain grid for a single datetime
    lons: full domain grid for a single datetime
    dtime: datetime object for the data (i.e. forecast time)
    lats_storm: list of latitude coordinate for storm reports
    lons_storm: list of longitude coordinate for storm reports
    times_storms: list of times for the storm reports start times
    gridsize: length in kilometers of the grids
    thres_dist: int in kilometers
    thres_time: int or float in minutes. time before and after storm, and time 
            before and after the forecast time (dtime) to consider as an overlap.
            storm_time +/- thres_time
            forecast_time +/- thres_time
    eps: epsilon
    use_ball: 
    sort_time: bool. If true, sort the storm reports by time
    tree: scipy.spatial.cKDTree object. if None create tree object. otherwise 
            use the provided KDTree
    kdworkers: 
    times_storms_end: 

    Returns
    ----------
    mask: 2D np array shape same as the original full lat/lon grid
    mask_reports_by_time: 1D np array shape same as the list of storm reports
    tree: spatial kdtree object
    '''
    mask_reports_by_time = np.ones(times_storms.shape, bool)

    if dtime is not None:
        # Determine storm reports within the specified time range
        delta_time = timedelta(minutes=thres_time)

        # Time range around storm report
        times_storms_start = times_storms - delta_time
        times_storms_end = times_storms + delta_time
        # Time range around forecast time
        time_forecast_start = dtime - delta_time
        time_forecast_ends = dtime + delta_time
        
        # Check time range of storm report overlaps with time range of forecast
        mask_reports_by_start_time = np.logical_and(times_storms_start <= time_forecast_start, times_storms_end >= time_forecast_start)
        mask_reports_by_end_time = np.logical_and(times_storms_start <= time_forecast_ends, times_storms_end >= time_forecast_ends)
        mask_reports_by_time = np.logical_or(mask_reports_by_start_time, mask_reports_by_end_time)
        #mask_reports_by_time = np.logical_and(times_storms >= time_range[0], times_storms <= time_range[1])
        if times_storms_end is not None:
            mask_end_time = np.logical_and(times_storms_end >= time_forecast_start, times_storms_end <= time_forecast_ends)
            mask_reports_by_time = np.logical_or(mask_reports_by_time, mask_end_time)

    # Compute the distance in terms of the number of grid cells
    thres_ngrids = np.max([thres_dist / gridsize, eps]) 

    # Compute mean difference between latitude and longitude grid lines
    delta_lat = np.diff(lats, n=1, axis=0).mean()
    delta_lon = np.diff(lons, n=1, axis=1).mean()
    if DB: print(f"Delta (degrees per grid): lat={delta_lat}. lon={delta_lon}")

    # Convert the thres distance in km into lat/lon degree distances
    radius_lat = thres_ngrids * delta_lat
    radius_lon = thres_ngrids * delta_lon
    dlens = [radius_lat, radius_lon]
    if DB: print(f"Delta (degrees): lat={radius_lat}. lon={radius_lon}")
    #radius = np.sqrt(np.dot(dlens))
    #radius = scipy.spatial.distance.euclidean([radius_lat, radius_lon], [0, 0])
    radius = np.linalg.norm(dlens)
    if DB: print(f"Radius: {radius}")

    # Set up kD tree
    if tree is None:
        # List of all lat/lon grid points. n x m. n grid points. m(=2)-dimensions
        lats_lons = np.concatenate((lats.reshape(-1, 1), lons.reshape(-1, 1)), axis=1)
        # KD Tree with lat/lon grid points
        tree = spatial.cKDTree(lats_lons)
    
    # Structure storm report points for query with the tree
    # Column 0 is the lats and col 1 is the lons
    storm_points = np.stack((lats_storms, lons_storms), axis=1)
    storm_points = storm_points[mask_reports_by_time]
    
    # Identify the nearest neighbors
    if not use_ball:
        # For each storm point, find k closest grid points
        distances, neighbor_inds = tree.query(storm_points, k=4, workers=kdworkers)
    else:
        # Find all points within radius
        neighbor_inds = tree.query_ball_point(storm_points, radius, workers=kdworkers) #thres_ngrids
        if storm_points.size > 0: 
            neighbor_inds = np.concatenate(neighbor_inds).astype(int)

    n_nans = np.sum(np.isnan(neighbor_inds))
    if n_nans > 0: print(f"[create_storm_mask()] {n_nans} NaNs dtime={dtime}") 
    
    # Set up mask for the storm reports and neighboring grid space
    ncoords = lats.size #lats_lons.shape[0]
    mask = np.zeros((ncoords, 1), int)
    if storm_points.size > 0: 
        mask[neighbor_inds.ravel()] = 1
    mask = mask.reshape(lats.shape)

    return mask, mask_reports_by_time, tree

def _create_round_kernel(ksize):
    '''
    Create a kernel where the true values are in a circle
    '''
    Y, X = np.ogrid[:ksize, :ksize]
    isodd = ksize % 2
    r = (ksize - 1) / 2 #+ isodd
    dist_from_center = np.sqrt((X - r)**2 + (Y - r)**2)
    footprint = dist_from_center <= (ksize - isodd) / 2 #r
    return footprint



"""
"""
def compute_performance(y, y_pred, threshs, compute_sr_pod_csi=False, DB=False):
    '''
    Compute confusion matrix, POD, SR, and CSI using the thresholds
    Parameters
    -----------
    y: truth values
    y_pred: predictions
    threshs: list or np.array of threshold values for the predictions
    Return
    -----------
    results: dict with tps, fps, fns, tns, pods, srs, csis
    '''
    npos = np.count_nonzero(y)
    if npos == 0: 
        print(f"compute_performance():: No positives. npos={npos}")

    shape = threshs.shape
    
    tps = np.zeros(shape)
    fps = np.zeros(shape)
    fns = np.zeros(shape)
    tns = np.zeros(shape)

    srs = np.zeros(shape)
    pods = np.zeros(shape)
    csis = np.zeros(shape)

    #QS = np.quantile(y_pred, [0, 1])
    y_max = np.max(y_pred)
    for i, t in enumerate(threshs):
        mask = (y_pred > t).astype(float)
        '''
        cmtx = confusion_matrix(y.ravel(), mask.ravel())
        print(cmtx)
        tns[i] = cmtx[0, 0]
        fps[i] = cmtx[0, 1]
        fns[i] = cmtx[1, 0]
        tps[i] = cmtx[1, 1]
        '''
        tps[i], fps[i], fns[i], tns[i] = contingency_curves(y, mask, [.5])
        #if DB: print(tps[i], fps[i], fns[i], tns[i] )

        if t > y_max and DB:
            print(f'compute_performance():: [{i}] ymax: {y_max}, thres: {t}')
            print(f'compute_performance():: \t tn: {tns[i]}  fp: {fps[i]}  fn: {fns[i]}  tp:{tps[i]}')
            #continue
        

    srs = None
    pods = None 
    csis = None
    if compute_sr_pod_csi:
        srs = np.nan_to_num(compute_sr(tps, fps)) #tps / (tps + fps)
        pods = np.nan_to_num(compute_pod(tps, fns)) #tps / (tps + fns)
        csis = np.nan_to_num(compute_csi(tps, fns, fps)) #tps / (tps + fns + fps)

    results = {'tps': tps, 'fps': fps, 'fns': fns, 'tns': tns, 
               'srs': srs, 'pods': pods, 'csis': csis}

    return results

def compute_metrics(y, y_pred, threshs=None, metrics=None, not_metrics=False, DB=False):
    ''' 
    Compute performance metrics

    Parameters
    -----------
    y: ground truth labels
    y_pred: predictions
    metrics: list of strings of metrics to compute, or not compute if not_metrics
            is true. set to None to compute all metrics
            options: pod (same as tpr and recall), sr (same as precision), csi, 
                     fpr, freq_bias, 
                     pss = TPR - FPR 
                     fvaf = 1 - MSE / VAR 
                          = 1 - (sum[(y_true - y_pred)**2]) / (sum[(y_true - y_mean)**2])
                     mse
    not_metrics: bool set true to NOT compute the specified metrics

    Return
    ---------
    scores: dict with selected score results
    '''
    all_metrics = ['pod', 'sr', 'csi', 'fpr', 'freq_bias', 'pss', 'fvaf', 'mse']
    metrics.replace(['tpr', 'recall'], 'pod')
    metrics.replace('sr', 'precision')
    metrics = np.unique(metrics)
    # Verify valid metrics specified
    if not set(metrics).issubset(all_metrics): #set(a) <= set(b)
        pass

    results = compute_performance(y, y_pred, threshs, DB=DB)
    tps = results['tps']
    fps = results['fps']
    fns = results['fns']
    tns = results['tns']

    scores = {}
    if metrics is None or ('pod' in metrics and not not_metrics): #same as tpr and recall
        scores['pods'] = compute_pod(tps, fns) #tps / (tps + fns)

    if metrics is None or ('sr' in metrics and not not_metrics): #same as precision
        scores['srs'] = compute_sr(tps, fps) #tps / (tps + fps)

    if metrics is None or ('csi' in metrics and not not_metrics): #tps / (tps + fns + fps)
        scores['csi'] = compute_csi(tps, fns, fps) 

    if metrics is None or ('fpr' in metrics and not not_metrics): 
        scores['fpr'] = fps / (fps + tns)

    if metrics is None or ('freq_bias' in metrics and not not_metrics):
        scores['freq_bias'] = (tps + fps) / (tps + fns) #POD/SR = (tps+fps)/(tps+fns)

    if metrics is None or ('pss' in metrics and not not_metrics): #tpr - fpr
        scores['pss'] = compute_pod(tps, fns) - fps / (fps + tns)

    if metrics is None or ('f1' in metrics and not not_metrics): 
        scores['f1'] = 2 * tps / (2 * tps + fps + fns)

    # TODO
    if metrics is None or ('fvaf' in metrics and not not_metrics): 
        scores['fvaf'] = 1. - np.mean((y - y_pred)**2) / np.var(y)

    return scores

"""
Plotting
"""
import cartopy.feature as cfeature
def _add_cartopy(figaxs=None, ax_projection=None, border_ls='--', lake_alpha=.5,
                 figsize=(15,5), dpi=200):
    '''
    Add cartography to existing plot
    ax_projection: see fig.add_axes(projection=)
    figaxs: tuple with (fig, axs)
    border_ls: borders linestyle
    lake_alpha: alpha value for lakes
    '''
    fig = None
    ax = None
    if figaxs is None:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_axes([1, 1, 1, 1], projection=ax_projection)
    else:
        fig, ax = figaxs

    if not isinstance(ax, (list, tuple, np.ndarray)):
        ax = [ax]
    for _a in ax:
        _a.set_facecolor(cfeature.COLORS['water'])
        _a.add_feature(cfeature.LAND)
        _a.add_feature(cfeature.COASTLINE)
        _a.add_feature(cfeature.BORDERS, linestyle=border_ls)
        _a.add_feature(cfeature.LAKES, alpha=lake_alpha)
        _a.add_feature(cfeature.STATES)
    return fig, ax

def plot_storms(mask, lons_storm, lats_storm, thres_dist=None, thres_time=None,
                figax=None, figsize=(10, 5), xlims=None, ylims=None, 
                invert_yaxis=False, bcmap=mpl.colors.ListedColormap(['w', 'k']), 
                add_cartopy=False):
    '''
    Plot storm report mask
    Parameters
    ------------
    mask: 2D np array that is the storm mask
    lons_storm: 1D array of the longitude coords for the reported storms
    lats_storm: 1D array of the latitude coords for the reported storms
    figax: tuple with the fig object in index 0 and the axes list in index 1
    figsize: tuple with the figure width and height
    xlims:
    ylims:
    bcmap: binary colormap for the storm mask

    Returns
    ------------
    figure and axes objects
    '''
    f_ = None
    a_ = None
    if figax is None:
        f_, a_ = plt.subplots(1, 2, figsize=figsize)
    else:
        f_, a_ = figax

    # Plot points for each storm report
    a_[0].plot(lons_storm, lats_storm, '*', ms=8)

    if ylims is not None: a_[0].set_ylim(ylims)
    if xlims is not None: a_[0].set_xlim(xlims)

    a_[0].set_aspect('equal')
    if invert_yaxis:
        a_[0].invert_yaxis()

    # Show the mask for the storm reports
    a_[1].imshow(mask, cmap=bcmap)

    title = 'Storm Mask'
    if thres_dist is not None:
        title += f' dist_radius={thres_dist}km'
    if thres_time:
        title += f' time_radius={thres_time}min'
    a_[1].set_title(title)

    return f_, a_

def plot_uh(uh, mask_uh, uh_thres=None, figax=None, figsize=(10, 5), 
            invert_yaxis=False, vrange=[0, 40], cmap='Spectral_r',
            bcmap=mpl.colors.ListedColormap(['gray', 'blue'])):
    '''
    Plot UH masks
    Parameters
    ------------
    uh: 
    mask_uh: 2D np array that is the uh mask
    uh_thres: int or float value of the UH thres above which UH is considered
    figax: tuple with the fig object in index 0 and the axes list in index 1
    figsize: tuple with the figure width and height
    vrange: list or tuple with the min and max for the uh colorbar
    cmap: colormap for the full uh domain
    bcmap: binary colormap for the uh mask

    Returns
    ------------
    figure and axes objects
    '''
    f_ = None
    a_ = None
    if figax is None:
        f_, a_ = plt.subplots(1, 2, figsize=figsize)
    else:
        f_, a_ = figax

    im = a_[0].imshow(uh, vmin=vrange[0], vmax=vrange[1], cmap=cmap)
    cb = plt.colorbar(im, ax=a_[0], label='UH')

    # Color for False and True
    a_[1].imshow(mask_uh, cmap=bcmap)
    if uh_thres is not None: 
        a_[1].set_title(rf'UH $>$ {uh_thres:.3f}')
    else:
        a_[1].set_title('UH')

    if invert_yaxis:
        a_[0].invert_yaxis()
        a_[1].invert_yaxis()
    return f_, a_

def plot_zh_n_probs(ZH, P, thres_prob=None, contours=None, figax=None, 
                    figsize=(10, 5), invert_yaxis=False, zrange=[0, 60], 
                    prange=None, zhcmap='Spectral_r', pcmap='magma_r'):
    '''
    Plot the Composite reflectivity and the tornado probabilities
    Parameters
    ------------
    ZH: 2D reflectivity
    P: 2D np array that is the ml probabilities
    contours: TODO. contour P onto ZH
    figax: tuple with the fig object in index 0 and the axes list in index 1
    figsize: tuple with the figure width and height
    zrange: list or tuple with the min and max for the ZH colorbar
    prange: list or tuple with the min and max for the probabilities colorbar
    zhcmap: colormap for the reflectivity #magma_r inferno plasma cividis
    pcmap: colormap for the probabilities

    Returns
    ------------
    figure and axes objects
    '''
    f_ = None
    a_ = None
    if figax is None:
        f_, a_ = plt.subplots(1, 2, figsize=figsize)
    else:
        f_, a_ = figax

    im = a_[0].imshow(ZH, cmap=zhcmap, vmin=zrange[0], vmax=zrange[1]) 
    a_[0].set_title('Composite Reflectivity')
    cb = plt.colorbar(im, ax=a_[0], label='dBZ')

    im = a_[1].imshow(P, cmap=pcmap) 
    #TODO if prange is not None: im = a_[1].imshow(P, cmap=pcmap, vmin=prange[0], vmax=prange[1])
    if thres_prob is None: 
        a_[1].set_title('Tornado Probability')
    else:
        a_[1].set_title(f'Tornado Probability thres={thres_prob:.03f}')
    cb = plt.colorbar(im, ax=a_[1], label=r'$p_{tor}$')

    if invert_yaxis:
        a_[0].invert_yaxis()
        a_[1].invert_yaxis()
    
    return f_, a_

def det_curve(y, y_pred, fpr_fnr=None, use_ppf=0, gridon=True, figax=None, figsize=(8, 8)):
    '''
    Detection error tradeoff (DET) curve
    Analyze false positive - miss (FN) trade-off
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_det.html#sphx-glr-auto-examples-model-selection-plot-det-py
    https://github.com/scikit-learn/scikit-learn/blob/5c4aa5d0d/sklearn/metrics/_ranking.py#L273
    '''
    fpr_plt, fnr_plt = fpr_fnr
    if use_ppf:
        fpr_plt = stats.norm.ppf(fpr)
        fnr_plt = stats.norm.ppf(fnr)

    #line_kwargs = {} if name is None else {"label": name}
    #line_kwargs.update(**kwargs)
    #fpr_plt, fnr_plt, thresholds = det_curve(y_true, y_scores)

    fig = None
    ax = None
    if figax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig, ax = figax

    ax.plot(fpr_plt, fnr_plt, lw=3)
    ax.set(xlabel='FPR', ylabel='FNR')
    ax.set_aspect('equal')
    ax.grid(gridon)
    #_a.legend()

    if not use_ppf:
        ticks = np.linspace(0, 1, 6)
        tick_labels = ["{:.3}".format(t) for t in ticks]

        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    else:
        ticks = [0.001, 0.01, 0.05, 0.20, 0.5, 0.80, 0.95, 0.99, 0.999]
        tick_labels = [
            "{:.0%}".format(s) if (100 * s).is_integer() else "{:.1%}".format(s)
            for s in ticks
        ]

        tick_locations = stats.norm.ppf(ticks)
        ax.set_xticks(tick_locations)
        ax.set_xticklabels(tick_labels)
        ax.set_xlim(-3, 3)
        ax.set_yticks(tick_locations)
        ax.set_yticklabels(tick_labels)
        ax.set_ylim(-3, 3)

    return fig, ax

from MLstatkit.stats import Delong_test, Bootstrapping
def compare_rocs(ytrue, ypreds): #, rocs:np.ndarray=None):
    '''
    Pair wise statistical comparison between a list of ROC curves. All curves
    are compared to the first curve ypreds[0].
    https://pmc.ncbi.nlm.nih.gov/articles/PMC2774909/

        ytrue: list of ground truth
        ypreds: matrix where each column contains the predictions for each model.
                column 0 is the reference model all other models are compared to
    '''
    # Perform DeLong's test
    zscores = []
    pvals = []
    for i in range(1, ypreds.shape[1]):
        z, p = Delong_test(ytrue, ypreds[:,0], ypreds[:,i])
        zscores.append(z)
        pvals.append(p)

    return pd.DataFrame({'z_score': zscores, 'p_value': pvals})


if "__main__" == __name__:
    args_list = ['', 
                 #'--dir_wofs_preds', '/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_preds', 
                 #'--dir_wofs_preds', '/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_preds1/tor_unet_sample50_50_classweights20_80_hyper', 
                 #'--dir_wofs_preds', '/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_preds1/tor_unet_sample50_50_classweights50_50_hyper', 
                 '--dir_wofs_preds', '/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_preds1/tor_unet_sample90_10_classweights20_80_hyper', 
                 '--loc_storm_report', '/ourdisk/hpc/ai2es/tornado/tor_spc_ncei/2019_actual_tornadoes_clean.csv', # Paper revision with NCEI due Jul 13 2025
                 #'/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/tornado_reports/tornado_reports_2019_spring.csv', #original
                 #'/ourdisk/hpc/ai2es/tornado/stormreports/processed/tornado_reports_2019.csv', 
                 '--out_dir', './wofs_evaluations/_test_wofs_eval/revisions', 
                 #'--out_dir', '/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_figs/2019/summary', 
                 '--year', '2019', 
                 '--date0', '2019-04-28', 
                 '--date1', '2019-06-03',
                 '--skip_clearday',
                 #'--skip_cleartime',
                 #'--forecast_duration', '5', '13', 
                 '--uh_compute', #
                 '--thres_dist', '50', '--thres_time', '20', '--stat', 'mean', 
                 '--nthresholds', '51', '--ml_probabs_dilation', '33',
                 #'--model_name', 'tor_unet_sample50_50_classweightsNone_hyper',
                 #'--model_name', 'tor_unet_sample50_50_classweights50_50_hyper',
                 '--model_name', 'tor_unet_sample90_10_classweights20_80_hyper',
                 '-w', '0', 
                 #'--write_storm_masks', 
                 '--dry_run'] 
                 #'--uh_thres_list', '0', '200'  #'363'
    
    args = parse_args(args_list) #args_list)#
    print(args)

    # "Round" dilation kernel mask
    footprint = _create_round_kernel(args.ml_probabs_dilation)

    # LOAD STORM REPORTS 
    # Storm reports
    date_start = args.date0
    date_end = args.date1

    df_storm_report, times_storms = load_storm_report_large(args.loc_storm_report, 
                                                            args.date0, args.date1)
    lats_storm = df_storm_report['Lat'].values 
    lons_storm = df_storm_report['Lon'].values
    print("Storm Report Shape", df_storm_report.shape)
    


    # TODO: pull IEM polygons
    # def generate_polygon_mask(polygons)
    # input: list of polygons
    # output: new ground truth mask


    
    # CONSTRUCT LIST OF DATA FILES
    # TODO (Maybe): create command line args for extr_args
    # Format string for python datetime object
    fmt_dt = '%Y-%m-%d_%H:%M:%S' #'%Y-%m-%d_%H_%M_%S' '%Y-%m-%d_%H_%M_%S'
    # regex pattern string for the datetime format
    fmt_re = r'\d{4}-\d{2}-\d{2}_\d{2}(_|:)\d{2}(_|:)\d{2}'
    extr_args = {'method': 're', 'fmt_dt': fmt_dt, 'fmt_re': fmt_re}

    dir_preds = os.path.join(args.dir_wofs_preds, args.year)
    rdates = sorted(os.listdir(dir_preds))
    print(f"{len(rdates)} run dates. {rdates}")

    nrdates = len(rdates)
    all_dfs = [None] * nrdates #[]
    for i, DATE in enumerate(rdates):
        print(f" ... ({i}) merging", DATE)
        _df = generate_init_times_files_list_all(dir_preds, DATE, 
                                                 save_path=None, 
                                                 df_index=False, 
                                                 **extr_args)
        all_dfs[i] = _df #all_dfs.append(_df)
    df_all_files = pd.concat(all_dfs, ignore_index=True, verify_integrity=True)
    all_dfs = None #del all_dfs

    
    ncpus = int(os.environ['SLURM_CPUS_PER_TASK'])
    #with ThreadPoolExecutor(max_workers=ncpus, thread_name_prefix='rundate_') as executor:


    # LOAD DATA 
    # 0. select files by initialization time
    # 1. select files by forecast time
    # 2. determine storm mask for the give time
    # 3. compare storm mask to ml preds and uh

    all_ftimes = df_all_files['forecast_time'].values
    ftimes = np.unique(all_ftimes)
    n_ftimes = ftimes.size
    print("# File Times", n_ftimes)

    y_preds = None
    y_uh = None

    sel_files_by_ftime = None
    y_storm = None

    nthreshs = args.nthresholds
    csithreshs = np.linspace(0, 1, nthreshs)
    uhthreshs = np.array([0, .01, .02, .025, .05, .1, .2, .25, .5, 
                          1, 5, 10, 20, 25, 40, 50, 80, 100]) #, 200, 300])
    nthreshs_uh = uhthreshs.size

    manager = Manager()
    lock = manager.Lock()
    lock_uh = manager.Lock()

    NANs = np.zeros((n_ftimes, nthreshs)) 
    results = manager.dict({'tps': NANs.copy(), 'fps': NANs.copy(), #np.full((n_ftimes, nthreshs), np.nan)
               'fns': NANs.copy(), 'tns': NANs.copy(),
               'npos': NANs[:,0].copy(), 'nneg': NANs[:,0].copy()} )
    
    NANs = np.zeros((n_ftimes, nthreshs_uh))  
    results_uh = manager.dict({'tps': NANs.copy(), 'fps': NANs.copy(), 
                  'fns': NANs.copy(), 'tns': NANs.copy(),
                  'npos': NANs[:,0].copy(), 'nneg': NANs[:,0].copy()})
    
    # pandas.DateFrame datetime format string
    fmt_dt_pd = '%Y-%m-%d %H:%M:%S'
    storm_timedelta = pd.Timedelta(hours=1) #timedelta(hours=1) #

    #kdtree = None

    # For histogram and reliability plots
    # Observed distribution
    #>obs_distr = np.zeros((0,))
    # UH distribution
    #>uh_distr = np.zeros((0,))
    # ML probab distribution
    #>prob_distr = np.zeros((0,))
    
    q_thres = .3
    q = 0
    qs = np.random.rand(df_all_files.shape[0])

    #prob_max = []
    #uh_max = []

    # List to print the forescast and storm datetimes of interest
    of_interest = []
    df_availstorms = pd.DataFrame({'date': [], 'ftime0': [], 
                                   'ftime1': [], 'storm_count': []})
    df_forecaststorms = pd.DataFrame({'rundate': [], 'ftime0': [], 
                                   'ftime1': [], 'forecastime': [], 
                                   'storm_pixel_count': []})

    # For selecting windows of forecast times
    n_3hrs = 37 #64 
    nf = len(args.forecast_duration)
    fslice = slice(0, n_3hrs)
    if nf == 1:
        fslice = slice(0, args.forecast_duration[0])
    elif nf == 2:
        fslice = slice(*args.forecast_duration)

    # create the shared pipe
    #conn1obs, conn2obs = Pipe()
    #conn1probs, conn2probs = Pipe()
    #conn1probsuh, conn2probsuh = Pipe()

    # Collect results
    #pipe_obs, pipe_probs, pipe_probs_uh
    def execute_run(r, lock, lock_uh):
            gc.collect()
            _pid = os.getpid()
            print(f"[execute_run()] Worker PID: {_pid}")
            print(f"[execute_run()] Python Thread ID: {threading.get_ident()}")
            print(f"[execute_run()] Native Thread ID: {threading.get_native_id()}")
            global args, df_all_files, rdates, fslice, fmt_dt_pd, storm_timedelta
            global times_storms, lats_storm, lons_storm, footprint, q_thres
            global qs, csithreshs, uhthreshs
            # read the data from the pipe
            #obs_distr = conn1obs.recv()
            #prob_distr = conn1probs.recv()
            #uh_distr = conn1probsuh.recv()

            # Observed distribution
            obs_distr = np.zeros((0,))
            # UH distribution
            uh_distr = np.zeros((0,))
            # ML probab distribution
            prob_distr = np.zeros((0,))

            rdate = rdates[r]

        # Iterate of WoFS run dates
        #for r, rdate in enumerate(rdates): # #[3:4]20190502
            rdate_files = df_all_files.loc[df_all_files['run_date'] == rdate]
            itimes = np.unique(rdate_files['init_time'].values)
            itimes = np.unique(itimes)

            # Isoloate storms in day forecast window
            t0 = rdate_files['forecast_time'].min()
            t0 = datetime.strptime(t0, fmt_dt).strftime(fmt_dt_pd)
            t1 = rdate_files['forecast_time'].max()
            t1 = datetime.strptime(t1, fmt_dt).strftime(fmt_dt_pd)

            storms_for_rdate = df_storm_report[t0:t1]
            _latstorm = storms_for_rdate['Lat'].values
            _lonstorm = storms_for_rdate['Lon'].values
            _times = np.array([d.to_pydatetime() for d in pd.to_datetime(storms_for_rdate.index.values)])

            nstorms = storms_for_rdate.shape[0]
            print(nstorms, "Storms for run date:\n", storms_for_rdate)

            if args.write_storm_masks:
                df_availstorms = pd.concat([df_availstorms, 
                                            pd.DataFrame({'date': [rdate], 'ftime0': [t0], 
                                            'ftime1': [t1], 'storm_count': [nstorms]})])
            if nstorms == 0:
                print(f"No storms on {rdate}. nstorms={nstorms}")
                if args.skip_clearday: return #continue

            # Check storm coords overlap with WoFS domain
            _fn = rdate_files['filename_path'].values[0]
            wofs_preds = xr.load_dataset(_fn)
            xlats = np.squeeze(wofs_preds['XLAT'].values)
            xlons = np.squeeze(wofs_preds['XLONG'].values)
            #dateindex = datetime.strptime(rdate, '%Y%m%d').strftime('%Y-%m-%d')

            y_storm, _, _ = create_storm_mask(xlats, xlons, None, 
                                            _latstorm, _lonstorm, 
                                            _times, args.gridsize,
                                            thres_dist=1, 
                                            thres_time=args.thres_time)
            nstorms = np.count_nonzero(y_storm)
            if nstorms == 0: 
                print(f"No storms on {rdate} in WoFS spatial domain. nstorms={nstorms}. skip={args.skip_clearday}")
                if args.skip_clearday: return #continue

            # Iterate over initialization times
            #for i, itime in enumerate(itimes): ####
            for i, itime in enumerate(itimes[:3]): ########################## ********
                init_files = rdate_files.loc[rdate_files['init_time'] == itime]
                #forecast_times = init_files['forecast_time'].unique()
                #if args.dry_run: print(forecast_times[fslice])

                # Select forecast times 1hr before the first storm of the day
                #   and 1hr after the last storm of the day
                if i == 0: print("run: t0 t1", t0, t1)
                t0 = pd.to_datetime(storms_for_rdate.index[0]) - storm_timedelta
                t1 = pd.to_datetime(storms_for_rdate.index[-1]) + storm_timedelta
                sel_after_t0 = pd.to_datetime(init_files['forecast_time'], format=fmt_dt) >= t0
                sel_before_t1 = pd.to_datetime(init_files['forecast_time'], format=fmt_dt) <= t1
                forecast_times = init_files['forecast_time'].loc[sel_after_t0 & sel_before_t1].unique()
                if i == 0: print("itime: t0 t1", t0, t1)

                _fslice = fslice
                if nf != 2: 
                    nt = max(0, min(n_3hrs, forecast_times.size))
                    _fslice = slice(0, nt)
                if args.dry_run: 
                    print("Forecast Times:\n", forecast_times[_fslice])
                    print(f"Forecast Times Range: {t0} to {t1}")


                #####################################
                #wofs_preds = xr.load_dataset(f'/ourdisk/hpc/ai2es/nsnook/WoFS_summary_data_2019/summary_data_{forecast_times[0]}.netcdf', engine='netcdf4')
                #####################################


                # Iterate over first 3 hours of forecast times or the selected 
                #   forecast time range
                for f, ftime in enumerate(forecast_times[_fslice]): 
                    sel_files_by_ftime = select_files(df_all_files, itime, ftime, 
                                                    emember=None, rdate=rdate)
                    
                    #######################
                    #wofs_preds_mean = wofs_preds['ens_mean_ML_probability'][f]
                    #'ens_mean_UH'
                    #######################

                    # Load all ensembles for the forecast time
                    concat_dim = 'ENS'
                    _sel_fnpaths = sel_files_by_ftime['filename_path'].values
                    print("\nFILES TO AVERAGE")
                    print(f'r{r:2}:{rdate} | i{i:2}:{itime} | f{f:2}:{ftime}')
                    print(_sel_fnpaths)
                    wofs_preds = xr.open_mfdataset(_sel_fnpaths, concat_dim=concat_dim,
                                                combine='nested', coords='minimal',
                                                decode_times=False) #, engine='kvikio')
                    dtime = num2date(wofs_preds.Time[0], 'seconds since 2001-01-01')
                    wofs_preds.drop_vars('Time')

                    print(f'[r={r:2d}, {rdate}](i={i:2d}, {itime}) {f:2d} / {n_3hrs} current: {ftime} ({forecast_times[0]} to {forecast_times[-1]}) dtime=', dtime)

                    if sel_files_by_ftime.shape[0] != 18: 
                        print("ENS LOW", sel_files_by_ftime.shape)
                        of_interest.append(f'{rdate} | {itime} | {ftime} | {dtime}')
                    if args.dry_run:
                        print(sel_files_by_ftime.shape)
                        print(sel_files_by_ftime['filename_path'].values)

                    # COMPUTE MEAN/MEDIAN
                    if args.stat == 'mean':
                        # Use [0] to remove the singleton dimension. analogous to np.squeeze
                        y_preds = wofs_preds['ML_PREDICTED_TOR'].mean(dim=concat_dim).values[0]
                        if args.uh_compute:
                            y_uh = wofs_preds['UP_HELI_MAX'].mean(dim=concat_dim).values[0]

                    elif args.stat == 'median':
                        y_preds = wofs_preds['ML_PREDICTED_TOR'].median(dim=concat_dim).values[0]
                        if args.uh_compute:
                            y_uh = wofs_preds['UP_HELI_MAX'].median(dim=concat_dim).values[0]
                            
                    
                    shape = y_preds.shape
                
                    # CREATE STORM MASK for the given forecast time, distance thres, and time thres
                    # Extract lat/lon once per day or init time instead
                    if f == 0:
                        xlats = np.squeeze(wofs_preds['XLAT'].values)
                        xlons = np.squeeze(wofs_preds['XLONG'].values)

                    y_storm, sel_storms, _ = create_storm_mask(xlats, xlons, dtime, 
                                                                lats_storm, lons_storm, 
                                                                times_storms, 
                                                                args.gridsize, 
                                                                thres_dist=args.thres_dist, 
                                                                thres_time=args.thres_time, 
                                                                kdworkers=-1) #-1
                    
                    npos = np.count_nonzero(y_storm)
                    if args.write_storm_masks:
                        df_forecaststorms = pd.concat([df_forecaststorms,
                                                    pd.DataFrame({'rundate': [rdate], 
                                                                    'ftime0': [t0],
                                                                    'ftime1': [t1], 
                                                                    'forecastime': [ftime], 
                                                                    'storm_pixel_count': [npos]})])
                    # write stormmasks
                    #dir_stormmask = f'/ourdisk/hpc/ai2es/tornado/tornado_report_masks_2019_spring/50km_20min/{args.year}/{rdate}/{itime}'
                    dir_stormmask = f'/ourdisk/hpc/ai2es/tornado/tor_spc_ncei/report_masks_2019_spring/50km_20min/{args.year}/{rdate}/{itime}'
                    fn_stormmask = os.path.join(dir_stormmask, f'{ftime}.txt')
                    if not os.path.exists(dir_stormmask): 
                        os.makedirs(dir_stormmask, exist_ok=True) 
                    if npos == 0: 
                        print(f"No tornados at {itime} {ftime} or within the WoFS Domain. npos={npos}")
                        if not os.path.exists(fn_stormmask) and args.write_storm_masks: 
                            np.savetxt(fn_stormmask, np.array([[0]]), fmt='%d')
                        if args.skip_cleartime: continue
                    elif not os.path.exists(fn_stormmask) and args.write_storm_masks: 
                        np.savetxt(fn_stormmask, y_storm, fmt='%d')


                    # COMPUTE PERFORMANCE
                    #y_storm = masks #"ground truth"
                    #_npos = np.count_nonzero(y_storm)
                    _nneg = np.count_nonzero(y_storm==0) #np.count_nonzero(~y_storm)
                    print(f"pos: {npos}  neg: {_nneg}; total: {npos + _nneg}; size: {y_storm.size}")
                    if args.dry_run: 
                        classes, counts = np.unique(y_storm, return_counts=True)
                        print("Storm Class Counts:\n", np.stack((classes, counts), axis=-1), f'\n(r{r:2}:{rdate} i{i:2}:{itime} f{f:2}:{ftime})')


                    # Dilate
                    if args.ml_probabs_dilation > 0:
                        ksize = args.ml_probabs_dilation
                        y_preds = grey_dilation(y_preds, footprint=footprint) 
                    if args.ml_probabs_norm:
                        y_preds = normalize(y_preds.reshape(-1, 1), norm='max', 
                                            axis=0).reshape(shape)


                    # Calculate performance ML
                    print(f" [PID={_pid}] Calculate performance ML...")
                    tps, fps, fns, tns = contingency_curves(y_storm.ravel(), y_preds.ravel(), csithreshs.tolist())
                    with lock:
                        results['tps'][f] += tps
                        results['fps'][f] += fps
                        results['fns'][f] += fns
                        results['tns'][f] += tns
                        results['npos'][f] += npos
                        results['nneg'][f] += _nneg
                    print(f" [PID={_pid}] Done calculating performance ML")

                    #q = np.random.rand(1)
                    #if q > q_thres:
                    #if qs[q] > q_thres:
                    qi = np.random.choice(qs)
                    print("qi", qi)
                    if qi > q_thres:
                        obs_distr = np.append(obs_distr, y_storm.ravel())
                        prob_distr = np.append(prob_distr, y_preds.ravel())
                    #if not args.uh_compute: q += 1

                    # UH
                    if args.uh_compute:
                        # Dilate
                        if args.ml_probabs_dilation > 0:
                            ksize = args.ml_probabs_dilation
                            y_uh = grey_dilation(y_uh, footprint=footprint) 
                        if args.ml_probabs_norm:
                            y_uh = normalize(y_uh.reshape(-1, 1), norm='max', axis=0).reshape(shape)

                        # Max UH over time
                        #if q > q_thres:
                        #if qs[q] > q_thres:
                        if qi > q_thres:
                            uh_distr = np.append(uh_distr, y_uh.ravel())
                        #q += 1

                        _res = compute_performance(y_storm, y_uh, uhthreshs)
                        with lock_uh:
                            results_uh['tps'][f] += _res['tps']
                            results_uh['fps'][f] += _res['fps']
                            results_uh['fns'][f] += _res['fns']
                            results_uh['tns'][f] += _res['tns']
                            results_uh['npos'][f] += npos
                            results_uh['nneg'][f] += _nneg

            # send the array via a pipe
            #pipe_obs.send(obs_distr)
            #pipe_probs.send(prob_distr)
            #pipe_probs_uh.send(uh_distr)
            #pipe_obs.close()
            #pipe_probs.close()
            #pipe_probs_uh.close()
            print(f'[PID={_pid}] FINAL RESULTS')
            print(results)
            if args.uh_compute: print(results_uh)
            return {#'results': results, 'results_uh': results_uh,
                    'obs_distr': obs_distr, 'prob_distr': prob_distr,
                    'uh_distr': uh_distr}

    nprocesses = int(os.environ['SLURM_NTASKS']) #SLURM_NPROCS
    print(nprocesses, "workers")

    obs_distr = np.zeros((0,))
    prob_distr = np.zeros((0,))
    uh_distr = np.zeros((0,))

    future_results = []
    locks = [lock for _ in range(nrdates)]
    locks_uh = [lock_uh for _ in range(nrdates)]

    #, mp_context=get_context('spawn')
    with ProcessPoolExecutor(max_workers=nprocesses, mp_context=get_context('fork')) as process_executor:
        #futures = [executor.submit(execute_run, *(conn2obs, conn2probs, conn2probsuh, r, qi)) for r, qi in enumerate(qs[:3])]
        #wait(futures)
        try: 
            future_results = process_executor.map(execute_run, range(nrdates), locks, locks_uh)
        except Exception as err:
            print("[PROCESS ERR]", err)
    
        # Wait until at least one task completes
        #done, not_done = wait(futures, return_when=FIRST_COMPLETED)

        for future in tqdm(as_completed(future_results), total=nrdates, desc='Futures Completed'):
            print("===="*9)
            try:
                res = future.result()
                obs_distr = np.append(obs_distr, res['obs_distr'])
                prob_distr = np.append(prob_distr, res['prob_distr'])
                uh_distr = np.append(uh_distr, res['uh_distr'])
                print("COMPLETED\n", res) #timeout=36_000
            except Exception as err:
                print("[FUTURE.RESULT() ERR]", err)
            print('\n\n')
            print("===="*9)
        #timeout=...

        #conn1obs.send(obs_distr)
        #conn1probs.send(prob_distr)
        #conn1probsuh.send(uh_distr)
        # read the data from the pipe
        #obs_distr = conn1obs.recv()
        #prob_distr = conn1probs.recv()
        #uh_distr = conn1probsuh.recv()

    if args.write:
        if args.write_storm_masks:
            fn_avail = fname = os.path.join(args.out_dir, f'{args.year}_adjusted_storm_counts.csv')
            df_availstorms.to_csv(fn_avail) #, index=False)
            fn_avail = fname = os.path.join(args.out_dir, f'{args.year}_forecast_storm_counts.csv')
            df_forecaststorms.to_csv(fn_avail) #, index=False)

        # Construct file name prefix and suffix
        fn_suffix = '_00_36slice' if nf == 1 else  f'_{fslice.start:02d}_{fslice.stop:02d}slice'
        legend_txt = ''
        if args.ml_probabs_dilation: 
            fn_suffix += '_dilated'
            radius_kms = np.ceil(args.gridsize * args.ml_probabs_dilation / 2)
            legend_txt = f'(neighborhood radius {radius_kms}km)'
        if args.ml_probabs_norm: 
            fn_suffix += '_normed'
            legend_txt += f'(normalized)'
        if args.skip_cleartime or args.skip_clearday:
            fn_suffix += '_skip'
        prefix = '_revision_'
        if args.dry_run:
            prefix += '_test_' 
            args.year = rdate
        if args.model_name != '': prefix += f'{args.model_name}_'

        # Set up figure
        figsize = (15, 15)
        fig, axs = plt.subplots(3, 2, figsize=figsize)
        axs = axs.ravel()
        title = rf'Radius {args.thres_dist}km Time $\pm$ {args.thres_time}min'
        fig.suptitle(title)

        tps = np.nansum(results['tps'], axis=0)
        fps = np.nansum(results['fps'], axis=0)
        fns = np.nansum(results['fns'], axis=0)
        tns = np.nansum(results['tns'], axis=0)

        srs_agg = compute_sr(tps, fps) 
        pods_agg = compute_pod(tps, fns) 
        csis_agg = compute_csi(tps, fns, fps)

        #far = 1 - srs_agg
        fpr = fps / (fps + tns)
        fnr = fns / (fns + tps)
        f1 = 2 * tps / (2 * tps + fps + fns)
        acc = (tps + tns) / (tps + fps + fns + tns)

        npos = np.nansum(results['npos'])
        nneg = np.nansum(results['nneg'])
        pos_rate = npos / (npos + nneg)
        print(f"npos: {npos}  nneg: {nneg}; total: {npos + nneg}")
        print(f"accuracy quantiles: {np.nanquantile(acc, [0, .5, 1])}")

        # Write for each model type and time interval
        fname = os.path.join(args.out_dir, f'{prefix}{args.year}_performance_results_{args.stat}{fn_suffix}.csv') 
        print(f" * SAVING {fname}")
        pd.DataFrame({'thres': csithreshs, 'tps': tps, 'fps': fps, 'fns': fns, 
                      'tns': tns, 'srs': srs_agg, 'pods': pods_agg, 'csis': csis_agg,
                      'fpr': fpr, 'fnr': fnr, 'acc': acc, 'f1': f1}).to_csv(fname, header=True, index=False)


        
        fname = os.path.join(args.out_dir, f'{prefix}{args.year}_performance_diagrams_{args.stat}{fn_suffix}.png')

        # CSI Performance Diagram
        _figax = plot_csi(obs_distr, prob_distr, #masks[ftime].ravel(), prob_max.ravel() #wofs_prob[ftime].ravel(), 
                         fname, threshs=csithreshs, 
                         label=f'ML Probabilities {legend_txt}', color='blue', 
                         save=False, srs_pods_csis=(srs_agg, pods_agg, csis_agg, pos_rate),
                         return_scores=False, draw_ann=3, fig_ax=(fig, axs[1]))
        #cb = fig.colorbar(axs[1].collections[-1], ax=axs[1])
        #cb = axs[1].collections[0].colorbar

        # ROC Curve
        _figax = plot_roc(obs_distr, prob_distr, fname, tpr_fpr=(pods_agg, fpr), 
                           fig_ax=(fig, axs[0]), save=False, plot_ann=3) 

        if args.uh_compute:
            _ypreds = np.stack([uh_distr, prob_distr], axis=-1)
            roc_stattest = compare_rocs(obs_distr, _ypreds)
            fn_roc_stattest = os.path.join(args.out_dir, f'{prefix}{args.year}_delong_result_{args.stat}{fn_suffix}_uh.csv') 
            roc_stattest.to_csv(fn_roc_stattest, header=True)
            print(f" [WRITE=args.write] Saved {fn_roc_stattest}")

        '''# PRC
        _figax = plot_prc(obs_distr, prob_distr, fname, 
                           pre_rec_posrate=(srs_agg, pods_agg, pos_rate), 
                           draw_ann=2, fig_ax=(fig, axs[2]), save=False)
        '''

        # DET Curve
        _figax = det_curve(obs_distr, prob_distr, fpr_fnr=(fpr, fnr), 
                           figax=(fig, axs[2]))

        # Calibration Diagram
        fig, ax = plot_reliabilty_curve(obs_distr, prob_distr, fname, save=False, 
                                        fig_ax=(fig, axs[3]), strategy='uniform')
        
        axs[4].plot(fpr, acc)
        axs[4].set(xlabel='FPR', ylabel='Accuracy')
        axs[4].set(xlim=[0, 1], ylim=[0, 1])
        axs[4].set_aspect('equal')
        axs[4].grid()
        
        from seaborn import histplot
        #Y = {'Train': y_preds, 'Val': y_preds_val}
        histplot(data=prob_distr, stat='probability', kde=True, #legend=True, 
                 log_scale=(False, True), alpha=.8, ax=axs[5]) #, common_norm=False)
        #axs[5].hist(prob_distr, bins=51, density=False)
        axs[5].set(title='ML Probability Distribution', xlabel='ML Probability')
        axs[5].set_ylabel('Log Probability')
        #axs[5].set_yscale('log')
        axs[5].set_xlim([0, 1])

        '''
        # Extract positions of square and above axes
        p1 = axs[1].get_position()
        p3 = axs[3].get_position()
        # make square axes with left position taken from above axes, and set position
        p1 = [p3.x0, p1.y0, p1.width, p1.height]
        axs[1].set_position(p1) 
        '''

        print(" * (Done plotting) SAVING plots")
        print(fname)
        plt.subplots_adjust(wspace=.18, hspace=.35)
        plt.savefig(fname, dpi=180, bbox_inches='tight')


        ################################################################
        ################################################################
        # Save predictions
        if args.uh_compute: 
            fn_outputs = os.path.join(args.out_dir, f'{prefix}{args.year}_youtputs_{args.stat}{fn_suffix}_uh.csv')
            print(f" [WRITE={args.write}] Saving performance measure confidence intervals (uh)", fn_outputs)
            pd.DataFrame({'y_true': obs_distr, 'y_uh': uh_distr, 
                          f'y_{args.model_name}': prob_distr}).to_csv(fn_outputs, header=True, index=False)
        else: 
            fn_outputs = os.path.join(args.out_dir, f'{prefix}{args.year}_youtputs_{args.stat}{fn_suffix}.csv')
            print(f" [WRITE={args.write}] Saving performance measure confidence intervals", fn_outputs)
            pd.DataFrame({'y_true': obs_distr, f'y_{args.model_name}': prob_distr}).to_csv(fn_outputs, header=True, index=False)


        '''
        # Calculate confidence intervals for AUROC
        metrics = ['roc_auc', 'pr_auc'] #, 'f1', 'accuracy', 'recall', 'precision']
        stats_model = pd.DataFrame({score: [] for score in metrics}) 
        #, index=['value', 'confidence_lower', 'confidence_upper'])
        stats_model['label'] = ['value', 'confidence_lower', 'confidence_upper']

        stats_uh = pd.DataFrame({score: [pd.NA] * 3 for score in metrics}) 
        stats_uh['label'] = ['score_value', 'confidence_lower', 'confidence_upper']

        n_bootstraps = 500

        for s, score in enumerate(metrics):
            og_score, confidence_lower, confidence_upper = Bootstrapping(obs_distr, prob_distr, score, n_bootstraps=n_bootstraps)
            print(f"{score}: {og_score:.3f}, Confidence interval: [{confidence_lower:.3f} - {confidence_upper:.3f}]")

            stats_model[score] = [og_score, confidence_lower, confidence_upper]
            
            if args.uh_compute: 
                og_score, confidence_lower, confidence_upper = Bootstrapping(obs_distr, uh_distr, score)
                print(f"{score}: {og_score:.3f}, Confidence interval (UH): [{confidence_lower:.3f} - {confidence_upper:.3f}]")
                
                stats_uh[score] = [og_score, confidence_lower, confidence_upper]

        fn_stats = os.path.join(args.out_dir, f'{prefix}{args.year}_performance_confidence_intervals_{args.stat}{fn_suffix}.csv') 
        print(f" [WRITE={args.write}] Saving performance measure confidence intervals (model)", fn_stats)
        print(stats_model)
        stats_model.set_index('label').to_csv(fn_stats, header=True)
        
        fn_stats = os.path.join(args.out_dir, f'{prefix}{args.year}_performance_confidence_intervals_{args.stat}{fn_suffix}_uh.csv') 
        print(f" [WRITE={args.write}] Saving performance measure confidence intervals (uh)", fn_stats)
        print(stats_uh)
        if args.uh_compute: stats_uh.set_index('label').to_csv(fn_stats, header=True)
        '''
        ################################################################
        ################################################################


        # UH
        if args.uh_compute:
            tps = np.nansum(results_uh['tps'], axis=0)
            fps = np.nansum(results_uh['fps'], axis=0)
            fns = np.nansum(results_uh['fns'], axis=0)
            tns = np.nansum(results_uh['tns'], axis=0)

            srs_agg = np.nan_to_num(compute_sr(tps, fps))
            pods_agg = np.nan_to_num(compute_pod(tps, fns))
            csis_agg = np.nan_to_num(compute_csi(tps, fns, fps))

            fpr = fps / (fps + tns)
            fnr = fns / (fns + tps)
            f1 = 2 * tps / (2 * tps + fps + fns)
            acc = (tps + tns) / (tps + fps + fns + tns)

            fname = os.path.join(args.out_dir, f'{prefix}{args.year}_performance_results_{args.stat}{fn_suffix}_uh.csv') 
            print(f" * SAVING {fname}")
            pd.DataFrame({'thres': uhthreshs, 'tps': tps, 'fps': fps, 'fns': fns, 
                          'tns': tns, 'srs': srs_agg, 'pods': pods_agg, 'csis': csis_agg,
                          'fpr': fpr, 'fnr': fnr, 'acc': acc, 'f1': f1}).to_csv(fname, header=True, index=False)

            print("uh max quantiles", np.nanquantile(uh_distr, [0, 1]))
            fname = os.path.join(args.out_dir, f'{prefix}{args.year}_performance_diagram_{args.stat}{fn_suffix}_uh.png')
            print(f" * SAVING {fname}")
            figax = plot_csi(obs_distr, uh_distr, 
                            fname, threshs=uhthreshs, label=f'UH {legend_txt}', 
                            color='blue', save=True, 
                            srs_pods_csis=(srs_agg, pods_agg, csis_agg, pos_rate), 
                            return_scores=False, fig_ax=None, figsize=(10, 6), tight=False)

    print("ENS Dimension Low:", of_interest)

    print('DONE.')
