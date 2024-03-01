"""
author Monique Shotande

Pixel-wise Evaluation of WoFS predictions.
Generates a pandas dataframe of all the files for convenience
when selecting all ensembles for specific forecast time, 
initialization time, and date.

Mean and median UH and ML probabilities can be computed across the 
ensembles. Results for UH can be optionally computed with the 
argument flag, --uh_compute.

min00-20 :  0,  5
min20-60 :  5, 12
min60-90 : 12, 18
min90-180: 18, 36

"""

import re, os, sys, errno, glob, argparse
from datetime import timedelta, datetime, date, time
from dateutil.parser import parse as parse_date

import xarray as xr
print("xr version", xr.__version__)
import numpy as np
print("np version", np.__version__)
import pandas as pd
print("pd version", pd.__version__)
from scipy import spatial
from scipy.ndimage import grey_dilation
from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn.metrics import confusion_matrix

#import netCDF4
#print("netCDF4 version", netCDF4.__version__)
from netCDF4 import num2date #, Dataset, date2num

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
        help='Directory location of WoFS predictions, with subdirectories for the initialization times, which each contain subdirectories for all ensembles, which contains the predictions for each forecast time. Example directory: wofs_preds/2023/20230602. Example directory structure: wofs_preds/2023/20230602/1900/ENS_MEM_13/wrfwof_d01_2023-06-02_20_30_00_predictions.nc')
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
                       help='List of at most 2 integers. The indices of the forecast range to evaluate on. For example, for the forecast range 0 to 20 min, the indices are 0 and 5. ')

    parser.add_argument('--uh_compute', action='store_true',
        help='Whether to compute performance results for the UH')
    parser.add_argument('--uh_thres_list', type=float, nargs='+',
        help='List of UH threshold values')

    # TODO: skip forecast time vs skip day
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
        help='Stat to use for encorporating the ensembles into the evaluations. median (default) | mean | agg (TODO)')
    
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
    parser.add_argument('-d', '--dry_run', action='store_true',
        help='For testing and debugging')

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
    the storm reports reside

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
    thres_time: int or float in minutes
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



if "__main__" == __name__:
    args_list = ['', 
                 #'--dir_wofs_preds', '/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_preds', 
                 #'--dir_wofs_preds', '/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_preds1/tor_unet_sample50_50_classweights20_80_hyper', 
                 #'--dir_wofs_preds', '/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_preds1/tor_unet_sample50_50_classweights50_50_hyper', 
                 '--dir_wofs_preds', '/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_preds1/tor_unet_sample90_10_classweights20_80_hyper', 
                 '--loc_storm_report', '/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/tornado_reports/tornado_reports_2019_spring.csv', 
                 #'/ourdisk/hpc/ai2es/tornado/stormreports/processed/tornado_reports_2019.csv', 
                 '--out_dir', './wofs_evaluations/_test_wofs_eval', 
                 #'--out_dir', '/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_figs/2019/20190430/summary', 
                 '--year', '2019', 
                 '--date0', '2019-04-28', 
                 '--date1', '2019-06-03',
                 '--forecast_duration', '0', '6',
                 '--thres_dist', '50', '--thres_time', '20', '--stat', 'mean', 
                 '--nthresholds', '51', '--ml_probabs_dilation', '33', '--uh_compute',
                 #'--skip_clearday',
                 #'--skip_cleartime',
                 '--model_name', 'tor_unet_sample50_50_classweights20_80_hyper',
                 '-w', '1', '--dry_run'] 
                 #, '--ml_probabs_norm',
                 #'--uh_thres_list', '0', '200'  #'363'
    args = parse_args(args_list)
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

    # Set reports to index by DateTime for convenience
    #storm_reports = df_storm_report.copy()
    #storm_reports = storm_reports.set_index('DateTime')


    # TODO: pull IEM polygons
    # def generate_polygon_mask(polygons)
    # input: list of polygons
    # output: new ground truth mask


    
    # CONSTRUCT LIST OF DATA FILES
    fmt_dt = '%Y-%m-%d_%H:%M:%S' #'%Y-%m-%d_%H_%M_%S' '%Y-%m-%d_%H_%M_%S'
    fmt_re = r'\d{4}-\d{2}-\d{2}_\d{2}(_|:)\d{2}(_|:)\d{2}'
    extr_args = {'method': 're', 'fmt_dt': fmt_dt, #args.extract_date_method, args.fmt_dt, args.fmt_re}
                 'fmt_re': fmt_re}
    dir_preds = os.path.join(args.dir_wofs_preds, args.year)
    rdates = sorted(os.listdir(dir_preds))
    all_dfs = []
    for DATE in rdates:
        _df = generate_init_times_files_list_all(dir_preds, DATE, 
                                                 save_path=None, 
                                                 df_index=False, 
                                                 **extr_args)
        all_dfs.append(_df)
    df_all_files = pd.concat(all_dfs, ignore_index=True, verify_integrity=True)
    del all_dfs


    # LOAD DATA 
    # 0. select files by initialization time
    # 1. select files by forecast time
    # 2. determine storm mask for the give time
    # 3. compare storm mask to ml preds and uh

    all_ftimes = df_all_files['forecast_time'].values
    ftimes = np.unique(all_ftimes)
    n_ftimes = ftimes.size
    print("# File Times", n_ftimes)

    sel_files_by_ftime = {}
    masks = {}
    wofs_cZH = {}
    wofs_prob = {}
    wofs_uh = {}

    nthreshs = args.nthresholds
    csithreshs = np.linspace(0, 1, nthreshs)
    uhthreshs = np.array([0, .01, .02, .025, .05, .1, .2, .25, .5, 
                          1, 5, 10, 20, 25, 40, 50, 80, 100]) #, 200, 300])
    nthreshs_uh = uhthreshs.size

    NANs = np.zeros((n_ftimes, nthreshs)) #np.full((n_ftimes, nthreshs), np.nan) 
    results = {'tps': NANs.copy(), 'fps': NANs.copy(), #np.full((n_ftimes, nthreshs), np.nan)
               'fns': NANs.copy(), 'tns': NANs.copy(),
               'npos': NANs[:,0].copy(), 'nneg': NANs[:,0].copy()} 
               #'srs': NANs.copy(), 'pods': NANs.copy(), 'csis': NANs.copy()}
    
    NANs = np.zeros((n_ftimes, nthreshs_uh)) #np.full((n_ftimes, nthreshs_uh), np.nan) 
    results_uh = {'tps': NANs.copy(), 'fps': NANs.copy(), 
                  'fns': NANs.copy(), 'tns': NANs.copy(),
                  'npos': NANs[:,0].copy(), 'nneg': NANs[:,0].copy()}
                  #'srs': NANs.copy(), 'pods': NANs.copy(), 'csis': NANs.copy()}
    
    kdtree = None

    prob_distr = np.zeros((0,))
    prob_max = []
    uh_max = []

    of_interest = []

    n_3hrs = 36 #64 

    nf = len(args.forecast_duration)
    fslice = slice(0, n_3hrs)
    if nf == 1:
        fslice = slice(0, args.forecast_duration[0])
    elif nf == 2:
        fslice = slice(*args.forecast_duration)

    # Iterate of WoFS run dates
    for r, rdate in enumerate(rdates): #[3:4] #20190502
        rdate_files = df_all_files.loc[df_all_files['run_date'] == rdate]
        itimes = np.unique(rdate_files['init_time'].values)
        itimes = np.unique(itimes)

        t0 = rdate_files['forecast_time'].min()
        t0 = datetime.strptime(t0, '%Y-%m-%d_%H:%M:%S').strftime('%Y-%m-%d %H:%M:%S')
        t1 = rdate_files['forecast_time'].max()
        t1 = datetime.strptime(t1, '%Y-%m-%d_%H:%M:%S').strftime('%Y-%m-%d %H:%M:%S')
        storms_for_rdate = df_storm_report[t0:t1]
        _latstorm = storms_for_rdate['Lat'].values
        _lonstorm = storms_for_rdate['Lon'].values
        _times = np.array([d.to_pydatetime() for d in pd.to_datetime(storms_for_rdate.index.values)])
        print(storms_for_rdate)

        nstorms = storms_for_rdate.shape[0]
        if nstorms == 0:
            print(f"No storms on {rdate}. nstorms={nstorms}")
            if args.skip_clearday: continue

        # Check coords
        _fn = rdate_files['filename_path'].values[0]
        wofs_preds = xr.load_dataset(_fn)
        xlats = np.squeeze(wofs_preds['XLAT'].values)
        xlons = np.squeeze(wofs_preds['XLONG'].values)
        #dateindex = datetime.strptime(rdate, '%Y%m%d').strftime('%Y-%m-%d')

        _mask, _, _ = create_storm_mask(xlats, xlons, None, 
                                        _latstorm, _lonstorm, 
                                        _times, args.gridsize,
                                        thres_dist=1, 
                                        thres_time=args.thres_time)
        nstorms = np.count_nonzero(_mask)
        if nstorms == 0: 
            print(f"No storms on {rdate} in WoFS spatial domain. nstorms={nstorms}. skip={args.skip_clearday}")
            if args.skip_clearday: continue

        # Iterate over initialization times
        for i, itime in enumerate(itimes): #(['1900', '1930']):
            init_files = rdate_files.loc[rdate_files['init_time'] == itime]
            forecast_times = np.unique(init_files['forecast_time'].values)
            if args.dry_run: print(forecast_times[fslice])

            # Iterate over first 3 hours of forecast times
            for f, ftime in enumerate(forecast_times[fslice]): 
                sel_files_by_ftime[ftime] = select_files(df_all_files, itime, 
                                                         ftime, emember=None,
                                                         rdate=rdate)

                # Load all ensembles for the forecast time
                concat_dim = 'ENS'
                _sel_fnpaths = sel_files_by_ftime[ftime]['filename_path'].values
                wofs_preds = xr.open_mfdataset(_sel_fnpaths, concat_dim=concat_dim,
                                               combine='nested', coords='minimal',
                                               decode_times=False)
                dtime = num2date(wofs_preds.Time[0], 'seconds since 2001-01-01')
                wofs_preds.drop_vars('Time')

                print(f'[{r:2d}, {rdate}]({i:2d}) {f:2d} / {n_3hrs}', itime, ftime, dtime)
                if sel_files_by_ftime[ftime].shape[0] != 18: 
                    print("ENS LOW", sel_files_by_ftime[ftime].shape)
                    of_interest.append(f'{rdate} | {itime} | {ftime} | {dtime}')
                if args.dry_run:
                    print(sel_files_by_ftime[ftime].shape)
                    print(sel_files_by_ftime[ftime]['filename_path'].values)

                # COMPUTE MEAN/MEDIAN
                if args.stat == 'mean':
                    # Use [0] to remove the singleton dimension. analogous to using np.squeeze
                    wofs_prob[ftime] = wofs_preds['ML_PREDICTED_TOR'].mean(dim=concat_dim).values[0]
                    if args.uh_compute:
                        wofs_cZH[ftime] = wofs_preds['COMPOSITE_REFL_10CM'].mean(dim=concat_dim).values[0]
                        wofs_uh[ftime] = wofs_preds['UP_HELI_MAX'].mean(dim=concat_dim).values[0]
                elif args.stat == 'median':
                    wofs_prob[ftime] = wofs_preds['ML_PREDICTED_TOR'].median(dim=concat_dim).values[0]
                    if args.uh_compute:
                        wofs_cZH[ftime] = wofs_preds['COMPOSITE_REFL_10CM'].median(dim=concat_dim).values[0]
                        wofs_uh[ftime] = wofs_preds['UP_HELI_MAX'].median(dim=concat_dim).values[0]
                elif args.stat == 'agg':
                    pass
                    # TODO (maybe): aggregate over all
            
                # CREATE STORM MASK for the given forecast time, distance thres, and time thres
                # Extract lat/lon once per day or init time instead
                #if i == 0:
                xlats = np.squeeze(wofs_preds['XLAT'].values)
                xlons = np.squeeze(wofs_preds['XLONG'].values)

                masks[ftime], sel_storms, kdtree = create_storm_mask(xlats, xlons, dtime, 
                                                                    lats_storm, lons_storm, 
                                                                    times_storms, 
                                                                    args.gridsize, 
                                                                    thres_dist=args.thres_dist, 
                                                                    thres_time=args.thres_time, 
                                                                    kdworkers=-1)#, times_storms_end=times_storms_end)
                npos = np.count_nonzero(masks[ftime])
                if npos == 0: 
                    #{rdate} | {itime}
                    print(f"No tornados at {itime} {ftime} or within the WoFS Domain. npos={npos}")
                    if args.skip_cleartime: continue

                # COMPUTE PERFORMANCE
                #if np.all(~masks[ftime])
                shape = wofs_prob[ftime].shape

                y_storm = masks[ftime] #"ground truth"
                _npos = np.count_nonzero(y_storm)
                _nneg = np.count_nonzero(~y_storm)
                print(f"pos: {_npos}  neg: {_nneg}; total: {_npos + _nneg}")
                if args.dry_run: 
                    classes, counts = np.unique(y_storm, return_counts=True)
                    print("Storm Counts", classes, ":", counts)

                y_preds = wofs_prob[ftime] 

                # Dilate
                if args.ml_probabs_dilation > 0:
                    ksize = args.ml_probabs_dilation
                    y_preds = grey_dilation(wofs_prob[ftime], footprint=footprint) #, size=(ksize, ksize)
                if args.ml_probabs_norm:
                    y_preds = normalize(y_preds.reshape(-1, 1), norm='max', axis=0).reshape(shape)

                # Max probability over time
                if f == 0: prob_max = y_preds
                else: prob_max = np.nanmax(np.stack([y_preds, prob_max], axis=0), axis=0)

                # Calculate performance ML
                tps, fps, fns, tns = contingency_curves(y_storm.ravel(), y_preds.ravel(), csithreshs.tolist())
                results['tps'][f] += tps
                results['fps'][f] += fps
                results['fns'][f] += fns
                results['tns'][f] += tns
                results['npos'][f] += _npos
                results['nneg'][f] += _nneg

                q = np.random.rand(1)
                if q > .9:
                    prob_distr = np.append(prob_distr, y_preds.ravel())

                # UH
                if args.uh_compute:
                    y_uh = wofs_uh[ftime]

                    # Dilate
                    if args.ml_probabs_dilation > 0:
                        ksize = args.ml_probabs_dilation
                        y_uh = grey_dilation(wofs_uh[ftime], footprint=footprint) #, size=(ksize, ksize)
                    if args.ml_probabs_norm:
                        y_uh = normalize(y_uh.reshape(-1, 1), norm='max', axis=0).reshape(shape)

                    # Max UH over time
                    if f == 0: uh_max = y_uh
                    else: uh_max = np.nanmax(np.stack([y_uh, uh_max], axis=0), axis=0)

                    _res = compute_performance(y_storm, y_uh, uhthreshs)
                    results_uh['tps'][f] += _res['tps']
                    results_uh['fps'][f] += _res['fps']
                    results_uh['fns'][f] += _res['fns']
                    results_uh['tns'][f] += _res['tns']
                    results_uh['npos'][f] += _npos
                    results_uh['nneg'][f] += _nneg


    if args.write:
        fn_suffix = f'_{fslice.start:02d}_{fslice.stop:02d}slice'
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
        prefix = ''
        if args.dry_run:
            prefix += '_test_' 
            args.year = rdate
        if args.model_name != '': prefix += f'{args.model_name}_'

        figsize = (15, 10)
        fig, axs = plt.subplots(2, 3, figsize=figsize)
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
        f1 = 2 * tps / (2 * tps + fps + fns)
        acc = (tps + tns) / (tps + fps + fns + tns)

        npos = np.nansum(results['npos'])
        nneg = np.nansum(results['nneg'])
        pos_rate = npos / (npos + nneg)
        print(f"npos: {npos}  nneg: {nneg}; total: {npos + nneg}")
        print(f"acc: {acc}")

        # Write for each model type and time interval
        fname = os.path.join(args.out_dir, f'{prefix}{args.year}_performance_results_{args.stat}{fn_suffix}.csv') 
        if args.dry_run: print(f"Saving {fname}")
        pd.DataFrame({'thres': csithreshs, 'tps': tps, 'fps': fps, 'fns': fns, 
                      'tns': tns, 'srs': srs_agg, 'pods': pods_agg, 'csis': csis_agg,
                      'fpr': fpr, 'f1': f1}).to_csv(fname, header=True)


        
        fname = os.path.join(args.out_dir, f'{prefix}{args.year}_performance_diagrams_{args.stat}{fn_suffix}.png')

        # CSI Performance Diagram
        #fname = os.path.join(args.out_dir, f'{prefix}{args.year}_performance_diagram_{args.stat}{fn_suffix}.png')
        _figax = plot_csi(masks[ftime].ravel(), prob_max.ravel(), #wofs_prob[ftime].ravel(), 
                         fname, threshs=csithreshs, 
                         label=f'ML Probabilities {legend_txt}', color='blue', 
                         save=False, srs_pods_csis=(srs_agg, pods_agg, csis_agg),
                         return_scores=False, fig_ax=(fig, axs[2]), figsize=figsize)
        #cb = fig.colorbar(axs[2].collections[-1], ax=axs[2])
        #cb = axs[2].collections[0].colorbar

        # ROC Curve
        #fname = os.path.join(args.out_dir, f'{prefix}{args.year}_roc_{args.stat}{fn_suffix}.png')
        _figax = plot_roc(masks[ftime].ravel(), prob_max.ravel(), fname, 
                           tpr_fpr=(pods_agg, fpr), fig_ax=(fig, axs[0]), 
                           figsize=figsize, save=False)

        # PRC
        #fname = os.path.join(args.out_dir, f'{prefix}{args.year}_prc_{args.stat}{fn_suffix}.png')
        _figax = plot_prc(masks[ftime].ravel(), prob_max.ravel(), fname, 
                           pre_rec_posrate=(srs_agg, pods_agg, pos_rate), draw_ann=2,
                           fig_ax=(fig, axs[1]), figsize=figsize, save=False)
        
        axs[3].plot(fpr, acc)
        axs[3].set(xlabel='FPR', ylabel='Accuracy')
        axs[3].set(xlim=[0, 1], ylim=[0, 1])
        axs[3].grid()
        
        axs[5].hist(prob_distr, bins=51, density=True)
        axs[5].set(title='ML Probability Distribution', xlabel='ML Probability')
        axs[5].set_xlim([0, .5])

        plt.subplots_adjust(wspace=.05, hspace=.25)
        '''
        # Extract positions of square and above axes
        p1 = axs[1].get_position()
        p3 = axs[3].get_position()
        # make square axes with left position taken from above axes, and set position
        p1 = [p3.x0, p1.y0, p1.width, p1.height]
        axs[1].set_position(p1) 
        '''

        print("Saving plots")
        print(fname)
        #fig.tight_layout(pad=1.5)
        plt.savefig(fname, dpi=160, bbox_inches='tight')

        # TODO: Calibration Diagram
        '''fname = os.path.join(args.out_dir, f'{prefix}{args.year}_calibration_curve_{args.stat}{fn_suffix}.png')
        fig, ax = plot_reliabilty_curve(masks[ftime].ravel(), prob_max.ravel(),  
                                        fname, save=True, strategy='uniform')'''
        
        # UH
        if args.uh_compute:
            tps = np.nansum(results_uh['tps'], axis=0)
            fps = np.nansum(results_uh['fps'], axis=0)
            fns = np.nansum(results_uh['fns'], axis=0)
            tns = np.nansum(results_uh['tns'], axis=0)

            srs_agg = np.nan_to_num(compute_sr(tps, fps))
            pods_agg = np.nan_to_num(compute_pod(tps, fns))
            csis_agg = np.nan_to_num(compute_csi(tps, fns, fps))

            fname = os.path.join(args.out_dir, f'{prefix}{args.year}_performance_results_{args.stat}{fn_suffix}_uh.csv') 
            if args.dry_run: print(f"Saving {fname}")
            pd.DataFrame({'thres': uhthreshs, 'tps': tps, 'fps': fps, 'fns': fns, 
                          'tns': tns, 'srs': srs_agg, 'pods': pods_agg, 'csis': csis_agg}).to_csv(fname, header=True)

            print("**", np.nanquantile(uh_max, [0, 1]))
            fname = os.path.join(args.out_dir, f'{prefix}{args.year}_performance_diagram_{args.stat}{fn_suffix}_uh.png')
            figax = plot_csi(masks[ftime].ravel(), uh_max.ravel(), #wofs_uh[ftime].ravel()
                            fname, threshs=uhthreshs, label=f'UH {legend_txt}', 
                            color='blue', save=True, srs_pods_csis=(srs_agg, pods_agg, csis_agg), 
                            return_scores=False, fig_ax=None, figsize=figsize)

        print(of_interest)

    print('DONE.')
