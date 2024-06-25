"""
OLD DEPREICATED
works but use for a single day and the old storm reports file format (i.e. /ourdisk/hpc/ai2es/alexnozka/tornado_reports/190430_rpts_filtered_torn.csv)


author Monique Shotande

Pixel-wsie Evaluation of WoFS predictions.
Generates a pandas dataframe of all the files for convenience
when selecting all ensembles for specific forecast time, 
initialization time, and date.

Mean and median UH and ML probabilities can be computed across the 
ensembles. Results for UH can be optionally computed with the 
argument flag, --uh_compute.
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
#import scipy
#print("scipy version", scipy.__version__)
from scipy import spatial
from scipy.ndimage import grey_dilation
from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn.metrics import confusion_matrix

#import wrf #wrf-python=1.3.2.5==py38h0e9072a_0
#print("wrf-python version", wrf.__version__)
#import metpy
#import metpy.calc

#import netCDF4
#print("netCDF4 version", netCDF4.__version__)
from netCDF4 import Dataset, date2num, num2date

#import tensorflow as tf
#print("tensorflow version", tf.__version__)
#from tensorflow import keras
#print("keras version", keras.__version__)

# Working directory expected to be tornado_jtti/
sys.path.append("lydia_scripts")
#from wofs_raw_predictions import load_wofs_file
from wofs_ensemble_predictions import extract_datetime, select_files
from scripts_tensorboard.unet_hypermodel import plot_reliabilty_curve, plot_csi
from scripts_tensorboard.unet_hypermodel import compute_csi, compute_sr, compute_pod, contingency_curves


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

'''import plotly.express as px
import plotly.offline as py
import plotly.graph_objs as go'''
#from plotly.subplots import make_subplots



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

    parser.add_argument('--loc_storm_report0', type=str, #required=True, 
        help='List of paths to the csv file that lists the storm reports')
    parser.add_argument('--loc_storm_report1', type=str, #required=True, 
        help='List of paths to the csv file that lists the storm reports')

    parser.add_argument('--out_dir', type=str, required=True, 
        help='Directory to store the results. By default, stored in loc_init directory')
    
    parser.add_argument('--date0', type=int, nargs=3, required=True,
                       help='Date of the corresponding storm report as a list of 3 ints. (YEAR, MONTH, DAY)')
    parser.add_argument('--date1', type=int, nargs=3, required=True,
                       help='Date of the corresponding storm report as a list of 3 ints. (YEAR, MONTH, DAY)')

    #parser.add_argument('--inits', type=str, nargs='+', #required=True, 
    #    help='Space delimited list of strings of the initialization times containing subdirectories for all ensembles and forecasts')

    #parser.add_argument('--for_ens', action='store_true',
    #    help='Create individual evaluations for each ensemble, as opposed to across the ensembles')
    
    parser.add_argument('--uh_compute', action='store_true',
        help='Whether to compute performance results for the UH')
    parser.add_argument('--uh_thres_list', type=float, nargs='+',
        help='List of UH threshold values')
    
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

    df_all_inits.sort_values(['run_date', 'init_time', 'ensemble_member',
                              'forecast_time', 'filename_path'], inplace=True)

    # For instances where there are wrfout and wrfwof files, drop the first (i.e. the wrfout)
    df_all_inits.drop_duplicates(subset=['run_date', 'init_time', 
                                         'ensemble_member', 'forecast_time'], 
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
    # Determine storm reports within the specified time range
    delta_time = timedelta(minutes=thres_time)
    time_range = [dtime - delta_time, dtime + delta_time]
    
    mask_reports_by_time = np.logical_and(times_storms >= time_range[0], times_storms <= time_range[1])
    if times_storms_end is not None:
        mask_end_time = np.logical_and(times_storms_end >= time_range[0], times_storms_end <= time_range[1])
        mask_reports_by_time = np.logical_or(mask_reports_by_time, mask_end_time)

    # Compute the distance in terms of the number of grid cells
    thres_ngrids = np.max([thres_dist / gridsize, eps]) 

    # Compute mean difference between latitude and longitude grid lines
    delta_lat = np.diff(lats, n=1, axis=0).mean()
    delta_lon = np.diff(lons, n=1, axis=1).mean()

    # Convert the thres distance in km into lat/lon degree distances
    radius_lat = thres_ngrids * delta_lat
    radius_lon = thres_ngrids * delta_lon
    dlens = [radius_lat, radius_lon]
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
        # For each storm point, find k(=4) closest grid points
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
            print(f'[{i}] ymax: {y_max}, thres: {t}')
            print(f'\t tn: {tns[i]}  fp: {fps[i]}  fn: {fns[i]}  tp:{tps[i]}')
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


"""
Plotting
"""
def plot_storms(mask, lons_storm, lats_storm, thres_dist=None, thres_time=None,
                figax=None, figsize=(10, 5), xlims=None, ylims=None, 
                invert_yaxis=False, bcmap=mpl.colors.ListedColormap(['w', 'k'])):
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
                 '--dir_wofs_preds', '/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_preds1/tor_unet_sample50_50_classweights20_80_hyper', 
                 #'--dir_wofs_preds', '/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_preds1/tor_unet_sample50_50_classweights50_50_hyper', 
                 #'--dir_wofs_preds', '/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_preds1/tor_unet_sample90_10_classweights20_80_hyper', 
                 '--loc_storm_report0', '/ourdisk/hpc/ai2es/alexnozka/tornado_reports/190430_rpts_filtered_torn.csv', #190429_rpts_filtered_torn.csv
                 '--loc_storm_report1', '/ourdisk/hpc/ai2es/alexnozka/tornado_reports/190501_rpts_filtered_torn.csv', 
                 '--out_dir', '.', 
                 #'--out_dir', '/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_figs/2019/20190430/summary', 
                 '--date0', '2019', '4', '30', 
                 '--date1', '2019', '5', '1', #'--uh_compute', 
                 '--thres_dist', '50', '--thres_time', '20', '--stat', 'median', 
                 '--nthresholds', '51', '--ml_probabs_dilation', '33', '--model_name', 'tor_unet_sample50_50_classweights20_80_hyper',
                 '-w', '1', '--dry_run'] 
                 #, '--ml_probabs_norm', 
                 #'--uh_thres_list', '0', '200'  #'363'
    args = parse_args() #args_list)

    # LOAD STORM REPORTS 
    # Storm reports for 20190430
    YEAR = args.date0[0]
    MONTH = args.date0[1]
    DAY = args.date0[2]

    df_storm_report, date_storm_report = load_storm_report(args.loc_storm_report0, 
                                                           YEAR, MONTH, DAY)
    lats_storm = df_storm_report['Lat'].values 
    lons_storm = df_storm_report['Lon'].values

    # Convert npdate64 values into datetime objects
    # TODO: speed up option using existing npdate64 values
    times_storms = np.array([d.to_pydatetime() for d in pd.to_datetime(df_storm_report['DateTime'].values)])
    
    # TODO: create one big storm reports file
    # Storm reports for 20190501
    YEAR1 = args.date1[0]
    MONTH1 = args.date1[1]
    DAY1 = args.date1[2]

    df_storm_report1, _ = load_storm_report(args.loc_storm_report1, YEAR1, MONTH1, DAY1)
    df_storm_report = pd.concat([df_storm_report, df_storm_report1])

    lats_storm = np.concatenate([lats_storm, df_storm_report1['Lat'].values])
    lons_storm = np.concatenate([lons_storm, df_storm_report1['Lon'].values])

    # Convert npdate64 values into datetime objects
    # TODO: speed up option using existing npdate64 values
    times_storms1 = np.array([d.to_pydatetime() for d in pd.to_datetime(df_storm_report1['DateTime'].values)])
    times_storms = np.concatenate([times_storms, times_storms1])


    # TODO: pull IEM polygons
    # def generate_polygon_mask(polygons)
    # input: list of polygons
    # output: new ground truth mask

    
    # CONSTRUCT LIST OF DATA FILES
    fmt_dt = '%Y-%m-%d_%H:%M:%S' #'%Y-%m-%d_%H_%M_%S' '%Y-%m-%d_%H_%M_%S'
    fmt_re = r'\d{4}-\d{2}-\d{2}_\d{2}(_|:)\d{2}(_|:)\d{2}'
    extr_args = {'method': 're', 'fmt_dt': fmt_dt, #args.extract_date_method, args.fmt_dt, args.fmt_re}
                 'fmt_re': fmt_re}
    DATE = f'{YEAR}{MONTH:02d}{DAY:02d}'
    dir_preds = os.path.join(args.dir_wofs_preds, str(YEAR))
    '''
    INIT_TIME = '1900'
    df_all_files = generate_ensemble_files_list_all(dir_preds, DATE, 
                                                    INIT_TIME, ens_names=[], 
                                                    save_path=None, #TODO; args.out_dir
                                                    df_index=False, **extr_args)
    '''
    df_all_files = generate_init_times_files_list_all(dir_preds, DATE, 
                                                      save_path=None, 
                                                      df_index=False, 
                                                      **extr_args)
    
    # LOAD DATA 
    # TODO: args: select files by season
    # 0. select files by initialization time
    # 1. select files by forecast time
    # 2. determine storm mask for the give time
    # 3. compare storm mask to ml preds and uh
    # TODO: check 4 / 73 2019-04-30_19:25:00 (17, 5)

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
    uhthreshs = np.array([0, .01, .02, .025, .05, .1, .2, .25, .5, 1, 5, 10, 20, 25, 40, 50, 80, 100, 200, 300])
    nthreshs_uh = uhthreshs.size
    #np.append(np.array([0]), np.logspace(-4, 0, nthreshs-1, base=10))

    NANs = np.full((n_ftimes, nthreshs), np.nan) 
    results = {'tps': NANs.copy(), 'fps': NANs.copy(), #np.full((n_ftimes, nthreshs), np.nan)
               'fns': NANs.copy(), 'tns': NANs.copy()} 
               #'srs': NANs.copy(), 'pods': NANs.copy(), 'csis': NANs.copy()}
    
    NANs = np.full((n_ftimes, nthreshs_uh), np.nan) 
    results_uh = {'tps': NANs.copy(), 'fps': NANs.copy(), 
                  'fns': NANs.copy(), 'tns': NANs.copy()}
                  #'srs': NANs.copy(), 'pods': NANs.copy(), 'csis': NANs.copy()}
    
    kdtree = None

    prob_max = []
    uh_max = []

    itimes = df_all_files['init_time'].values
    itimes = np.unique(itimes)
    n_3hrs = 36 #64 #0:5 (0-20min); 5:12 (20-60min); 12:18 (60-90min); 18:36 (90-180min)
    #/ourdisk/hpc/ai2es/tornado/stormreports/processed/tornado_reports_2019.csv

    # Iterate over initialization times
    for i, itime in enumerate(itimes): #['1900']
        init_files = df_all_files.loc[df_all_files['init_time'] == itime]
        forecast_times = np.unique(init_files['forecast_time'].values)
        print(forecast_times[:n_3hrs])

        # Iterate over first 3 hours of forecast times
        for f, ftime in enumerate(forecast_times[:n_3hrs]):
            sel_files_by_ftime[ftime] = select_files(df_all_files, itime, 
                                                     ftime, emember=None)

            # Load all ensembles for the forecast time
            concat_dim = 'ENS'
            _sel_fnpaths = sel_files_by_ftime[ftime]['filename_path'].values
            wofs_preds = xr.open_mfdataset(_sel_fnpaths, concat_dim=concat_dim,
                                           combine='nested', coords='minimal',
                                           decode_times=False)
            dtime = num2date(wofs_preds.Time[0], 'seconds since 2001-01-01')
            wofs_preds.drop_vars('Time')

            print(f'{f} / {n_3hrs}', itime, ftime, dtime)
            print(sel_files_by_ftime[ftime].shape)
            if args.dry_run:
                print(sel_files_by_ftime[ftime])
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
            # TODO (Maybe unnecessary): extract lat/lon once per day instead
            xlats = np.squeeze(wofs_preds['XLAT'].values)
            xlons = np.squeeze(wofs_preds['XLONG'].values)
            #dtime = dt_objs[ti]
            masks[ftime], sel_storms, kdtree = create_storm_mask(xlats, xlons, dtime, 
                                                                lats_storm, lons_storm, 
                                                                times_storms, 
                                                                args.gridsize, 
                                                                thres_dist=args.thres_dist, 
                                                                thres_time=args.thres_time, 
                                                                kdworkers=-1)

            # COMPUTE PERFORMANCE
            #if np.all(~masks[ftime])
            shape = wofs_prob[ftime].shape

            y_storm = masks[ftime] #"ground truth" 

            y_preds = wofs_prob[ftime] 

            # Dilate
            if args.ml_probabs_dilation > 0:
                ksize = args.ml_probabs_dilation
                y_preds = grey_dilation(wofs_prob[ftime], size=(ksize, ksize))
            if args.ml_probabs_norm:
                y_preds = normalize(y_preds.reshape(-1, 1), norm='max', axis=0).reshape(shape)

            # Max probability over time
            if f == 0: prob_max = y_preds
            else: prob_max = np.nanmax(np.stack([y_preds, prob_max], axis=0), axis=0)

            # Calculate performance ML
            tps, fps, fns, tns = contingency_curves(y_storm.ravel(), y_preds.ravel(), csithreshs.tolist())
            results['tps'][f] = tps
            results['fps'][f] = fps
            results['fns'][f] = fns
            results['tns'][f] = tns
            #results['srs'][f] = compute_sr(tps, fps) #tps / (tps + fps)
            #results['pods'][f] = compute_pod(tps, fns) #tps / (tps + fns)
            #results['csis'][f] = compute_csi(tps, fns, fps) #tps / (tps + fns + fps)

            # UH
            if args.uh_compute:
                y_uh = wofs_uh[ftime]

                # Dilate
                if args.ml_probabs_dilation > 0:
                    ksize = args.ml_probabs_dilation
                    y_uh = grey_dilation(wofs_uh[ftime], size=(ksize, ksize))
                if args.ml_probabs_norm:
                    y_uh = normalize(y_uh.reshape(-1, 1), norm='max', axis=0).reshape(shape)

                # Max UH over time
                if f == 0: uh_max = y_uh
                else: uh_max = np.nanmax(np.stack([y_uh, uh_max], axis=0), axis=0)

                _res = compute_performance(y_storm, y_uh, uhthreshs)
                results_uh['tps'][f] = _res['tps']
                results_uh['fps'][f] = _res['fps']
                results_uh['fns'][f] = _res['fns']
                results_uh['tns'][f] = _res['tns']
                #results_uh['srs'][f] = _res['srs'] 
                #results_uh['pods'][f] = _res['pods']
                #results_uh['csis'][f] = _res['csis'] 
                '''
                #uh_norm = normalize(y_uh.reshape(-1, 1), norm='max', axis=0).reshape(shape)
                #uh_Normalizer = MinMaxScaler(norm='max').fit(y_uh)
                #uh_norm = uh_Normalizer.transform(y_uh) # (y_uh - args.uh_min) / (args.uh_max - args.uh_min) #* (max - min) + min
                uh_norm = normalize(y_uh.reshape(-1, 1), norm='max', axis=0).reshape(shape)
                tps, fps, fns, tns = contingency_curves(y_storm, uh_norm, uhthreshs.tolist())
                results_uh['tps'][f] = tps
                results_uh['fps'][f] = fps
                results_uh['fns'][f] = fns
                results_uh['tns'][f] = tns
                results_uh['srs'][f] = compute_sr(tps, fps) 
                results_uh['pods'][f] = compute_pod(tps, fns) 
                results_uh['csis'][f] = compute_csi(tps, fns, fps) 
                if args.dry_run:
                    print("# tps nan, 0s", np.count_nonzero(np.isnan(tps)), np.count_nonzero(tps == 0))
                    print("# fps nan, 0s", np.count_nonzero(np.isnan(fps)), np.count_nonzero(fps == 0))
                    print("# fns nan, 0s", np.count_nonzero(np.isnan(fns)), np.count_nonzero(fns == 0))
                    print("# tns nan, 0s", np.count_nonzero(np.isnan(tns)), np.count_nonzero(tns == 0))
                '''


    if args.write:
        fn_suffix = ''
        if args.ml_probabs_dilation: fn_suffix += '_dilated'
        if args.ml_probabs_norm: fn_suffix += '_normed'
        prefix = '_test_' if args.dry_run  else ''
        #if args.model_name != '': prefix += f'{args.model_name}_'

        # TODO: write for each model type and time interval
        #fname = os.path.join(args.out_dir, f'{prefix}performance_results_{args.stat}{fn_suffix}.png') 
        #pd.Dataframe(results).to_csv(fname)
        #fname = os.path.join(args.out_dir, f'{prefix}performance_results_{args.stat}{fn_suffix}_uh.png') 
        #pd.Dataframe(results_uh).to_csv(fname)

        figsize = (8, 8)

        tps = np.nansum(results['tps'], axis=0)
        fps = np.nansum(results['fps'], axis=0)
        fns = np.nansum(results['fns'], axis=0)
        tns = np.nansum(results['tns'], axis=0)

        srs_agg = compute_sr(tps, fps) 
        pods_agg = compute_pod(tps, fns) 
        csis_agg = compute_csi(tps, fns, fps)

        fname = os.path.join(args.out_dir, f'{prefix}performance_diagram_{args.stat}{fn_suffix}.png')
        figax = plot_csi(masks[ftime].ravel(), prob_max.ravel(), #wofs_prob[ftime].ravel(), 
                    fname, threshs=csithreshs, label='Storms v ML', color='blue', 
                    save=True, srs_pods_csis=(srs_agg, pods_agg, csis_agg), 
                    return_scores=False, fig_ax=None, figsize=figsize)
        title = rf'dist $<$ {args.thres_dist}km, time $<$ {args.thres_time}'
        #plt.title(title)

        #fig, ax = plot_reliabilty_curve(y_train.ravel(), xtrain_preds.ravel(),  
        #                                fname, save=False, strategy='uniform')
        
        # UH
        if args.uh_compute:
            tps = np.nansum(results_uh['tps'], axis=0)
            fps = np.nansum(results_uh['fps'], axis=0)
            fns = np.nansum(results_uh['fns'], axis=0)
            tns = np.nansum(results_uh['tns'], axis=0)

            srs_agg = np.nan_to_num(compute_sr(tps, fps))
            pods_agg = np.nan_to_num(compute_pod(tps, fns))
            csis_agg = np.nan_to_num(compute_csi(tps, fns, fps))

            print("**", np.nanquantile(uh_max, [0, 1]))
            fname = os.path.join(args.out_dir, f'{prefix}performance_diagram_{args.stat}{fn_suffix}_uh.png')
            figax = plot_csi(masks[ftime].ravel(), uh_max.ravel(), #wofs_uh[ftime].ravel()
                            fname, threshs=uhthreshs, label='Storms v UH', 
                            color='blue', save=True, srs_pods_csis=(srs_agg, pods_agg, csis_agg), 
                            return_scores=False, fig_ax=None, figsize=figsize)

