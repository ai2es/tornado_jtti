"""
author Monique Shotande

** BOOTSTRAP STATISTICAL COMPARISON TEST **

Inputs are the augmented storm reports and ML probabilities. Augmented refers to
dilation based on time (e.g., time 20min window) and space (e.g., 50km radius) of
the storms and the predictions.

Outputs result of DeLong statisical test comparing all models to the UH and 
bootstrap resample of the augmented ground truth reports and the ML predictions


          cooresponding index range
min00-20 :  0,  5
min20-60 :  5, 13
min60-90 : 13, 19
min90-180: 19, 37

"""

import re, os, sys, stat, errno, glob, argparse
from datetime import timedelta, datetime, date, time
from dateutil.parser import parse as parse_date
import time as ttime

import xarray as xr
print("xr version", xr.__version__)
import numpy as np
print("np version", np.__version__)
import pandas as pd
print("pd version", pd.__version__)
from scipy import spatial, stats
from scipy.ndimage import grey_dilation
from sklearn.metrics import confusion_matrix

from netCDF4 import num2date #, Dataset, date2num

import threading
from concurrent.futures import (ThreadPoolExecutor, ProcessPoolExecutor, wait, 
                                as_completed, ALL_COMPLETED)
from multiprocessing import Manager, shared_memory, Pipe, get_context

from tqdm.auto import tqdm

from MLstatkit.stats import Delong_test, Bootstrapping

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
        sys.argv[1:] = args_list
        
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
    parser.add_argument('--forecast_duration', type=int, nargs='+', default=[36],
                       help='List of at most 2 integers. The indices of the \
                       forecast range to evaluate on. For example, for the \
                       forecast range 0 to 20 min, the indices are 0 and 5. ')

    # Skip forecast time vs skip day
    parser.add_argument('--skip_clearday', action='store_true',
        help='Whether to skip days where there are no tornadoes')
    parser.add_argument('--skip_cleartime', action='store_true',
        help='Whether to skip forecast times where there are no tornadoes')
        
    parser.add_argument('--thres_dist', type=float, default=10,
        help='Threshold distance, in kilometers, to use for the storm mask construction')
    
    parser.add_argument('--thres_time', type=float, default=5,
        help='Threshold time, in minutes, to use for the storm mask construction')
    
    parser.add_argument('--stat', type=str, default='median',
        help='Stat to use for encorporating the ensembles into the evaluations. median (default) | mean')
    
    parser.add_argument('--ml_probabs_dilation', type=int, default=0,
        help='Dilate ML probabilities to the given number of pixels') 

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


def compare_rocs(ytrue, ypreds, rownames:list[str], db=False):
    '''
    Pair wise statistical comparison between a list of ROC curves. All curves
    are compared to the first curve ypreds[0].
    https://pmc.ncbi.nlm.nih.gov/articles/PMC2774909/

    Purpose of DeLong's Test
    To assess whether the AUC of one model is significantly different from that of another model.
    It provides a p-value indicating the significance of the difference.
    How DeLong's Test Works
    Steps Involved
    Calculate AUCs: Compute the AUC for both models using their predicted probabilities.
    Set Null Hypothesis: The null hypothesis states that the AUCs of the two models are equal.
    Statistical Testing: Use the DeLong method to derive the test statistic and p-value.

        ytrue: list of ground truth
        ypreds: matrix where each column contains the predictions for each model.
                column 0 is the reference model all other models are compared to
    '''
    # Perform DeLong's test
    zscores = []
    pvals = []
    for i in range(1, ypreds.shape[1]):
        if db: print(f' [compare_rocs()] y0 v {rownames[i-1]}')
        z, p = Delong_test(ytrue, ypreds[:,0], ypreds[:,i])
        zscores.append(z)
        pvals.append(p)

    return pd.DataFrame({'z_score': zscores, 'p_value': pvals}, index=rownames)


###
# Profiling
import time
import functools
import tracemalloc
from typing import Callable, Any, Dict, Tuple, Optional
import psutil

def profile_perf(
    *,
    label: Optional[str] = None,
    log: bool = True,
    logger: Optional[Callable[[str], None]] = None,
    return_stats: bool = False,
    sample_rss_peak: bool = False,
    sample_interval_sec: float = 0.02,
) -> Callable:
    """
    Decorator to profile wall time, CPU time, and memory for a function call.

    Args:
        label: Optional label to include in the log line.
        log: If True, print/log a one-line summary after the call.
        logger: Callable that accepts a string; defaults to print().
        return_stats: If True, return (result, stats_dict). Otherwise return result only.
        sample_rss_peak: If True, sample process RSS in a background thread to estimate peak RSS.
        sample_interval_sec: Sampling interval for RSS peak (seconds).

    Stats dict fields:
        - wall_sec
        - cpu_user_sec, cpu_sys_sec, cpu_total_sec
        - rss_before, rss_after, rss_delta
        - rss_peak_sampled (None if sampling disabled)
        - py_alloc_current, py_alloc_peak (from tracemalloc)
        - io_read_bytes_delta, io_write_bytes_delta (if available)
        - start_time, end_time (epoch seconds)
        - pid
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            proc = psutil.Process(os.getpid())

            # Counters before
            start_time = time.time()
            t0 = time.perf_counter()
            cpu0 = proc.cpu_times()
            rss_before = proc.memory_info().rss
            try:
                io0 = proc.io_counters()
            except Exception:
                io0 = None

            # Tracemalloc setup
            was_tracing = tracemalloc.is_tracing()
            if not was_tracing:
                tracemalloc.start()

            # Optional RSS peak sampling
            peak_rss = rss_before
            stop_flag = False
            def _sampler():
                nonlocal peak_rss, stop_flag
                while not stop_flag:
                    try:
                        peak_rss = max(peak_rss, proc.memory_info().rss)
                    except Exception:
                        pass
                    time.sleep(sample_interval_sec)

            sampler_thread = None
            if sample_rss_peak:
                sampler_thread = threading.Thread(target=_sampler, daemon=True)
                sampler_thread.start()

            # Execute the function
            exc = None
            result = None
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                exc = e
            finally:
                # Stop sampler
                if sample_rss_peak:
                    stop_flag = True
                    if sampler_thread is not None:
                        sampler_thread.join(timeout=1.0)

                # Collect after-counters
                wall_sec = time.perf_counter() - t0
                end_time = time.time()
                cpu1 = proc.cpu_times()
                rss_after = proc.memory_info().rss
                try:
                    io1 = proc.io_counters()
                except Exception:
                    io1 = None

                # Tracemalloc snapshot
                py_current, py_peak = (0, 0)
                try:
                    py_current, py_peak = tracemalloc.get_traced_memory()
                except RuntimeError:
                    # Not tracing; ignore
                    pass
                finally:
                    if not was_tracing:
                        # Only stop if we started it
                        try:
                            tracemalloc.stop()
                        except Exception:
                            pass

            # Build stats
            stats: Dict[str, Any] = {
                "label": label or func.__name__,
                "pid": proc.pid,
                "start_time": start_time,
                "end_time": end_time,
                "wall_sec": wall_sec,
                "cpu_user_sec": getattr(cpu1, "user", 0.0) - getattr(cpu0, "user", 0.0),
                "cpu_sys_sec": getattr(cpu1, "system", 0.0) - getattr(cpu0, "system", 0.0),
                "cpu_total_sec": (getattr(cpu1, "user", 0.0) - getattr(cpu0, "user", 0.0))
                                + (getattr(cpu1, "system", 0.0) - getattr(cpu0, "system", 0.0)),
                "rss_before": rss_before,
                "rss_after": rss_after,
                "rss_delta": rss_after - rss_before,
                "rss_peak_sampled": peak_rss if sample_rss_peak else None,
                "py_alloc_current": py_current,
                "py_alloc_peak": py_peak,
                "io_read_bytes_delta": (io1.read_bytes - io0.read_bytes) if (io0 and io1) else None,
                "io_write_bytes_delta": (io1.write_bytes - io0.write_bytes) if (io0 and io1) else None,
            }

            # Log summary
            if log:
                log_fn = logger or print
                peak_info = f", rss_peak≈{stats['rss_peak_sampled']:,}B" if stats["rss_peak_sampled"] else ""
                log_fn(
                    f"[{stats['label']}] wall={stats['wall_sec']:.3f}s, "
                    f"cpu={stats['cpu_total_sec']:.3f}s (user={stats['cpu_user_sec']:.3f}, sys={stats['cpu_sys_sec']:.3f}), "
                    f"rssΔ={stats['rss_delta']:,}B (before={stats['rss_before']:,}B, after={stats['rss_after']:,}B)"
                    f"{peak_info}, py_peak={stats['py_alloc_peak']:,}B"
                )

            if exc is not None:
                # Re-raise the original exception after measuring
                raise exc

            return (result, stats) if return_stats else result
        return wrapper
    return decorator
###

def do_bootstrap(y_true, youts, metric, n_bootstraps, thres, name):
    #gc.collect()
    print("[do_bootstrap()]", metric, "for", name)
    print(f"[do_bootstrap()] Worker PID: {os.getpid()}")
    print(f"[do_bootstrap()] Python Thread ID: {threading.get_ident()}")
    print(f"[do_bootstrap()] Native Thread ID: {threading.get_native_id()}")

    og_score, confidence_lower, confidence_upper = Bootstrapping(y_true, youts, metric, n_bootstraps=n_bootstraps, threshold=thres)
    print(f"{metric}: {og_score:.3f}, Confidence interval: [{confidence_lower:.3f} - {confidence_upper:.3f}]")

    colname = f'{metric}_{name}'
    stats_result = pd.DataFrame({colname: [confidence_lower, og_score, confidence_upper]},
                                index=['confidence_lower', 'value', 'confidence_upper']) 
    print('[do_bootstrap()]\n', stats_result)
    return stats_result



if "__main__" == __name__:
    args_list = [#'', 
                 #'--dir_wofs_preds', '/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_preds', 
                 #'--dir_wofs_preds', '/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_preds1/tor_unet_sample50_50_classweights20_80_hyper', 
                 '--dir_wofs_preds', '/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_preds1/tor_unet_sample50_50_classweights50_50_hyper', 
                 #'--dir_wofs_preds', '/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_preds1/tor_unet_sample90_10_classweights20_80_hyper', 
                 '--loc_storm_report', '/ourdisk/hpc/ai2es/tornado/tor_spc_ncei/2019_actual_tornadoes_clean.csv', # Paper revision with NCEI due Jul 13 2025
                 #'/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/tornado_reports/tornado_reports_2019_spring.csv', #original
                 #'/ourdisk/hpc/ai2es/tornado/stormreports/processed/tornado_reports_2019.csv', 
                 '--out_dir', './wofs_evaluations/_test_wofs_eval/revisions', 
                 #'--out_dir', '/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_figs/2019/summary/compare_models/',
                 #'--out_dir', '/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_figs/2019/20190430/summary', 
                 '--year', '2019', 
                 #'--forecast_duration', '5', '13',
                 '--skip_clearday',
                 #'--skip_cleartime',
                 '--stat', 'mean', 
                 '-w', '1', 
                 '--dry_run'] 
    args = parse_args(args_list) #
    print(args)

    
    # CONSTRUCT LIST OF DATA FILES
    # Format string for python datetime object
    fmt_dt = '%Y-%m-%d_%H:%M:%S' #'%Y-%m-%d_%H_%M_%S' '%Y-%m-%d_%H_%M_%S'
    # regex pattern string for the datetime format
    fmt_re = r'\d{4}-\d{2}-\d{2}_\d{2}(_|:)\d{2}(_|:)\d{2}'
    extr_args = {'method': 're', 'fmt_dt': fmt_dt, 'fmt_re': fmt_re}

    #dir_preds = os.path.join(args.dir_wofs_preds, args.year)
    #dir_preds_list = os.listdir(dir_preds)
    #rdates = sorted(dir_preds_list)
    #print(f"{len(rdates)} run dates. {rdates}")
    
    # pandas.DateFrame datetime format string
    fmt_dt_pd = '%Y-%m-%d %H:%M:%S'
    storm_timedelta = pd.Timedelta(hours=1) #timedelta(hours=1) #


    ##########
    dirs = np.array(['tor_unet_sample50_50_classweightsNone_hyper',
                     'tor_unet_sample50_50_classweights50_50_hyper',
                     'tor_unet_sample50_50_classweights20_80_hyper',
                     'tor_unet_sample90_10_classweights20_80_hyper'])
    
    tuners = ['tor_unet_sample50_50_classweightsNone_hyper', 
              'tor_unet_sample50_50_classweights50_50_hyper',
              'tor_unet_sample50_50_classweights20_80_hyper',
              'tor_unet_sample90_10_classweights20_80_hyper']
    
    models = ['tor_unet_sample50_50_classweightsNone_hyper', 
              'tor_unet_sample50_50_classweights50_50_hyper',
              'tor_unet_sample50_50_classweights20_80_hyper',
              'tor_unet_sample90_10_classweights20_80_hyper',
              'uh']
    
    #tuners_legendtxt = ['S50', 'S50W50', 'S50W80', 'S10W80', 'UH']
    tuners_legendtxt = ['S50', 'S50W50', 'S50W80']
    leadtimes = ['0-20min', '20-60min', '60-90min', '90-180min']
    
    
    ##########
    # For selecting windows of forecast times
    n_3hrs = 37 #64 
    nf = len(args.forecast_duration)
    fslice = slice(0, n_3hrs)
    if nf == 1:
        fslice = slice(0, args.forecast_duration[0])
    elif nf == 2:
        fslice = slice(*args.forecast_duration)


    dir_youts = '/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_figs/2019/summary/'
    fns_3hr = [
        #dir_youts + 'tor_unet_sample50_50_classweightsNone_hyper/_revision_tor_unet_sample50_50_classweightsNone_hyper_2019_youtputs_mean_00_36slice_dilated_skip.csv',
        dir_youts + 'tor_unet_sample50_50_classweightsNone_hyper/from_read_means/tor_unet_sample50_50_classweightsNone_hyper_2019_youtputs_mean_00_36slice_dilated_skip_uh.csv',
        dir_youts + 'tor_unet_sample50_50_classweights50_50_hyper/_revision_tor_unet_sample50_50_classweights50_50_hyper_2019_youtputs_mean_00_36slice_dilated_skip.csv',
        dir_youts + 'tor_unet_sample50_50_classweights20_80_hyper/_revision_tor_unet_sample50_50_classweights20_80_hyper_2019_youtputs_mean_00_36slice_dilated_skip.csv',]
        #dir_youts + 'tor_unet_sample90_10_classweights20_80_hyper/_revision_tor_unet_sample90_10_classweights20_80_hyper_2019_youtputs_mean_00_36slice_dilated_skip.csv']
    
    # ROC/Performance Curves
    fns_roc_csi = [
        dir_youts + 'tor_unet_sample50_50_classweightsNone_hyper/from_read_means/tor_unet_sample50_50_classweightsNone_hyper_2019_performance_results_mean_00_36slice_dilated_skip.csv', 
        dir_youts + 'tor_unet_sample50_50_classweights50_50_hyper/_revision_tor_unet_sample50_50_classweights50_50_hyper_2019_performance_results_mean_00_36slice_dilated_skip.csv',
        dir_youts + 'tor_unet_sample50_50_classweights20_80_hyper/_revision_tor_unet_sample50_50_classweights20_80_hyper_2019_performance_results_mean_00_36slice_dilated_skip.csv',]
        #dir_youts + 'tor_unet_sample90_10_classweights20_80_hyper/_revision_tor_unet_sample90_10_classweights20_80_hyper_2019_performance_results_mean_00_36slice_dilated_skip.csv']

    fns_leadtimes = ['_revision_tor_unet_sample50_50_classweights50_50_hyper_2019_performance_results_mean_00_05slice_dilated_skip_uh.csv',
        '_revision_tor_unet_sample50_50_classweights50_50_hyper_2019_performance_results_mean_05_13slice_dilated_skip_uh.csv',
        '_revision_tor_unet_sample50_50_classweights50_50_hyper_2019_performance_results_mean_13_19slice_dilated_skip_uh.csv',
        '_revision_tor_unet_sample50_50_classweights50_50_hyper_2019_performance_results_mean_19_37slice_dilated_skip.csv']


    def read_youts(fn, nrowsload, read_all_cols):
        ''' Read the csv with the ground truth and predictions'''
        if read_all_cols: 
            column_names = pd.read_csv(fn, nrows=0).columns.tolist()

            col_dtypes = {col: 'float32' for col in column_names}
            col_dtypes['y_true'] = 'int32'

            ys = pd.read_csv(fn, nrows=nrowsload, dtype=col_dtypes, memory_map=True)
        else: 
            ys = pd.read_csv(fn, nrows=nrowsload, dtype='float32', 
                                usecols=lambda col: not col in ['y_true', 'y_uh'])
        return ys

    def read_thres_atmax(fn, col_thres='thres', metric='csis'):
        ''' Get threshold of max CSI (or other metric) to use for Bootstrap 

        Parameters
            col_thres: name of the column with the thresholds
            metric: name of the column with the metric of interest
        Return
            thres_atmax: threshold where the max of the metric occurs
        '''
        #print(fn, ":", pd.read_csv(fn, nrows=0).columns)
        thres_csi = pd.read_csv(fn, usecols=[col_thres, metric])
        thres_atmax = thres_csi.loc[thres_csi[metric].argmax(), col_thres]
        return thres_atmax

    def read_youts_and_thres(fn, nrowsload, read_all_cols, fn_perfcurves, 
                             col_thres='thres', metric='csis', db=True):
        ''' Get the y_true, y_preds, and threshold of the max CSI (or other metric)'''
        if db:
            print(f"[read_youts_and_thres()] Python Thread ID: {threading.get_ident()}")
            print(f"[read_youts_and_thres()] Native Thread ID: {threading.get_native_id()}\n")
        print("\n[YOUTS] Reading", fn)
        youts = read_youts(fn, nrowsload, read_all_cols)
        print("[PERF] Reading", fn_perfcurves, "\n")
        thres = read_thres_atmax(fn_perfcurves, col_thres, metric)
        return youts, thres

    ncpus = int(os.environ['SLURM_CPUS_PER_TASK'])

    youts = []
    threshs = []
    errors = []

    nrowsload = 30_000_000 #50_000_000 #54_709_999
    col_thres = 'thres'
    col_metric = 'csis'

    futures = []
    nfiles = len(fns_3hr)
    # Thread args lists
    nrowsload_t = [nrowsload] * nfiles
    read_all_cols_t = [True] + [False] * (nfiles-1)
    col_thres_t = [col_thres] * nfiles
    col_metric_t = [col_metric] * nfiles

    if len(fns_3hr) > 1:
        with ThreadPoolExecutor(max_workers=ncpus, thread_name_prefix='readcsv_') as thread_executor:
            io_data = thread_executor.map(read_youts_and_thres, fns_3hr, nrowsload_t,
                                          read_all_cols_t, fns_roc_csi, col_thres_t, col_metric_t)
            for ys, thres in io_data:
                youts.append(ys)
                threshs.append(thres)

            #youts, threshs = map(list, zip(*list(io_data)))
            #youts = pd.concat(youts, axis='columns')
            '''
            #done, not_done = wait(io_data, return_when=ALL_COMPLETED)
            try:
                youts, threshs = map(list, zip(*list(data)))
                youts = pd.concat(youts, axis='columns')
                print("youts columns: ", youts.columns)
            except Exception as err:
                errors.append(err)   
            '''         
            '''
            for ys_thres in tqdm(as_completed(io_data), total=nfiles, desc='Reading csvs'):
                try:
                    ys, thres = ys_thres
                    youts.append(ys)
                    threshs.append(thres)
                except Exception as err:
                    errors.append(err)
            youts = pd.concat(youts, axis='columns')
            print("youts columns: ", youts.columns)
            '''
            '''
            for i, (fn, fn_perf) in enumerate(zip(fns_3hr, fns_roc_csi)):
                print("submitting", fn)
                read_all_cols = i == 0
                futures.append(thread_executor.submit(read_youts_and_thres, fn, 
                                                    nrowsload, read_all_cols, 
                                                    fn_perf, col_thres, col_metric))
            for fut in tqdm(as_completed(futures), total=nfiles, desc="Reading csvs"):
                try:
                    ys, thres = fut.result()
                    youts.append(ys)
                    threshs.append(thres)
                except Exception as err:
                    print(f"Error reading a file: {err}")
            youts = pd.concat(youts, axis='columns')
            '''
            #ys = read_youts(fn, nrowsload, read_all_cols)
            # Get threshold of max CSI to use for Bootstrap
            #thres_csi = pd.read_csv(fns_roc_csi[i], usecols=['thres', 'csis'])
            #thres_atmax = thres_csi.loc[thres_csi['csis'].argmax(), 'thres']
            #thres_atmax = read_thres_atmax(fn_perf, col_thres, col_metric)
            #ys, thres_atmax = read_youts_and_thres(fn, nrowsload, read_all_cols, 
            #                                       fn_perf, col_thres, col_metric)
        #    threshs.append(thres_atmax)
        #    youts.append(ys)
        youts = pd.concat(youts, axis='columns')
    else: youts = pd.read_csv(fns_3hr[0], nrows=nrowsload) #, dtype='float32'
    print("THREAD ERRS:", errors, flush=True)
    print(youts.head(5))
    print("youts columns: ", youts.columns)
    print("\nThresholds at max CSI:", threshs, "\n")


    # Construct file name prefix and suffix
    fn_suffix = '_00_36slice' if nf == 1 else  f'_{fslice.start:02d}_{fslice.stop:02d}slice'
    legend_txt = ''
    if args.ml_probabs_dilation: 
        fn_suffix += '_dilated'
    if args.skip_cleartime or args.skip_clearday:
        fn_suffix += '_skip'
    prefix = ''
    if args.dry_run:
        prefix += '_bootstrap_' 
    prefix += '_'.join(tuners_legendtxt) + '_' #f'{args.model_name}_'


    # DeLong Test
    y_true = youts['y_true'].values
    print("ytrue cats:", np.unique(y_true))
    mask_nans = np.isnan(y_true)
    print("ytrue NaNs:", np.sum(mask_nans))
    print("First NaN at", np.where(mask_nans)[0])
    y_true = y_true[~mask_nans]
    #y_uh = youts['y_uh']
    _cols = list(youts.columns)
    if 'y_uh' in _cols:
        _cols.remove('y_uh')
        print("removed y_uh col")
        youts = youts[['y_uh'] + _cols]
    y_preds = youts.drop(columns=['y_true']).values
    y_preds = y_preds[~mask_nans]
    model_names = youts.columns[1:]
    print("cols", youts.columns)

    roc_stattest = compare_rocs(y_true, y_preds, rownames=model_names[1:], db=True)

    print(f"\nDeLong Test for ROC Curves {model_names[0]} v {model_names[1:]}")
    print(roc_stattest)

    fn_roc_stattest = os.path.join(args.out_dir, f'{prefix}{args.year}_delong_result_{args.stat}{fn_suffix}.csv') 
    print(f"\n [WRITE={args.write}] Saving {fn_roc_stattest}\n")
    if args.write: roc_stattest.to_csv(fn_roc_stattest, header=True)


    # Calculate confidence intervals for AUROC
    metrics = ['roc_auc', 'pr_auc'] 
    stats_model = pd.DataFrame({score: [pd.NA] * 3 for score in metrics},
                               index=['confidence_lower', 'value', 'confidence_upper']) 

    stats_uh = pd.DataFrame({score: [pd.NA] * 3 for score in metrics},
                            index=['confidence_lower', 'value', 'confidence_upper']) 

    n_bootstraps = 10

    '''
    for s, metric in enumerate(metrics):
        og_score, confidence_lower, confidence_upper = Bootstrapping(y_true, y_preds[:,0], metric, n_bootstraps=n_bootstraps)
        print(f"{metric}: {og_score:.3f}, Confidence interval: [{confidence_lower:.3f} - {confidence_upper:.3f}]")

        stats_model[metric] = [confidence_lower, og_score, confidence_upper]
        
        # UH
        #og_score, confidence_lower, confidence_upper = Bootstrapping(y_true, y_uh, score, n_bootstraps=n_bootstraps)
        #print(f"{metric}: {og_score:.3f}, Confidence interval (UH): [{confidence_lower:.3f} - {confidence_upper:.3f}]")
        
        #stats_uh[metric] = [confidence_lower, og_score, confidence_upper]

    '''

    nmodels = len(model_names)
    nmetrics = len(metrics)
    njobs_todo = nmodels * nmetrics
    metrics_p = np.repeat(metrics, nmodels) #[f'{m}_{model_names[0]}' for m in metrics]
    nbootstraps_p = [n_bootstraps] * njobs_todo
    names_p = np.repeat(model_names, nmetrics)
    thres_p = np.repeat(threshs, nmetrics)
    youts_p = []
    for c in range(nmodels):
        youts_p += [y_preds[:,c]] * nmetrics
    print("process metrics list", metrics_p)
    print("process model names", names_p)
    print("process thresholds", thres_p, '\n')

    nprocesses = int(os.environ['SLURM_NTASKS']) #SLURM_NPROCS
    print(nprocesses, "workers")

    t0 = ttime.perf_counter()
    with ProcessPoolExecutor(max_workers=nprocesses) as process_executor:
        cpu_results = process_executor.map(do_bootstrap, [y_true] * njobs_todo, 
                                           youts_p,
                                           #[y_preds[:,0], y_preds[:,0], 
                                           # y_preds[:,1], y_preds[:,1]], 
                                            metrics_p, nbootstraps_p, thres_p, names_p)
        
        '''
        while True:
            #import psutil, os
            #print(psutil.Process(os.getpid()).memory_info().rss / 1024**2, "MB")
            
            running = [f for f in cpu_results if f.running()]
            done = [f for f in cpu_results if f.done()]
            cancelled = [f for f in cpu_results if f.cancelled()]
            pending = [f for f in cpu_results if not f.running() and not f.done()]

            print(f"🔄 Running: {len(running)} | ✅ Done: {len(done)} | ⏳ Pending: {len(pending)} | ❌ Cancelled: {len(cancelled)}")

            if len(done) == len(cpu_results):
                break

            time.sleep(30)
            
            #import psutil, os
            #print(psutil.Process(os.getpid()).memory_info().rss / 1024**2, "MB")

        '''
        
        df_final = pd.concat(list(cpu_results), axis='columns')
        print("\nCPU result\n", df_final.T)
        fn_bs_results = os.path.join(args.out_dir, f'{prefix}{args.year}_performance_bootstrap_{args.stat}{fn_suffix}.csv')
        print(f"\n [WRITE={args.write}] Saving {fn_bs_results}\n")
        if args.write: df_final.to_csv(fn_bs_results, header=True)

    t1 = ttime.perf_counter()
    dtime = (t1 - t0) / 60 
    print(f"Bootstrap elapsed: {dtime:.03f} min")

    #fn_stats = os.path.join(args.out_dir, f'{prefix}{args.year}_performance_bootstrap_{args.stat}{fn_suffix}.csv') 
    #print(f" [WRITE={args.write}] Saving performance measure confidence intervals (model)", fn_stats)
    #print(stats_model)
    #if args.write: stats_model.set_index('label').to_csv(fn_stats, index=False)
    
    #fn_stats = os.path.join(args.out_dir, f'{prefix}{args.year}_performance_bootstrap_intervals_{args.stat}{fn_suffix}_uh.csv') 
    #print(f" [WRITE={args.write}] Saving performance measure confidence intervals (uh)", fn_stats)
    #print(stats_uh)
    #if args.write: stats_uh.set_index('label').to_csv(fn_stats, index=False)


    print('DONE.')
