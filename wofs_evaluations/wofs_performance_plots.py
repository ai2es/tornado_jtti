"""
author: Monique Shotande

Read performance results from multiple files and plot them in the same figure
"""

import re, os, sys, errno, glob, argparse

import xarray as xr
print("xr version", xr.__version__)
import numpy as np
print("np version", np.__version__)
import pandas as pd
print("pd version", pd.__version__)


# Working directory expected to be tornado_jtti/
sys.path.append("lydia_scripts")
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

    parser.add_argument('--files_wofs_eval_results', type=str, nargs='*', required=True, 
        help='List of results files to read')

    parser.add_argument('--out_dir', type=str, required=True, 
        help='Directory to store the results. By default, stored in loc_init directory')

    parser.add_argument('--legend_txt', type=str, #required=True,
                        help='Text to use in the legend for the performance curve')
    parser.add_argument('--ax_title', type=str, #required=True,
                       help='Title for the figure')
    
    parser.add_argument('--stat', type=str, default='median',
        help='Stat to use for encorporating the ensembles into the evaluations. median (default) | mean')
    parser.add_argument('--dilate', action='store_true',
        help='Whether results were dilated')

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



if "__main__" == __name__:
    # TODO: ML lead times 0-20, 20-60, 60-90, 90-180
    # TODO: ML lead times 0-20, 20-60, 60-90, 90-180 (dialated)
    # TODO: UH lead times 0-20, 20-60, 60-90, 90-180
    # TODO: UH lead times 0-20, 20-60, 60-90, 90-180 (dialated)
    # TODO: ML, UH, ML (dialated), UH (dialated) 3hrs
    # TODO: mean different models ML (dialated)
    # TODO: mean different models UH (dialated)
    # TODO: median different models ML (dialated)
    # TODO: median different models UH (dialated)

    args_list = ['', 
                 '--files_wofs_eval_results', 
                 '/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_figs/2019/summary/tor_unet_sample90_10_classweights20_80_hyper/2019_performance_results_mean_00_36slice.csv', 
                 '--out_dir', '.', 
                 '--legend_txt', 'lgd_txt', 
                 '--ax_title', 'ax_title', 
                 '--stat', 'mean', 
                 '--dilate',
                 #'--model_name', '',
                 '-w', '1', '--dry_run'] 

    args = parse_args(args_list)
    #https://matplotlib.org/stable/users/explain/artists/transforms_tutorial.html
    #bbox_transform=ax.transAxes

    dirs = ['/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_figs/2019/summary/tor_unet_sample50_50_classweightsNone_hyper',
            '/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_figs/2019/summary/tor_unet_sample50_50_classweights50_50_hyper',
            '/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_figs/2019/summary/tor_unet_sample50_50_classweights20_80_hyper']
            #'/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_figs/2019/summary/tor_unet_sample90_10_classweights20_80_hyper']


    #'2019_performance_results_mean_00_36slice_dilated_uh.csv'
    #fn_lists = args.files_wofs_eval_results
    suffix = '_dilated' if args.dilate  else ''
    fns = [f'2019_performance_results_{args.stat}_00_36slice{suffix}.csv',
           f'2019_performance_results_{args.stat}_00_05slice{suffix}.csv', #min00_20 
           f'2019_performance_results_{args.stat}_05_12slice{suffix}.csv', #min20_60 
           f'2019_performance_results_{args.stat}_12_18slice{suffix}.csv', #min60_90 
           f'2019_performance_results_{args.stat}_18_36slice{suffix}.csv'] #min90_180
    
    fn_lists = [os.path.join(d, f'2019_performance_results_{args.stat}_00_36slice{suffix}.csv') for d in dirs[1:]] 


    prefix = ''
    fn_suffix = ''
    if args.dry_run:
        prefix = '_test_'
    #fn_suffix += '_dilated' #_uh_multiple

    fname = os.path.join(args.out_dir, f'{prefix}2019_performance_diagram_multiple_{args.stat}{fn_suffix}.png')
    nfiles = len(fn_lists) #args.files_wofs_eval_results)
    figax = None
    ccycle = plt.cm.tab10.colors #plt.cycler("color", plt.cm.tab10.colors)
    for f, fn in enumerate(fn_lists):
        res = pd.read_csv(fn)
        tps = res['tps']
        fps = res['fps']
        fns = res['fns']
        fps = res['fps']
        srs = np.nan_to_num(compute_sr(tps, fps))
        pods = np.nan_to_num(compute_pod(tps, fns))
        csis = np.nan_to_num(compute_csi(tps, fns, fps))
        '''srs = res['srs']
        pods = res['pods'] 
        csis = res['csis']'''
        thres = res['thres']

        save = (f == nfiles - 1)

        figax = plot_csi([0, 1], [0, 1], fname, threshs=thres, 
                         label=f'{args.legend_txt}_{f}', color=ccycle[f], 
                         save=save, srs_pods_csis=(srs, pods, csis), 
                         return_scores=False, fig_ax=figax, figsize=(10, 10))
        #ax.legend(loc='lower center', bbox_to_anchor=(0.5, -.32)) #[plt0, plt1],  #ax.transData
