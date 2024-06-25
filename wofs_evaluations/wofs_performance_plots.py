"""
author: Monique Shotande

Read performance results from multiple csv files and plot them in the same figure
Compare model configurations and lead times
"""

import re, os, sys, errno, glob, argparse
import pickle

import xarray as xr
print("xr version", xr.__version__)
import numpy as np
print("np version", np.__version__)
import pandas as pd
print("pd version", pd.__version__)


# Working directory expected to be tornado_jtti/
sys.path.append("lydia_scripts")
from scripts_tensorboard.unet_hypermodel import plot_reliabilty_curve, plot_csi
from scripts_tensorboard.unet_hypermodel import compute_csi, compute_sr, compute_pod
from scripts_tensorboard.unet_hypermodel import plot_roc

sys.path.append("wofs_evaluations")
from wofs_preds_evaluations import det_curve


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
        sys.argv = args_list
        
    parser = argparse.ArgumentParser(description='WoFS ensemble evals', epilog='AI2ES')

    parser.add_argument('--files_wofs_eval_results', type=str, nargs='*', required=True, 
        help='List of results files to read')

    parser.add_argument('--in_dir', type=str, required=True, 
        help='Directory containing results directories to read results files from')
    
    parser.add_argument('--out_dir', type=str, required=True, 
        help='Directory to store the results. By default, stored in loc_init directory')

    parser.add_argument('--legend_txt', type=str, #required=True,
                        help='Text to use in the legend for the performance curve')
    parser.add_argument('--ax_title', type=str, #required=True,
                       help='Title for the figure')
    
    parser.add_argument('--stat', type=str, default='median',
        help='Stat to use for encorporating the ensembles into the evaluations. median (default) | mean')
    parser.add_argument('--dilate', action='store_true', help='Whether results were dilated')
    parser.add_argument('--uh', action='store_true', help='Generate figures for UH')
    parser.add_argument('--leadtimes', action='store_true', 
                        help='Generate figures comparing lead times')
    parser.add_argument('--skip_clearday', action='store_true',
                        help='Whether to skip days where there are no tornadoes')

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
    args_list = ['', 
                 '--files_wofs_eval_results', '',
                 #'/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_figs/2019/summary/tor_unet_sample90_10_classweights20_80_hyper/2019_performance_results_mean_00_36slice.csv', 
                 '--in_dir', '/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_figs/2019/summary/',
                 '--out_dir', './wofs_evaluations/_test_wofs_eval', 
                 '--legend_txt', 'lgd_txt', 
                 '--ax_title', 'ax_title', 
                 '--stat', 'mean', 
                 '--dilate',
                 #'--uh',
                 '--leadtimes',
                 '--skip_clearday',
                 #'--model_name', '',
                 '-w', '1', '--dry_run'] 

    args = parse_args(args_list)
    #https://matplotlib.org/stable/users/explain/artists/transforms_tutorial.html
    #bbox_transform=ax.transAxes

    dir_results = args.in_dir #'/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_figs/2019/summary/'
    dirs = ['tor_unet_sample50_50_classweightsNone_hyper',
            'tor_unet_sample50_50_classweights50_50_hyper',
            'tor_unet_sample50_50_classweights20_80_hyper',
            'tor_unet_sample90_10_classweights20_80_hyper']
    
    tuners = ['tor_unet_sample50_50_classweightsNone_hyper', 
              'tor_unet_sample50_50_classweights50_50_hyper',
              'tor_unet_sample50_50_classweights20_80_hyper',
              'tor_unet_sample90_10_classweights20_80_hyper',
              'uh']
    
    tuners_legendtxt = ['Sampling: 50/50 \nClass Weighting: None', 
              'Sampling: 50/50 \nClass Weighting: 50/50',
              'Sampling: 50/50 \nClass Weighting: 20/80',
              'Sampling: 90/10 \nClass Weighting: 20/80',
              'UH']
    leadtimes = ['0-20min', '20-60min', '60-90min', '90-180min']
    legend_txts = tuners_legendtxt

    #'2019_performance_results_mean_00_36slice_dilated_uh.csv'
    #fn_lists = args.files_wofs_eval_results
    suffix = '_dilated' if args.dilate  else ''
    suffix += '_skip' if args.skip_clearday  else ''
    suffix += '_uh' if args.uh  else ''
    
    fn_lists = [os.path.join(dir_results, d, f'{d}_2019_performance_results_{args.stat}_00_36slice{suffix}.csv') for d in dirs] 
    fn_uh = os.path.join(dir_results, dirs[-1], f'{dirs[-1]}_2019_performance_results_{args.stat}_00_36slice{suffix}_uh.csv')
    fn_lists += [fn_uh]
    

    i_model = 3
    _model = 'uh' if args.uh  else dirs[i_model]
    if args.leadtimes:
        fns = [f'2019_performance_results_{args.stat}_00_05slice{suffix}.csv', #min00_20 
               f'2019_performance_results_{args.stat}_05_13slice{suffix}.csv', #min20_60 
               f'2019_performance_results_{args.stat}_13_19slice{suffix}.csv', #min60_90 
               f'2019_performance_results_{args.stat}_19_37slice{suffix}.csv', #min90_180
               f'2019_performance_results_{args.stat}_00_36slice{suffix}.csv']

        fn_lists = [os.path.join(dir_results, dirs[i_model], f'{dirs[i_model]}_{f}') for f in fns[:-1]] 
        suffix += '_leadtimes'
        legend_txts = leadtimes

    prefix = ''
    if args.dry_run:
        prefix = '_test_'
    if args.leadtimes:
        prefix += f'{_model}_'

    fname = os.path.join(args.out_dir, f'{prefix}2019_performance_diagram_multiple_{args.stat}{suffix}.png')
    fname_zoom = os.path.join(args.out_dir, f'{prefix}2019_performance_diagram_multiple_{args.stat}{suffix}_zoomin.png')
    fname_csv = os.path.join(args.out_dir, f'{prefix}2019_performance_diagram_multiple_{args.stat}{suffix}.csv')
    fname_figobj = os.path.join(args.out_dir, f'{prefix}2019_performance_diagram_multiple_{args.stat}{suffix}_figobj.pkl')

    fig, axs = plt.subplots(2, 2, figsize=(20, 12))
    axs = axs.ravel()
    #figax = (fig, axs[1]) #None
    ccycle = plt.cm.tab10.colors

    nfiles = len(fn_lists)
    columns = ['model', 'csi_max', 'pod_max', 'sr_max', 'f1_max', 'auc', 'auc_prc']
    if args.leadtimes:
        columns = ['leadtime'] + columns
    ncols = len(columns)
    model_scores = np.empty((nfiles, ncols), dtype=object)

    for f, fn in enumerate(fn_lists):
        res = pd.read_csv(fn)
        tps = res['tps']
        fps = res['fps']
        fns = res['fns']
        tns = res['tns']
        fpr = res['fpr']
        fnr = res['fnr']
        npos = tps[0] + fns[0]
        total = tps[0] + fps[0] + fns[0] + tns[0]
        pos_rate = npos / total
        srs = np.nan_to_num(compute_sr(tps, fps))
        pods = np.nan_to_num(compute_pod(tps, fns))
        csis = np.nan_to_num(compute_csi(tps, fns, fps))
        '''srs = res['srs']
        pods = res['pods'] 
        csis = res['csis']'''
        thres = res['thres']

        save = (f == nfiles - 1) 
        csiargs = {'show_cb': save}

        _, _, scores = plot_csi([0, 1], [0, 1], fname, threshs=thres, 
                         label=legend_txts[f], color=ccycle[f], #f'{args.legend_txt}_{f}'
                         save=False, srs_pods_csis=(srs, pods, csis, pos_rate), 
                         return_scores=True, fig_ax=(fig, axs[1]), figsize=(10, 10),
                         pt_ann=False, **csiargs)
        
        idx = scores['index_max_csi']
        csi_max = scores['csis'][idx]
        pod_max = scores['pods'][idx]
        sr_max = scores['srs'][idx]
        idx = scores['index_max_f1']
        f1_max = scores['f1s'][idx]
        auc = scores['auc']

        # TODO: fix legend
        #ax.legend(loc='lower center', bbox_to_anchor=(0.5, -.32)) #[plt0, plt1],  #ax.transData
        
        # ROC Curve
        kwargs = {'label': legend_txts[f]}
        _, _, scores = plot_roc(tps, tns, fname, #obs_distr, prob_distr
                 tpr_fpr=(pods, fpr), fig_ax=(fig, axs[0]), plot_ann=save,
                 save=False, return_scores=True, **kwargs)
        
        auc_roc = scores['auc']
        if args.leadtimes:
            model_scores[f] = [legend_txts[f], _model, csi_max, pod_max, sr_max, f1_max, auc_roc, auc]
        else:
            model_scores[f] = [tuners[f], csi_max, pod_max, sr_max, f1_max, auc_roc, auc]

        # Detection error tradeoff (DET) curve
        det_curve(tps, tns, fpr_fnr=(fpr, fnr), figax=(fig, axs[2]))

        # Calibration Diagram
        #fig, ax = plot_reliabilty_curve(obs_distr, prob_distr, fname, save=False, 
        #                                fig_ax=(fig, axs[3]), strategy='uniform')

        if save: 
            all_labels = [l.get_label() for l in axs[0].lines]
            _, inds = np.unique(all_labels, return_index=True)
            inds = np.sort(inds)
            labels = np.take(all_labels, inds)
            line_handles = np.take(axs[0].lines, inds)
            #tmp = axs[0].lines[0:-1:3]
            axs[0].legend(line_handles, labels, loc='upper left', bbox_to_anchor=(1.01, 1.02))
            
            all_labels = [l.get_label() for l in axs[1].lines]
            _, inds = np.unique(all_labels, return_index=True)
            inds = np.sort(inds)
            labels = np.take(all_labels, inds)
            line_handles = np.take(axs[1].lines, inds)
            axs[1].legend(line_handles, labels, loc='upper left', bbox_to_anchor=(1.3, 1.02))

            axs[2].legend(legend_txts[:f+1], loc='upper left', bbox_to_anchor=(1.01, 1.02))

            plt.subplots_adjust(wspace=.5)
            plt.savefig(fname, dpi=160, bbox_inches='tight')

    axs[1].set(xlim=[0, .4], ylim=[0, .4])
    plt.savefig(fname_zoom, dpi=160, bbox_inches='tight')

    df = pd.DataFrame(model_scores, columns=columns)
    df.to_csv(fname_csv, index=False)
    print(df)

    # Save interactive fig object
    #with open(fname_figobj, 'wb') as fp: pickle.dump(fig, fp) 
    #pickle.dump(fig, open(fname_figobj, 'wb')) 