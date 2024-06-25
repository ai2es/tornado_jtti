"""
author: Monique Shotande

Combine the predictions for each ensemble forecast time for an initialization 
time into respective gifs

Create gifs for each forecast time across the ensembles using the ensemble, 
mean, median, mode, min, max or other summary stats

"""

import os, re, sys, glob, time, argparse
from datetime import datetime
from copy import deepcopy

import numpy as np
print("np version", np.__version__)

import pandas as pd
print("pd version", pd.__version__)
# Display all pd.DataFrame columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.options.display.width = 1000

import xarray as xr
print("xr version", xr.__version__)

import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import matplotlib as mpl
plt.rcParams.update({"text.usetex": True})
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib import ticker


from wofs_raw_predictions import plot_hist, plot_pcolormesh


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
    grp.add_argument('--loc_date', type=str, #required=True, 
        help='Directory location of date of WoFS runs, with subdirectories for the initialization times, which each contain subdirectories for all ensembles, which contains the predictions for each forecast time. Example directory: wofs_preds/2023/20230602. Example directory structure: wofs_preds/2023/20230602/1900/ENS_MEM_13/wrfwof_d01_2023-06-02_20_30_00_predictions.nc')
    grp.add_argument('--loc_files_list', type=str, #required=True, 
        help='Path to the csv file that lists all the predictions files for the run date')

    #parser.add_argument('-s', '--subject', type=int, help="Interger associated with the subject")

    parser.add_argument('--inits', type=str, nargs='+', #required=True, 
        help='Space delimited list of strings of the initialization times containing subdirectories for all ensembles and forecasts')
    parser.add_argument('--ensembles', type=str, nargs='+', #required=True, 
        help='Space delimited list of ensembles to use')

    parser.add_argument('--for_ens', action='store_true',
        help='Create individual gifs for each ensemble, as opposed to across the ensembles')
    
    parser.add_argument('--min_max', action='store_true',
        help='Use flag to compute the ensemble min and max')
    parser.add_argument('--uh', action='store_true',
        help='Use flag to include UH')
    parser.add_argument('--uh_thres', type=int, default=10,
        help='Threshod UH (updraft helicity) to use for rendering paintball plots of ensemble UH')

    parser.add_argument('--out_dir', type=str, #required=True, 
        help='Directory to store the gif. By default, stored in loc_init directory')

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


def extract_datetime(strg, method='re', fmt_dt='%Y-%m-%d_%H_%M_%S', 
                     fmt_re=r'\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2}'):
    '''
    Extract datetime object and string from a longer string
    Parameters
    ----------
    strg: string. any string with a date (e.g. a file name path)
    method: string. either 'dt' to use date.strptime or 're' to use regular 
            expressions
    fmt_dt: string. python datetime object format or string with datetime format
    fmt_re: string. regex string pattern that corresponds to the datetime format
    '''
    dt_obj = None
    dt_str = None

    if method == 're':
        match = re.search(fmt_re, strg)
        if match is None: return None, None
        dt_str = match.group() #dt_obj.strftime(fmt_dt)
        dt_obj = datetime.strptime(dt_str, fmt_dt) #.date()
    else:
        dt_obj = datetime.strptime(strg, fmt_dt)
        dt_str = dt_obj.strftime(fmt_dt)
    return dt_str, dt_obj

def select_files(df_files, itime, ftime, emember=None, rdate=None, 
                 return_mask=False):
    '''
    Get a list of files to use based in initialization time, forecast time and/or
    the ensemble member
    Parameters
    ----------
    itime: string or int or None. initialization time for the forecasts. if None
            get all files from all init times with the specified forecast time
            or ensemble
    ftime: None or string. common forecast across all ensembles for the selected 
            files. emember can be ignored if ftime is not None
    emember: None, int, or string. ensemble member for the selected files.
            ftime can be ignore if emember is not None
    rdate: str for the date of the wofs run
    '''
    n = df_files.shape[0]
    sel_itime = np.ones((n,), bool)
    if isinstance(itime, (int, str)):
        init_type = type(df_files['init_time'][0]) #df_files.dtypes['init_time']
        # Cast to appropriate type. init times could be stored as int or str in the dataframe
        sel_itime = df_files['init_time'] == init_type(itime)
    elif itime is not None: 
        raise ValueError(f'itime should be None, str or int. but was type {type(itime)}')

    sel_ftime = np.ones((n,), bool)
    if ftime: sel_ftime = df_files['forecast_time'] == ftime

    sel_emember = np.ones((n,), bool)
    if emember: sel_emember = df_files['ensemble_member'] == emember

    sel_rdate = np.ones((n,), bool)
    if rdate: sel_rdate = df_files['run_date'] == rdate

    sel_mask = sel_itime & sel_ftime & sel_emember & sel_rdate
    sel_files = df_files.loc[sel_mask]
    if return_mask: return sel_files, sel_mask
    else: return sel_files

def load_files(fnpath_list):
    '''
    '''
    pass

def paintball_plot(datalist, thres, fig_ax=None, figsize=(8,8), **kwargs):
    '''
    Plot blobs
    '''
    fig = None
    ax = None

    if fig_ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig, ax = fig_ax

    ds_thres = datalist > thres #.to_numpy()
    ds_thres = ds_thres.astype(float)

    nensembles = datalist.shape[0] #datalist.ENS.size

    img_mtx = ds_thres[0]
    img_mtx[img_mtx == 0] = np.nan

    for e in range(nensembles):
        # Get ensemble data
        _t = ds_thres[e] #datalist.to_numpy()
        img_mtx[_t == 1] = e + 1
    _cmap = mpl.colors.ListedColormap(plt.get_cmap('tab20').colors, 
                                      name='from_list', N=nensembles)
    im = ax.imshow(img_mtx, vmin=1, vmax=nensembles, alpha=.5, cmap=_cmap) #'tab20b'
    #, yincrease=True) #, shading='gouraud', **kwargs)
                  
    ax.set_aspect('equal', 'box')
    if 'title' in kwargs.keys():
        ax.set_title(kwargs['title'])
    ax.invert_yaxis()
    
    return fig, ax, im


def create_gif(data, data_contours, fnpath, data_uh=None, stat="", frame_labels=None, 
               nframes=None, interval=50, figsize=(20, 10), dpi=250, 
               tight=False, DB=0, write=False, **kwargs):
    ''' 
    Create a .gif or movie file of the storm data
    matplotlib documentation 
    https://www.c-sharpcorner.com/article/create-animated-gif-using-python-matplotlib/


    @param data: data to plot as heatmap
    @param data_uh: 
    @param fnpath: file name suffix to append to the file name
    @param interval: Delay between frames in milliseconds. See matplotlib 
            FuncAnimation documentation for more information
    @param figsize: tuple with the width and height of the figure
    @param DB: int debug flag to print out additional debug information
    @param kwargs: additional keyword args for the animator
            repeat: bool, default: True. Whether the animation repeats when the 
                    sequence of frames is completed.
            repeat_delay: int, default: 0. The delay in milliseconds between 
                    consecutive animation runs, if repeat is True.
    '''
    #from matplotlib.animation import FuncAnimation, PillowWriter
    #import imageio
    fig = None
    axs = None
    if data_uh is None:
        fig, axs = plt.subplots(1, 2, figsize=figsize)
    else:
        fig, axs = plt.subplots(2, 2, figsize=figsize)
    axs = axs.ravel()
    #nframes = data['patch'].size

    def plotter(data_mesh, data_contour, title, i, ax, cmap="Spectral_r", 
                vmin=0, vmax=60, clevels=[.2, .5], ccolors=['dimgray', 'yellow'],
                clegtitle='Tor Probability'): 
        '''
        :param data_mesh: data to plot in pclolrmesh
        :param data_contour: data to plot as contours
        :param i: index
        '''
        ZH_distr = ax.pcolormesh(data_mesh, cmap=cmap, vmin=vmin, vmax=vmax)

        if data_contour is not None:
            contour = ax.contour(data_contour, levels=clevels, colors=ccolors)

            proxy = [plt.Rectangle((0,0), 1, 1, fc=pc.get_edgecolor()[0]) for i, pc in enumerate(contour.collections)]
            ax.legend(proxy, 
                    [rf'$\geq${q:.2e}% ' if q < .01 else rf'$\geq${q:.02f}' for q in clevels], 
                    loc='upper left', bbox_to_anchor=(1.2, 1), title=clegtitle)
            
        ax.set_aspect('equal', 'box') #.axis('equal') #
        ax.set_title(title)
        #ax.text(0, -3, f'patch {i:03d}')

        return ZH_distr, ax

    vmin = 0
    vmax = 60

    uh_min = 0 #np.nanmin(data_uh)
    uh_max = 40 #np.nanmax(data_uh)

    uh_thres = 10
    if 'uh_thres' in kwargs.keys():
        uh_thres = kwargs.pop('uh_thres')

    def draw_storm_frame(pi):
        '''
        Draw a patch as a single frame
        @param pi: patch index (aka draw frame index)
        '''
        if isinstance(axs, list) or isinstance(axs, np.ndarray):
            for a in axs: a.clear()
        else: axs.clear()

        dt_label = frame_labels[pi] if frame_labels is not None else ""

        # Plot reflectivity
        title = f'{stat} Composite Reflectivity: DT{dt_label}'
        plotter(data[pi], data_contours[pi], title, pi, 
                axs[0], cmap="Spectral_r", vmin=vmin, vmax=vmax)
        
        # Plot probabilities
        pmax = np.max(data_contours[pi])
        title = f'{stat} Tornado Probability: DT{dt_label}'
        plotter(data_contours[pi], None, title, pi, 
                axs[1], cmap="cividis", vmin=0, vmax=pmax)

        # Plot UH
        if data_uh is not None:
            data_uh_stat = deepcopy(data_uh)
            if isinstance(data_uh, tuple):
                data_uh_tuple = deepcopy(data_uh)
                data_uh_stat = data_uh_tuple[0]
                data_uh_ens = data_uh_tuple[1]
                ntimes = len(data_uh_ens)
                nensembles = data_uh_ens[0].shape[0]

                pbargs = {'title': f'ensembles Updraft Helicity: DT{dt_label}\n(thres {uh_thres})'} #'data_contours': None}
                paintball_plot(data_uh_ens[pi], uh_thres, fig_ax=(fig, axs[3]), 
                               **pbargs)
            else:
                # with helicity contours
                title = f'{stat} Updraft Helicity: DT{dt_label}'
                plotter(data_uh_stat[pi], data_uh_stat[pi], title, pi,
                        axs[3], cmap="Spectral_r", clevels=[20, 40],
                        vmin=uh_min, vmax=uh_max, clegtitle='Helicity')
            # with tor probs
            title = f'{stat} Updraft Helicity: DT{dt_label}\n(with Tor Prob Contours)'
            plotter(data_uh_stat[pi], data_contours[pi], title, pi,
                    axs[2], cmap="Spectral_r", vmin=uh_min, vmax=uh_max)
            

        fig.canvas.draw() # draw the canvas, cache renderer
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        WIDTH, HEIGHT = fig.canvas.get_width_height()
        image = image.reshape(HEIGHT, WIDTH, 3) # 3 for RGB
        return image

    #fname = os.path.basename(wofs.filenamepath) + f'__{suffix}.gif'

    # Plot first frame ot get colorbar to render properly
    dt_label = frame_labels[0] if frame_labels is not None else ""
    title = f'{stat} Composite Reflectivity: DT{dt_label}'
    ZH_distr, ax = plotter(data[0], data_contours[0], title, 0,
                           axs[0], cmap="Spectral_r", vmin=vmin, vmax=vmax)
    cb_ZH_distr = fig.colorbar(ZH_distr, ax=ax)
    cb_ZH_distr.set_label('dBZ', rotation=-90, labelpad=30)
    
    # Probabilities
    pmax = np.nanmax(data_contours[0])
    title = f'{stat} Tornado Probability: DT{dt_label}'
    prob, ax = plotter(data_contours[0], None, title, 0, 
                       axs[1], cmap="cividis", vmin=0, vmax=pmax)
    cb = fig.colorbar(prob, ax=ax)
    cb.set_label('p_{tor}', rotation=-90, labelpad=30)

    # UH
    if data_uh is not None: # and len(data_uh) == 2:
        data_uh_stat = deepcopy(data_uh)
        data_uh_ens = None
        if isinstance(data_uh, tuple):
            data_uh_tuple = deepcopy(data_uh)
            data_uh_stat = data_uh_tuple[0]
            data_uh_ens = data_uh_tuple[1]
            ntimes = len(data_uh_ens)
            nensembles = data_uh_ens[0].shape[0]

            pbargs = {'title': f'ensembles Updraft Helicity: DT{dt_label}\n(thres {uh_thres} N={nensembles})'} 
            _, _ax, _im = paintball_plot(data_uh_ens[0], uh_thres, fig_ax=(fig, axs[3]), 
                           **pbargs)
            cb = fig.colorbar(_im, ax=_ax)
            cb.set_label("Ensemble", rotation=270, labelpad=20)
            d = (nensembles-1) / (2*nensembles)
            cbticks = np.arange(1+d, nensembles-d, 2*d)
            cb.set_ticks(cbticks) #np.linspace(1, nensembles, 2*nensembles+1)
            cb.set_ticklabels(np.linspace(1, nensembles, nensembles, dtype=int))
            #cb.ax.yaxis.get_ticklabels()
            #cb.get_ticks()
            #tick_locator = ticker.MaxNLocator(nbins=nensembles)
            #cb.locator = tick_locator
            #cb.update_ticks()
        else:
            # with helicity contours
            title = f'{stat} Updraft Helicity: DT{dt_label}'
            uh_distr, ax = plotter(data_uh_stat[0], data_uh_stat[0], title, 0,
                                axs[3], cmap="Spectral_r", clevels=[20, 40], 
                                vmin=uh_min, vmax=uh_max, clegtitle='Helicity')
            _cb = fig.colorbar(uh_distr, ax=ax)
            _cb.set_label('Helicity', rotation=-90, labelpad=30)
            
        # with tor probs
        title = f'{stat} Updraft Helicity: DT{dt_label}\n(with Tor Prob Contours)'
        uh_distr, ax = plotter(data_uh_stat[0], data_contours[0], title, 0,
                                axs[2], cmap="Spectral_r", 
                                vmin=uh_min, vmax=uh_max)
        _cb = fig.colorbar(uh_distr, ax=ax)
        _cb.set_label('Helicity', rotation=-90, labelpad=30)


    #if tight

    animator = FuncAnimation(fig, draw_storm_frame, frames=nframes, 
                             interval=interval, **kwargs)
    print(" n fig axes", len(fig.axes)) #fig.delaxes(fig.axes[-1])

    if write:
        print("  Saving", fnpath)
        #writer = PillowWriter(fps=30)
        animator.save(fnpath, dpi=dpi) #, writer=writer)

    return fig, animator


if "__main__" == __name__:
    # TODO: gif per ensemble
    #exfile = '/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_preds/2023/20230602/1900/ENS_MEM_13/wrfwof_d01_2023-06-02_20_30_00_predictions_predictions.nc'
    #2019-04-30_22:55:00
    t0 = time.time()

    fmt_dt = '%Y-%m-%d_%H:%M:%S' #'%Y-%m-%d_%H_%M_%S'
    fmt_re = r'\d{4}-\d{2}-\d{2}_\d{2}(_|:)\d{2}(_|:)\d{2}' #r'\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2}'
    args = parse_args(args_list=['', 
                                 #'--loc_date', '/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_preds/2019/20190430',
                                 '--loc_files_list', '/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_preds/2019/20190430/wofs_preds_files.csv',
                                 #'--loc_date', '/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_preds/2023/20230602',
                                 #'--loc_files_list', '/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_preds/2023/20230602/wofs_preds_files.csv',
                                 '--inits', '1900', '1930', '2000',
                                 '--for_ens',
                                 '--uh',
                                 '--uh_thres', '80',
                                 '--out_dir', '/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_figs/2019/20190430',
                                 #'--out_dir', '/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_figs/2023/20230602',
                                 '--write', '1'])

    print(args)

    # Create a table of files, with columns: initialization time, ensemble member, and forecast time
    df_files = []
    if args.loc_date: 
        rootpath_parts = args.loc_date.split('/')
        path_parts = list(filter(lambda x: x != '', rootpath_parts))

        for init_time in os.listdir(args.loc_date):
            dir_init_time = os.path.join(args.loc_date, init_time)
            if not os.path.isdir(dir_init_time): continue
            print(dir_init_time)

            for ensemble in os.listdir(dir_init_time):
                dir_ensemble = os.path.join(dir_init_time, ensemble)
                if not os.path.isdir(dir_ensemble): continue
                if not 'ENS' in dir_ensemble: continue
                print(dir_ensemble)

                wofs_preds = os.listdir(dir_ensemble)
                for wp_file in wofs_preds:
                    wofs_fnpath = os.path.join(dir_ensemble, wp_file)
                    if not os.path.isfile(wofs_fnpath): continue
                    if 'wrfout' in wofs_fnpath: continue
                    print(wofs_fnpath)
                    forecast_time, _ = extract_datetime(wp_file, method='re', 
                                                        fmt_dt=fmt_dt,
                                                        fmt_re=fmt_re)
                    
                    #wofs_fnpath = os.path.join(dir_ensemble, wp_file)
                    entry = {'run_date': rootpath_parts[-1], 'init_time': int(init_time),
                            'ensemble_member': ensemble, 'forecast_time': forecast_time,
                            'filename_path': wofs_fnpath}
                    df_files.append(entry)

        df_files = pd.DataFrame(df_files)
        df_files.sort_values(['run_date', 'init_time', 'ensemble_member', 
                              'forecast_time', 'filename_path'], inplace=True)
        csvfname = os.path.join(args.loc_date, 'wofs_preds_files.csv')
        df_files.to_csv(csvfname) #, index=False)
        print(f"Saving WoFS files list {csvfname}")
    elif args.loc_files_list:
        df_files = pd.read_csv(args.loc_files_list)
        df_files.sort_values(['run_date', 'init_time', 'ensemble_member',
                              'forecast_time', 'filename_path'], inplace=True)
        print("Loaded", args.loc_files_list)
    else: 
        raise ValueError("[ARGUMENT ERROR] either args.loc_date or args.loc_files_list should not be None")


    #init_times = df_files['init_time']
    #forecast_times = df_files['forecast_time'].values #.to_numpy() #


    # dict of summary function options
    #funcs = {'mean': np.mean, 'median': np.median, 
    #         'min': np.min, 'max': np.max}
    
    # Iterate over the initialization times
    for itime in args.inits:
        init_files = df_files.loc[df_files['init_time'] == int(itime)]
        forecast_times = init_files['forecast_time']
        print(f" [{itime}] \n{forecast_times}")
        dirpath = os.path.join(args.out_dir, itime, 'summary')
        
        # All forecast times
        wofs_cZH_all = {'mean': [], 'median': [], 'min': [], 'max': []}
        wofs_prob_all = {'mean': [], 'median': [], 'min': [], 'max': []}
        wofs_uh_all = {'mean': [], 'median': [], 'min': [], 'max': []}
        wofs_uh_ens_all = [] #filenamepath
        
        # Iterate over the forecast times for all ensembles
        for i, ftime in enumerate(forecast_times): #ftime = '2023-06-02_19_30_00'
            print(f"\n*** [{i}] init={itime}  forecast={ftime}")
            wofs_files = select_files(df_files, itime, ftime, emember=None)
            print("WOFS FILES\n", wofs_files[['ensemble_member', 'filename_path']])

            # Load WoFS preds files
            #coords = 'different'
            #print("coords", coords)
            concat_dim = 'ENS' #'Time'
            _files = list(wofs_files['filename_path'])
            wofs_preds = xr.open_mfdataset(_files, concat_dim=concat_dim,
                                           combine='nested', decode_times=False)
                                           #coords=['XTIME', 'XLONG', 'Time', 'XLAT'],
            #{'XTIME', 'XLONG', 'XLONG_U', 'Time', 'XLAT_V', 'XLONG_V', 'XLAT', 'XLAT_U'}
            wofs_preds.drop_vars('Time')
            print("wofs_preds")
            print(wofs_preds)
            print("- - - - - ")
            print(wofs_preds['UP_HELI_MAX'])

            wofs_cZH = {}
            wofs_prob = {}
            wofs_uh = {}
            #wofs_uh_ens = []

            wofs_cZH['mean'] = wofs_preds['COMPOSITE_REFL_10CM'].mean(dim=concat_dim).values
            wofs_prob['mean'] = wofs_preds['ML_PREDICTED_TOR'].mean(dim=concat_dim).values
            wofs_cZH_all['mean'].append(wofs_cZH['mean'][0])
            wofs_prob_all['mean'].append(wofs_prob['mean'][0])

            wofs_cZH['median'] = wofs_preds['COMPOSITE_REFL_10CM'].median(dim=concat_dim).values
            wofs_prob['median'] = wofs_preds['ML_PREDICTED_TOR'].median(dim=concat_dim).values
            wofs_cZH_all['median'].append(wofs_cZH['median'][0])
            wofs_prob_all['median'].append(wofs_prob['median'][0])

            if args.uh:
                wofs_uh['mean'] = wofs_preds['UP_HELI_MAX'].mean(dim=concat_dim).values
                wofs_uh_all['mean'].append(wofs_uh['mean'][0])
                wofs_uh['median'] = wofs_preds['UP_HELI_MAX'].median(dim=concat_dim).values
                wofs_uh_all['median'].append(wofs_uh['median'][0])
                #
                wofs_uh_ens_all.append(wofs_preds['UP_HELI_MAX'].isel(Time=0).values)

            if args.min_max:
                wofs_cZH['max'] = wofs_preds['COMPOSITE_REFL_10CM'].max(dim=concat_dim).values
                wofs_prob['max'] = wofs_preds['ML_PREDICTED_TOR'].max(dim=concat_dim).values
                wofs_uh['max'] = wofs_preds['UP_HELI_MAX'].max(dim=concat_dim).values
                wofs_cZH_all['max'].append(wofs_cZH['max'][0])
                wofs_prob_all['max'].append(wofs_prob['max'][0])

                wofs_cZH['min'] = wofs_preds['COMPOSITE_REFL_10CM'].min(dim=concat_dim).values
                wofs_prob['min'] = wofs_preds['ML_PREDICTED_TOR'].min(dim=concat_dim).values
                wofs_cZH_all['min'].append(wofs_cZH['min'][0])
                wofs_prob_all['min'].append(wofs_prob['min'][0])

                if args.uh:
                    wofs_uh['max'] = wofs_preds['UP_HELI_MAX'].max(dim=concat_dim).values
                    wofs_uh['min'] = wofs_preds['UP_HELI_MAX'].min(dim=concat_dim).values
                    wofs_uh_all['max'].append(wofs_uh['max'][0])
                    wofs_uh_all['min'].append(wofs_uh['min'][0])


            '''
            for stat in ['mean', 'median', 'max', 'min']:
                #wofs_cZH[stat] = wofs_preds['COMPOSITE_REFL_10CM'].mean(dim=concat_dim)
                #wofs_prob[stat] = wofs_preds['ML_PREDICTED_TOR'].mean(dim=concat_dim)
                print(f"{stat} preds [0 .25 .5 1]qtl", np.nanquantile(wofs_prob[stat], 
                                                                    [0, .25, .5, 1]))
            
                # PLOTS
                fname = os.path.join(dirpath, f'{ftime}_hist_{stat}.png')
                print(fname)
                hist_args = {'bins': 'fd', 'density': False}
                _args = deepcopy(args)
                _args.write = 3
                plot_hist(_args, wofs_prob[stat], f'Ensemble {stat} Tor Probab Distributions', 
                        f'{stat} Tor Probab', ['density', 'cum. density'], fname, 
                        show_text=True, figsize=(8, 6), dpi=110, save=False, **hist_args)

                fname = os.path.join(dirpath, f'{ftime}_preds_{stat}.png')
                print(fname)
                plot_pcolormesh(args, wofs_prob[stat], fname, f'Ensemble {stat} Tor Probability',
                                cb_label=r'p_{tor}', fig_ax=None, data_contours=None, 
                                cmap="cividis", vmin=0, vmax=1, alpha=None, dpi=250, 
                                figsize=(10, 9), tight=False, close=True, save=True)
                
                fname = os.path.join(dirpath, f'{ftime}_cZH_preds_{stat}.png')
                print(fname)
                plot_pcolormesh(args, wofs_cZH[stat], fname, f'Ensemble {stat} Composite Reflectivity',
                                cb_label='dBZ', fig_ax=None, data_contours=wofs_prob[stat],
                                cmap="Spectral_r", vmin=0, vmax=60, alpha=None, dpi=250, 
                                figsize=(10, 9), tight=False, close=True, save=True)
                '''

        nframes = len(wofs_cZH_all['mean'])
        anim_args = {'repeat': True, 'repeat_delay': 100}

        for stat in ['mean', 'median']:
            dat = wofs_cZH_all[stat] #np.concatenate(wofs_cZH_all[stat], axis=0)
            dat_prob = wofs_prob_all[stat]

            suffix = ''
            uh_arg = None
            if args.uh is not None: 
                data_uh = wofs_uh_all[stat]
                uh_arg = (data_uh, wofs_uh_ens_all)
                anim_args['uh_thres'] = args.uh_thres
                suffix = f'_uh_thres={args.uh_thres}'

            fnpath = os.path.join(dirpath, f'{itime}_cZH_preds_{stat}{suffix}.gif')
            print(f"{dat[0].shape} {dat_prob[0].shape} nframe={nframes}", fnpath)

            #wofs_uh_ens_all = [] #filenamepath
            create_gif(dat, dat_prob, fnpath, data_uh=uh_arg, 
                       stat=stat, frame_labels=forecast_times.values,  
                       nframes=nframes, interval=800, figsize=(20, 15), dpi=180, 
                       DB=True, write=args.write, **anim_args)
    
    t1 = time.time()
    print(f"DONE. Elapsed {(t1 - t0) / 60} min")
