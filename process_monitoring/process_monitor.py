"""
Process monitor to measure memory and time utilization of a process.
"""
import math
import os, sys, psutil #, platform, memory_profiler
from multiprocessing import Pipe, Process
from subprocess import Popen, PIPE
import time, datetime 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


class ProcessMonitor(psutil.Process):
    """
    See psutil.Process for argument information.
    Augemented Process class that organizes attributes of interest for the
    process into a pandas DataFrame. 
    TODO: include gpu monitoring and recording of multiple time intervals
    by a parallel process
    """
    def __init__(self, pid=None):
        super().__init__(pid)
        # Name of program file without .py extension
        self.program_name = self.cmdline()[1][:-3]
        # Time elapsed since the program started, includes time elapsed during sleep
        self.start_perf_counter = 0
        # CPU time consumed by the current process (system + user CPU time )
        # Does NOT include time elapsed during sleep
        self.start_process_time = 0
        print("Process Monitor Initialized. Call start_timer() to track elapsed time.")

    def start_timer(self, attrs=None):
        '''
        Start tracking the execution of the process of a point of interest
        '''
        # Time elapsed since program started, includes time elapsed during sleep
        self.start_perf_counter = time.perf_counter()
        # CPU time of the current process (system + user CPU time ), excludes sleep time
        self.start_process_time = time.process_time()
        self.monitored_attrs = attrs
        self.init_performance_monitor()
        #TODO gpu

    def init_performance_monitor(self):
        #TODO make dictionary construction dynamic
        # Track process attributes throughout execution
        info = self.as_dict(attrs=self.monitored_attrs)
        #psutil.cpu_count()
        #['cpu_percent', 'memory_percent', 'cpu_affinity', 'io_counters', 'num_threads', 'threads', 'cpu_times', 'memory_info', 'memory_full_info', 'open_files']
        '''self.performance_monitor = {'time': [self.start_perf_counter], 'cpu_percent': [info['cpu_percent']], 
                                    'memory_percent': [info['memory_percent'].vms], 'num_threads': [info['num_threads']], 
                                    'cpu_times.user': [info['cpu_times'].user], 'cpu_times.system': [info['cpu_times'].system], 
                                    'cpu_times.idle': [info['cpu_times'].idle], 'memory_info.vms': [info['memory_full_info'].vms], 
                                    'memory_info.rss': [info['memory_full_info'].rss]}'''
        self.performance_monitor = {'time': [], 'cpu_percent': [], 'memory_percent.rss': [], 'memory_percent.vms': [], 
                                    'num_threads': [], 'cpu_times.user': [], 'cpu_times.system': [], 
                                    'memory_info.vms': [], 'memory_info.rss': [] #(LINUX)'cpu_times.idle': [], 
                                    }
    
    def update_performance_monitor(self):
        info = self.as_dict(attrs=self.monitored_attrs)
        #psutil.cpu_count()
        #self.performance_monitor = {'time': [self.start_perf_counter]}.append({key: ([val] if not isinstance(val, list) else val) for key, val in info.items()})
        self.performance_monitor['time'].append(time.perf_counter() - self.start_perf_counter)
        self.performance_monitor['cpu_percent'].append(info['cpu_percent'])
        self.performance_monitor['memory_percent.rss'].append(info['memory_percent'])
        self.performance_monitor['memory_percent.vms'].append(self.memory_percent('vms'))
        self.performance_monitor['num_threads'].append(info['num_threads'])
        self.performance_monitor['cpu_times.user'].append(info['cpu_times'].user)
        self.performance_monitor['cpu_times.system'].append(info['cpu_times'].system)
        #if LINUX: self.performance_monitor['cpu_times.idle'].append(info['cpu_times'].idle)
        self.performance_monitor['memory_info.vms'].append(info['memory_info'].vms)
        self.performance_monitor['memory_info.rss'].append(info['memory_info'].rss)
        #TODO gpu

    def write_performance_monitor(self, output_path=None):
        '''
        Save the performance information into a csv file
        @param output_path: override default save location, {program_name}_performance_monitor.csv
        '''
        nitems = len(self.performance_monitor)

        if output_path is None:
            # Name of program file without .py extension
            output_path = self.program_name + '_performance_monitor.csv'

        df = pd.DataFrame(self.performance_monitor)
        df.to_csv(output_path, index=False)
        print(f"Performance monitor data written to {output_path}")

    def plot_performance_monitor(self, attrs=None, ax=None, output_path=None, write=False, format='png'):
        '''
        Plot performance monitoring information over time
        @params attrs: list of attrs to plot. all atributes are plot if is None.
        @param ax: ax for the plot
        @param output_path: override default save location, {program_name}_performance_monitor_plot.png
        @param write: flag specifying whether to save the plot
        @param format: string indicating the image format for the plot file (eg 'png', 'pdf', etc). see pandas.DataFrame.plot for more details
        '''
        if attrs is None: attrs = list(self.performance_monitor.keys())

        ntimes = len(self.performance_monitor['time'])
        df = pd.DataFrame(data=self.performance_monitor, index=range(ntimes))

        # Columns to plot
        ys = [a for a in attrs if not 'time' in a]
        ncols = df.shape[1]
        nrows = int(math.ceil(float(ncols) / 2))
        axs = df.plot(kind='line', x='time', y=ys, sharex=True,
                        subplots=True, grid=True, figsize=(10,12)) #xlabel='Time (s)', 
        plt.xlabel('Time (s)')

        if output_path is None:
            # Name of program file without .py extension
            output_path = f"{self.program_name}_performance_monitor_plot.{format}"

        if write:
            print(f"Saving plots to {output_path}")
            plt.savefig(output_path, format=format, dpi=220)
        
    def end_timer(self):
        # Time since the program started, includes time elapsed during sleep
        self.end_perf_counter = time.perf_counter()
        self.elapsed_perf_counter = self.end_perf_counter - self.start_perf_counter
        # CPU time consumed by the current process (system + user CPU time )
        # Does NOT include time elapsed during sleep
        self.end_process_time = time.process_time()
        self.elapsed_process_time = self.end_process_time - self.start_process_time

    def expand_attr(self, attr):
        '''
        Expand complex attribute with multiple sub-attributes into multiple attributes
        '''
        def namedtuple_to_dict(attr, details):
            return {'{0}.{1}'.format(attr, k): [v] for k, v in details._asdict().items()}
        
        details = getattr(self, attr)()
        new_entries = {attr: [details]} 
        if isinstance(details, list) and len(details) > 0: 
            #if hasattr(details[0], '_asdict'): new_entries = namedtuple_to_dict(attr, details)
            new_entries[attr] = [str(details).replace(',', ' ')]
        elif hasattr(attr, '_asdict'):
            # separate the entries of the named tuples into individual dict items
            new_entries = namedtuple_to_dict(attr, details)
        return new_entries

    def print(self, attrs=None, write=False, output_path=None):
        '''
        Display process information in human readable form. All the attributes are 
        displayed if attrs is None, otherwise the specified the list of attributes 
        in attrs is displayed. 
        @param attrs: list of psutil.Process attributes to track
        @param write: flag whether to write process information into .csv file 
                        with the same name as the program file
        @param output_path: override default save location, {program_name}.csv
        '''
        # Helper function
        def format_val(attr, val):
            '''
            Format output strings for cpu_times and memory_info*
            based on size and units
            '''
            units = ''
            #if attr == 'cpu_times':
            if 'time' in attr or 'elapsed' in attr:
                units = 'sec'
                # More than 1 hours (60 min, 3600 sec)
                if (val / 3600) >= 1:
                    val /= 3600
                    units = 'hrs'
                # More than 3 minutes (180 sec)
                elif (val / 60) >= 3:
                    val /= 60
                    units = 'min'
            elif '_info' in attr:
                units = 'byte'
                # More than 1TB
                if val > 1e12:
                    val /= 1e12
                    units = 'TB'
                elif val > 1e9:
                    val /= 1e9
                    units = 'GB'
                elif val > 1e6:
                    val /= 1e6
                    units = 'MB'
                # More than 5K
                elif val > 5e3:
                    val /= 1e3
                    units = 'KB'
            return '{0} {1}'.format(val, units)

        #TODO override .as_dict
        info = self.as_dict(attrs=attrs)
        if attrs is None or 'elapsed_perf_counter' in attrs:
            info['elapsed_perf_counter'] = self.elapsed_perf_counter
        if attrs is None or 'elapsed_process_time' in attrs:
            info['elapsed_process_time'] = self.elapsed_process_time

        # organize the presentation of the process information
        info_up = dict(info)
        for key in info.keys():
            details = info[key]
            
            if key in ['cpu_affinity', 'connections']: 
                info_up[key] = str(details).replace(',', '')
            elif key in ['elapsed_perf_counter', 'elapsed_process_time']:
                info_up[key] = format_val(key, details)
            elif isinstance(details, list) and len(details) > 0:
                dets = [str(i) for i in details]
                sep = ' ' if key == 'cmdline' else '. '
                info_up[key] = sep.join(dets)
            elif key == 'create_time':
                info_up[key] = datetime.datetime.fromtimestamp(details).strftime("%A %B %d %Y %H:%M:%S %Z")
            elif key in ['io_counters', 'threads', 'cpu_times', 'memory_info', 'memory_full_info', 
                        'memory_maps', 'open_files', 'num_ctx_switches']:
                # separate the entries of the named tuples into individual dict items
                new_entries = {'{0}.{1}'.format(key, k): format_val(key, v) for k, v in details._asdict().items()}
                info_up.pop(key)
                info_up.update(new_entries)
            elif 'percent' in key:
                info_up[key] = f"{details:.2f} %"
            else: info_up[key] = str(details) #.replace(',', ' ')
        #TODO gpu

        df = pd.DataFrame(data=info_up, index=['Information']).T
        if write:
            if output_path is None:
                output_path = self.program_name + '.csv'
            print(f"Writing {output_path}")
            df.to_csv(output_path, header=False)

        # Display entire dataframe
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        print(df)

        return df


# Run if this file is executed directly
if __name__ == "__main__":
    proc = ProcessMonitor()

    # Busy cycles
    out = '\n\n'
    proc.start_timer()
    st = time.time()
    t = st #time.time()
    while t - st <= 10:
        proc.update_performance_monitor()
        t = time.time()
        out += f'{t} ' 
    #print(out)

    proc.end_timer()
    fn_prefix = 'test_process_monitor'
    proc.write_performance_monitor(output_path=None)
    proc.plot_performance_monitor(attrs=None, ax=None, write=True, output_path=None)
    proc.print(write=True, output_path=None)
