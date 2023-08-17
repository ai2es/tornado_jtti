''' This program is responsible for sifting through the different folders and running the evaluation_functions.py evaluation '''

# IMPORTS 
import tqdm
import pandas as pd
import sys, os

sys.path.append(os.path.dirname(os.getcwd()))

sys.path.append(os.path.abspath("/ourdisk/hpc/ai2es/alexnozka/tools/MontePython/"))
from monte_python import *
import monte_python

sys.path.append(os.path.abspath('/ourdisk/hpc/ai2es/alexnozka/'))
from evaluation_functions import *
from analysis_functions import *

# This is where the magic happens

# This is the day desired, formatted yyyymmdd/
target_day = '20230602' + '/'

#wofs_target_root_path = '/ourdisk/hpc/ai2es/wofs/2023/' + target_day
wofs_target_root_path = '/ourdisk/hpc/ai2es/tornado/wofs-preds-2023/' + target_day 
# This root path is subject to change as we run more models, should keep similar format
ml_target_root_path = '/ourdisk/hpc/ai2es/momoshog/Tornado/tornado_jtti/wofs_preds/2023/' + target_day
destination_root_path = '/ourdisk/hpc/ai2es/alexnozka/results/moniquemodels/model1/2023/' + target_day

# Looping through the different model initialization times
for times in tqdm.tqdm(os.listdir(wofs_target_root_path), desc='init_time'):
    # The goal of this is to prevent going into day-before data and limit the amount of times we re-run data
    if int(times) > 1800: #and times in [] # This helps not re-run times that have previously been run
        
        # This is to prevent going into initialization times that aren't present in the ML data
        if times not in ['1900','1930','2000','2030','2100']:
            continue
        
        # This goes through each ensemble member
        for ensemble in range(1,19):
            # This is to prevent re-runs of ensembles within a given timestep
            '''
            if ensemble in [1,2] and times == '2030':
                continue
            '''
            wofs_target_path = wofs_target_root_path + times + '/ENS_MEM_' + str(ensemble) + '/'
            ml_target_path = ml_target_root_path + times + '/ENS_MEM_' + str(ensemble) + '/'
            destination_path = destination_root_path + times + '/'
            build_tables(wofs_target_path,ml_target_path,destination_path=destination_path,plot=True)