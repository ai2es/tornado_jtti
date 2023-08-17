''' The aim of this python script is to run the analysis_functions and generate the CSI plots 
The main goal/purpose of this file is to gather/sift through the files in the correct manner to gain the best results'''

import tqdm
import pandas as pd
import sys, os

sys.path.append(os.path.dirname(os.getcwd()))

sys.path.append(os.path.abspath('/ourdisk/hpc/ai2es/alexnozka/'))
from evaluation_functions import *
from analysis_functions import *

# Setting the target root path
date_string = '20190430'
target_root_path = '/ourdisk/hpc/ai2es/alexnozka/results/neighborhooding/' + date_string + '/'
destination_root_path = target_root_path


final_df = pd.DataFrame()
# Iterate throught the WoFS initialization times
for files in tqdm.tqdm(os.listdir(target_root_path), desc='init_time'):
    #print(file)
    
    # Evaluates the functions within the specified timesteps
    if files in ['2130','2200','2230','2300','2330']:
        continue
    
    #print(files)
    print(files)
    total_df = pd.DataFrame()
    temp_df = pd.DataFrame()
    target_path = target_root_path + files + '/'
    destination_path = target_path
    # Iterate throught the files within each intialization time dir
    try:
        for file in tqdm.tqdm(os.listdir(target_path), desc='timestep'):
            #print(file)
            # All the excel spreadsheets have this in their name
            if 'properties' in file and files not in file:
                #print("File approved: " + file)
                #print(target_path)
                sheetNames = pd.ExcelFile(os.path.join(target_path, file)).sheet_names
                for sheet in tqdm.tqdm(sheetNames):
                    if len(total_df) == 0:
                        total_df = pd.read_excel(os.path.join(target_path, file),sheet)
                    else:
                        temp_df = pd.read_excel(os.path.join(target_path, file),sheet)
                        total_df =+ pd.concat([total_df, temp_df])
        predictions = np.array(total_df['ML_Max_Prob'])
        tor_reports = np.array(total_df['ML_Stat'])
        UH_swaths = np.array(total_df['MaxUH'])
        UH_thresh = np.array([np.min(UH_swaths),np.max(UH_swaths)])
        #UH_thresh = np.array([20,120])

        plot_performance_diagram(predictions, tor_reports, UH_swaths, UH_thresh, destination_path)
        #iterate_performance_diagram(predictions, tor_reports, UH_swaths, UH_thresh, destination_path)
        #plotly_reliability_diagram(predictions, tor_reports, UH_swaths, destination_path)
        plot_reliability_diagram(predictions, tor_reports, UH_swaths, destination_path)
        # Concatenating the dataframes together
        if len(final_df) == 0:
            final_df = total_df
        else:
            total_df =+ pd.concat([final_df, total_df])
        print(final_df)
    # If not selecting a directory, skip
    except(NotADirectoryError):
        print(target_path)
        pass
    

predictions = np.array(final_df['ML_Max_Prob'])
tor_reports = np.array(final_df['ML_Stat'])
UH_swaths = np.array(final_df['MaxUH'])
#UH_thresh = np.array([20,120])
UH_thresh = np.array([np.min(UH_swaths),np.max(UH_swaths)])
print(final_df)
final_df.to_csv(target_root_path + date_string + '_compiled_storms.csv')
plot_performance_diagram(predictions, tor_reports, UH_swaths, UH_thresh, destination_root_path)
#iterate_performance_diagram(predictions, tor_reports, UH_swaths, UH_thresh, destination_root_path)
plot_reliability_diagram(predictions, tor_reports, UH_swaths, destination_path)