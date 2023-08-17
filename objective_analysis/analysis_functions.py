import datetime
import pandas as pd
import tqdm
from sklearn.calibration import calibration_curve

import sys, os
sys.path.append(os.path.abspath('/ourdisk/hpc/ai2es/alexnozka/'))
from evaluation_functions import *
sys.path.append(os.path.abspath('/ourdisk/hpc/ai2es/alexnozka/tools/'))
from gewitter_functions import *

# Generating the performance diagram for WoFS vs ML
def plot_performance_diagram(predictions,tors,UH_swaths,UH_thresh,destination_path):
    
    #true_tor = np.where(predictions > 0, 1, 0)
    true_tor = np.where(tors > 1, 1, 0)
    predicted_tor = predictions
    uh_preds = UH_swaths

    # Since I did not have to np.ravel my arrays, I just have to do this assignment
    t_probs = predicted_tor.ravel()
    t_true = true_tor.ravel()
    
    #lets just focus on the output from class 1 (note, the sum of these two columns should be 1)
    y_preds = t_probs
        
    #chose you thresholds, this will make 10 evenly spaced points between 0 and 1. 
    threshs = np.linspace(0,1,11)
    # Making 10 bins between the min and max UH values ### MIGHT BE SUBJECT TO CHANGE
    rangeUH = UH_thresh[0]+UH_thresh[1]
    nbins = 10
    width = rangeUH/nbins
    threshsUH = []
    for bins in range(0,nbins):
        threshsUH.append((width*bins) + UH_thresh[0])
    print(threshs)
    print(threshsUH)
    #it_threshsUH = threshsUH.values

    #pre-allocate a vector full of 0s to fill with our results 
    podsML,podsUH = np.zeros(len(threshs)),np.zeros(len(threshsUH))
    srsML, srsUH = np.zeros(len(threshs)),np.zeros(len(threshsUH))
    csisML, csisUH  = np.zeros(len(threshs)),np.zeros(len(threshsUH))

    plt.figure()
    #for each threshold!
    for i,t in enumerate(tqdm.tqdm(threshsUH,desc='WoFS_data')):
        #make a dummy binary array full of 0s
        uh_preds_bi = np.zeros(uh_preds.shape,dtype=int)
        
        #find where the prediction is greater than or equal to the threshold
        idx = np.where(uh_preds >= t)
        #set those indices to 1
        uh_preds_bi[idx] = 1
        #get the contingency table again 
        table = get_contingency_table(uh_preds_bi,t_true) #gewitter_functions.
        #calculate pod, sr and csi 
        podsUH[i] = get_pod(table)
        srsUH[i] = get_sr(table)
        csisUH[i] = csi_from_sr_and_pod(srsUH[i],podsUH[i])
        
    for i,t in enumerate(tqdm.tqdm(threshs,desc='ML_data')):
        #make a dummy binary array full of 0s
        y_preds_bi = np.zeros(y_preds.shape,dtype=int)
        
        #find where the prediction is greater than or equal to the threshold
        idx = np.where(y_preds >= t)
        #set those indices to 1
        y_preds_bi[idx] = 1
        #get the contingency table again 
        table = get_contingency_table(y_preds_bi,t_true) #gewitter_functions.
        #calculate pod, sr and csi 
        podsML[i] = get_pod(table)
        srsML[i] = get_sr(table)
        csisML[i] = csi_from_sr_and_pod(srsML[i],podsML[i])
        
    ax = make_performance_diagram_axis() #gewitter_functions.
    ax.plot(srsML,podsML,'-',color='dodgerblue',markerfacecolor='w',lw=2, label = 'ML Performance')
    ax.plot(srsUH,podsUH,'-',color='red',markerfacecolor='w',lw=2, label = 'WoFS Performance')
    ax.legend()

    path_parts = destination_path.split('/')

    day = ''
    time = ''
    for parts in path_parts:
        if len(parts) == 6:
            day = parts
            ax.set_title('WoFS vs ML Performance on ' + day)
        if len(parts) == 4:
            time = parts
            ax.set_title('WoFS vs ML Performance on ' + day + ' at ' + time)

    if len(time) > 0:
        plt.savefig(destination_path + day + '_' + time + '_' + 'performance_diagram.png')
    elif len(time) == 0 and len(day) > 0:
        plt.savefig(destination_path + day + '_' + 'performance_diagram.png')
    else:
        ax.set_title('WoFS vs ML Performance')
        plt.savefig(destination_path + 'performance_diagram.png')

    # Annotating the CSIs and thresholds
    '''for i,t in enumerate(threshs):
        text = np.char.ljust(str(np.round(t,2)),width=4,fillchar='0')
        ax.text(srsML[i]+0.02,podsML[i]+0.02,text,fontsize=9,color='white') #path_effects=pe1'''
    '''for i,t in enumerate(threshsUH):
        text = np.char.ljust(str(np.round(t,2)),width=4,fillchar='0')
        ax.text(srsUH[i]+0.02,podsUH[i]+0.02,text,fontsize=9,color='white') #path_effects=pe1'''

# This function operates similarly to the previous function, except makes different numbers of bins/evaluation criterion to compare the model/wofs performance    
def iterate_performance_diagram(predictions,tors,UH_swaths,UH_thresh,destination_path):
    ''' THIS SHOULD ALL BE THE SAME AS THE PREVIOUS FUNCTION, CALCULATING THE ML '''

    #true_tor = np.where(predictions > 0, 1, 0)
    true_tor = np.where(tors > 1, 1, 0)
    predicted_tor = predictions
    uh_preds = UH_swaths

    # Since I did not have to np.ravel my arrays, I just have to do this assignment
    t_probs = predicted_tor.ravel()
    t_true = true_tor.ravel()
    
    #lets just focus on the output from class 1 (note, the sum of these two columns should be 1)
    y_preds = t_probs
        
    #chose you thresholds, this will make 20 evenly spaced points between 0 and 1. 
    threshs = np.linspace(0,1,11)
    podsML = np.zeros(len(threshs))
    srsML = np.zeros(len(threshs))
    csisML = np.zeros(len(threshs))

    plt.figure()
    #for each threshold!
    for i,t in enumerate(tqdm.tqdm(threshs, desc = 'ML Data')):
        #make a dummy binary array full of 0s
        y_preds_bi = np.zeros(y_preds.shape,dtype=int)
        
        #find where the prediction is greater than or equal to the threshold
        idx = np.where(y_preds >= t)
        #set those indices to 1
        y_preds_bi[idx] = 1
        #get the contingency table again 
        table = get_contingency_table(y_preds_bi,t_true) #gewitter_functions.
        #calculate pod, sr and csi 
        podsML[i] = get_pod(table)
        srsML[i] = get_sr(table)
        csisML[i] = csi_from_sr_and_pod(srsML[i],podsML[i])

    ax = make_performance_diagram_axis() #gewitter_functions.
    ax.plot(srsML,podsML,'-',color='dodgerblue',markerfacecolor='w',lw=2, label = 'ML Performance')

    '''THIS IS WHERE WE MAKE THINGS A LITTLE DIFFERENT'''

    nbins = 10
    #threshmax = UH_thresh[1]
    analysis_lines = []
    nbins = [5,10,20]
    for bins in nbins:
        threshsUH = pd.cut(UH_thresh,bins, labels=False)
        podsUH = np.zeros(bins)
        srsUH = np.zeros(bins)
        csisUH = np.zeros(bins)

        for i,t in enumerate(tqdm.tqdm(threshsUH)):
            #make a dummy binary array full of 0s
            uh_preds_bi = np.zeros(y_preds.shape,dtype=int)
            
            #find where the prediction is greater than or equal to the threshold
            idx = np.where(uh_preds >= t)
            #set those indices to 1
            uh_preds_bi[idx] = 1
            #get the contingency table again 
            table = get_contingency_table(uh_preds_bi,t_true) #gewitter_functions.
            #calculate pod, sr and csi 
            podsUH[i] = get_pod(table)
            srsUH[i] = get_sr(table)
            csisUH[i] = csi_from_sr_and_pod(srsUH[i],podsUH[i])

        analysis_lines.append([podsUH,srsUH,csisUH])
    
    ax.plot(analysis_lines[0][1],analysis_lines[0][0],'-',color='y',markerfacecolor='w',lw=2, label = 'UH Performance 5 bins')
    ax.plot(analysis_lines[1][1],analysis_lines[1][0],'-',color='r',markerfacecolor='w',lw=2, label = 'UH Performance 10 bins')
    ax.plot(analysis_lines[2][1],analysis_lines[2][0],'-',color='m',markerfacecolor='w',lw=2, label = 'UH Performance 20 bins')
    ax.legend()

    path_parts = destination_path.split('/')

    day = ''
    time = ''
    for parts in path_parts:
        if len(parts) == 6:
            day = parts
            ax.set_title('WoFS vs ML Performance on ' + day)
        if len(parts) == 4:
            time = parts
            ax.set_title('WoFS vs ML Performance on ' + day + ' at ' + time)

    if len(time) > 0:
        plt.savefig(destination_path + day + '_' + time + '_' + 'iter_performance_diagram.png')
    elif len(time) == 0 and len(day) > 0:
        plt.savefig(destination_path + day + '_' + 'iter_performance_diagram.png')
    else:
        ax.set_title('WoFS vs ML Performance')
        plt.savefig(destination_path + 'iter_performance_diagram.png')


# This function and imports are for the Reliability diagram generated through Plotly
# TODO: Get Lydia's graph working with the multi-axis
# This converts the UH to probabilities (Might not be needed later)
def uh_to_prob(uh):
    uh_max = np.max(uh)
    uh_probs = np.zeros(len(uh))
    for i in range(0,len(uh)):
        uh_probs[i] = uh[i]/uh_max
    return uh_probs
'''def uh_to_prob(uh,max_uh):
    #max_uh = np.max(uh)
    uh_probs = np.zeros(len(uh))
    for i in range(0,len(uh)):
        uh_probs[i] = uh[i]/max_uh
    return uh_probs
def prob_to_uh(prob,max_uh):
    #uh_max = np.max(uh)
    uhs = np.zeros(len(prob))
    for i in range(0,len(prob)):
        uhs[i] = uh[i]*max_uh
    return uhs'''
### This function keeps UH as a "percentage"
def plot_reliability_diagram(predictions,tors, UH_swaths, destination_path):
    true_tor = np.where(tors > 1, 1, 0)
    predicted_tor_ML = predictions
    #predicted_tor_UH = UH_swaths
    
    t_probs = predicted_tor_ML.ravel()
    #uh_probs = UH_swaths.ravel()
    t_true = true_tor.ravel()
    
    prob_true_t, prob_pred_t = calibration_curve(t_true, t_probs, n_bins=10)
    
    #plt.figure()
    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot([0,1], linestyle='--')
    ax.plot(prob_pred_t,prob_true_t, linestyle='-', color = 'dodgerblue',label='ML')

    max_UH = np.max(UH_swaths)
    uh_probs = uh_to_prob(UH_swaths) # Old method
    #uh_probs = uh_to_prob(UH_swaths, max_UH) # New method 
    uh_probs = uh_probs.ravel()
    secax = ax.secondary_xaxis('top') # Old method
    #secax = ax.secondary_xaxis('top', functions = (uh_to_prob(max_uh = max_UH), prob_to_uh(max_uh = max_UH))) # New method
    secax.set_xlabel('Updraft Helicity')
    #secax.set_xticks(np.arrange(0,6),[0,0.2*max_UH,0.4*max_UH,0.6*max_UH,0.8*max_UH,max_UH])
    prob_true_t, prob_pred_t = calibration_curve(t_true, uh_probs, n_bins=10)
    ax.plot(prob_pred_t,prob_true_t, linestyle='-', color = 'r', label='WoFS')

    plt.ylabel("Observed Frequency")
    plt.xlabel("Predicted Probability")
    plt.grid()
    plt.title("Reliability Diagram")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(destination_path + 'Reliability_Diagram_final.png')

def plot_reliability_diagram_ml(predictions,tors, destination_path):
    true_tor = np.where(tors > 1, 1, 0)
    predicted_tor_ML = predictions
    #predicted_tor_UH = UH_swaths
    
    t_probs = predicted_tor_ML.ravel()
    #uh_probs = UH_swaths.ravel()
    t_true = true_tor.ravel()
    
    prob_true_t, prob_pred_t = calibration_curve(t_true, t_probs, n_bins=10)
    
    #plt.figure()
    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot([0,1], linestyle='--')
    ax.plot(prob_pred_t,prob_true_t, linestyle='-', color = 'dodgerblue',label='ML')

    plt.ylabel("Observed Frequency")
    plt.xlabel("Predicted Probability")
    plt.grid()
    plt.title("Reliability Diagram")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(destination_path + 'Reliability_Diagram_ml.png')

