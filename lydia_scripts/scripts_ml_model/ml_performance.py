import xarray as xr
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import tqdm
import sys
sys.path.append("/home/lydiaks2/tornado_project/WAF_ML_Tutorial_Part1/scripts/")
import gewitter_functions
from gewitter_functions import get_pod,get_sr,csi_from_sr_and_pod
import pandas as pd
import netCDF4

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.calibration import calibration_curve


def get_arguments():
    #Define the strings that explain what each input variable means
    PATCHES_DIR_HELP_STRING = 'The directory where the unet mask files are stored. This directory should start from the root directory \'/\'.'
    
    #Tell python what arguments to expect from the .sh file
    INPUT_ARG_PARSER = argparse.ArgumentParser()
        
    INPUT_ARG_PARSER.add_argument(
        '--path_to_predictions', type=str, required=True,
        help=PATCHES_DIR_HELP_STRING)

    #Pull out all of the input strings from the .sh file
    args = INPUT_ARG_PARSER.parse_args()
    #path to patches is the location all the patch files are stored
    global path_to_predictions
    path_to_predictions = getattr(args, 'path_to_predictions')
        


def plot_roc_curve(predictions):

    # Pull out the labels and the predictions
    true_tor = np.where(predictions.truth_labels.values > 0, 1, 0)
    predicted_tor = predictions.predicted_tor.values
    
    # Reformat the predictions and labels to a 1D array
    t_probs = predicted_tor.ravel()
    t_true = true_tor.ravel()
    
    # Calculate the AUC
    t_auc = roc_auc_score(t_true, t_probs)
    
    # Get the probability of detection and probability of false detection from the predictions and labels
    pofds, pods = gewitter_functions.get_points_in_roc_curve(t_probs,t_true,threshold_arg=np.linspace(0,1,10))
    
    # Plot and save the ROC Curve
    plt.figure()
    plt.plot([0,1], linestyle='--')
    plt.plot(pofds, pods, linestyle='-', color='r', label='Testing: AUC = %.2f' % t_auc)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(path_to_predictions + '/roc_curve.png',dpi=500)
    
def plot_reliability_diagram(predictions):
    
    # Pull out the labels and the predictions
    true_tor = np.where(predictions.truth_labels.values > 0, 1, 0)
    predicted_tor = predictions.predicted_tor.values
    
    # Reformat the predictions and labels to a 1D array
    t_probs = predicted_tor.ravel()
    t_true = true_tor.ravel()
    
    # Bin the data and calculate the observed frequency and the predicted probabilities of the data
    prob_true_t, prob_pred_t = calibration_curve(t_true, t_probs, n_bins=10)
    
    # Plot the reliability diagram
    plt.figure()
    plt.plot([0,1], linestyle='--')
    plt.plot(prob_pred_t,prob_true_t, linestyle='-', color='r', label='Testing')
    plt.ylabel("Observed Frequency")
    plt.xlabel("Predicted Probability")
    plt.grid()
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path_to_predictions + '/reliability_diagram.png',dpi=500)
    
    
def plot_performance_diagram(predictions):
    
    # Pull out the labels and the predictions
    true_tor = np.where(predictions.truth_labels.values > 0, 1, 0)
    predicted_tor = predictions.predicted_tor.values
    
    # Reformat the predictions and labels to a 1D array
    y_preds = predicted_tor.ravel()
    t_true = true_tor.ravel()
        
    #Define the thresholds, this will make 20 evenly spaced points between 0 and 1. 
    threshs = np.linspace(0,1,11)

    #pre-allocate a vector full of 0s to fill with our results 
    pods = np.zeros(len(threshs))
    srs = np.zeros(len(threshs))
    csis = np.zeros(len(threshs))

    #Loop through each threshold
    for i,t in enumerate(tqdm.tqdm(threshs)):
        #make a dummy binary array full of 0s
        y_preds_bi = np.zeros(y_preds.shape,dtype=int)
        
        #find where the prediction is greater than or equal to the threshold
        idx = np.where(y_preds >= t)\

        #set those indices to 1
        y_preds_bi[idx] = 1

        #get the contingency table again 
        table = gewitter_functions.get_contingency_table(y_preds_bi,t_true)

        #calculate pod, sr and csi 
        pods[i] = get_pod(table)
        srs[i] = get_sr(table)
        csis[i] = csi_from_sr_and_pod(srs[i],pods[i])

    # Calculate the max CSI value of the thresholds we calculated
    maxcsi = np.nanmax(csis)
        
    # Plot the performance diagram
    plt.figure()
    ax = gewitter_functions.make_performance_diagram_axis()
    ax.plot(srs,pods,'-',color='r',markerfacecolor='w',lw=2, label='Testing: Max CSI = %.2f' % maxcsi)
    ax.legend(loc='upper right')
    for i,t in enumerate(threshs):
        text = np.char.ljust(str(np.round(t,2)),width=4,fillchar='0')
        ax.text(srs[i]+0.02,pods[i]+0.02,text,fontsize=9,color='white') #path_effects=pe1
    plt.savefig(path_to_predictions + '/performance_diagram.png',dpi=500)



def make_patch_images(predictions, write=False):

    # Make the directory for the images if it doesn't already exist
    if not os.path.exists(path_to_predictions + '/patch_images/'):
        os.mkdir(path_to_predictions + '/patch_images/')

    #isolate only tornadic patches
    tor_pred = predictions.where(predictions.truth_labels.max(axis=(1,2)) > 0, drop=True)

    # Loop through all the images that have a tornado in them
    for j in range(tor_pred.patch.values.shape[0]):

        # Plot composite reflectivity
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        plot0 = ax[0].pcolormesh(tor_pred.ZH_composite.values[j], cmap="Spectral_r", vmin=0, vmax=50)
        ax[0].contour(tor_pred.truth_labels.values[j])
        ax[0].set_title('Composite Reflectivity')
        cbar0 = fig.colorbar(plot0, ax=ax[0])
        cbar0.set_label('dBZ', rotation=0)

        # # Pull out the vorticity field
        # vor_level = tor_pred.VOR_max.values[j]

        # # Plot column max vorticity
        # plot1 = ax[1].pcolormesh(vor_level, cmap="seismic", vmin = min(vor_level.min(), -vor_level.max()), vmax = -min(vor_level.min(), -vor_level.max()))
        # ax[1].contour(tor_pred.truth_labels.values[j])
        # ax[1].set_title('Column Max Vorticity')
        # cbar1 = fig.colorbar(plot1, ax=ax[1])
        # cbar1.set_label('m$^2$s$^{-1}$', rotation=0)
        
        # Plot the tornado predictions
        plot1 = ax[1].pcolormesh(tor_pred.predicted_tor.values[j], cmap='cividis', vmin=0, vmax=1)
        ax[1].contour(tor_pred.truth_labels.values[j])
        ax[1].set_title('Tornado Predictions')
        cbar1 = fig.colorbar(plot1, ax=ax[1])
        cbar1.set_label('p$_{tor}$', rotation=0)

        # Save out the figure
        if write:
            fpath = path_to_predictions + '/patch_images/patch_' + str(j) + '.png'
            print("Saving", fpath)
            plt.savefig(fpath, dpi=500)


def main():

    # Get the inputs from the .sh file
    get_arguments()  

    # Load in the ML predictions (already been made)
    print("Loading Predictions")
    predictions = xr.open_dataset(path_to_predictions + '/predictions/test_predictions.nc')

    # Make the ROC Curve
    print('Making ROC Curve')
    plot_roc_curve(predictions)

    # Make the Reliability Diagram
    print('Making Reliability Diagram')
    plot_reliability_diagram(predictions)

    # Make the Performance Diagram
    print('Making Performance Diagram')
    plot_performance_diagram(predictions)

    # Save out individual patch images
    print('Making Images')
    make_patch_images(predictions)



if __name__ == "__main__":
    main()


