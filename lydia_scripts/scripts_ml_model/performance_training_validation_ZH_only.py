import numpy as np
import argparse
import tensorflow as tf 
from tensorflow import keras
import matplotlib.pyplot as plt
import tqdm
import sys
# import sklearn
sys.path.append("/home/lydiaks2/tornado_project/WAF_ML_Tutorial_Part1/scripts/")
import gewitter_functions
from gewitter_functions import get_pod,get_sr,csi_from_sr_and_pod
sys.path.append("/home/lydiaks2/tornado_project")
from custom_metrics import MaxCriticalSuccessIndex
from custom_losses import make_fractions_skill_score

from sklearn.calibration import calibration_curve


def get_arguments():
    #Define the strings that explain what each input variable means
    TRAINING_DIR_HELP_STRING = 'The directory where the training dataset is stored. This directory should start from the root directory \'/\'.'
    VALIDATION_DIR_HELP_STRING = 'The directory where the validation dataset is stored. This directory should start from the root directory \'/\'.'
    PREDICTIONS_DIR_HELP_STRING = 'The directory where the output images will be stored. This directory should start from the root directory \'/\'.'
    CHECKPOINT_DIR_HELP_STRING = 'The directory where the ML model is stored. This directory should start from the root directory \'/\'.'


    #Tell python what arguments to expect from the .sh file
    INPUT_ARG_PARSER = argparse.ArgumentParser()
        
    INPUT_ARG_PARSER.add_argument(
        '--path_to_training_data', type=str, required=True,
        help=TRAINING_DIR_HELP_STRING)
        
    INPUT_ARG_PARSER.add_argument(
        '--path_to_validation_data', type=str, required=True,
        help=VALIDATION_DIR_HELP_STRING)
        
    INPUT_ARG_PARSER.add_argument(
        '--path_to_predictions', type=str, required=True,
        help=PREDICTIONS_DIR_HELP_STRING)
        
    INPUT_ARG_PARSER.add_argument(
        '--model_checkpoint_path', type=str, required=True,
        help=CHECKPOINT_DIR_HELP_STRING)

    #Pull out all of the input strings from the .sh file
    args = INPUT_ARG_PARSER.parse_args()
    #path to patches is the location all the training dataset is stored
    global path_to_training_data
    path_to_training_data = getattr(args, 'path_to_training_data')
    #path to patches is the location all the validation dataset is stored
    global path_to_validation_data
    path_to_validation_data = getattr(args, 'path_to_validation_data')
    #path to patches is the location the output images will be stored
    global path_to_predictions
    path_to_predictions = getattr(args, 'path_to_predictions')
    #path to patches is the location the output images will be stored
    global checkpoint_path
    checkpoint_path = getattr(args, 'model_checkpoint_path')
        



def plot_roc_curve(train_pred, train_true, valid_pred, valid_true):
    
    # Pull out the training and validation labels
    true_tor_train = train_true[:,:,:,1]
    true_tor_valid = valid_true[:,:,:,1]
    
    # Make each dataset into a 1D vector
    t_probs_train = train_pred.ravel()
    t_true_train = true_tor_train.ravel()
    t_probs_valid = valid_pred.ravel()
    t_true_valid = true_tor_valid.ravel()

    # Calculate the probability of false detection and the probability of detection for the training and validation set
    pofds_train, pods_train = gewitter_functions.get_points_in_roc_curve(t_probs_train,t_true_train,threshold_arg=np.linspace(0,1,21))
    pofds_valid, pods_valid = gewitter_functions.get_points_in_roc_curve(t_probs_valid,t_true_valid,threshold_arg=np.linspace(0,1,21))
    
    # Plot the validation ROC curve
    plt.figure()
    plt.plot([0,1], linestyle='--')
    plt.plot(pofds_valid, pods_valid, linestyle='-', color='red', label='Validation')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid()
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path_to_predictions + '/roc_curve_val.png',dpi=500)
    
    # Plot the training ROC Curve
    plt.figure()
    plt.plot([0,1], linestyle='--')
    plt.plot(pofds_train, pods_train, linestyle='-', color='dodgerblue', label='Training')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid()
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path_to_predictions + '/roc_curve_train.png',dpi=500)
    
    # Plot the training and validation ROC Curve
    plt.figure()
    plt.plot([0,1], linestyle='--')
    plt.plot(pofds_valid, pods_valid, linestyle='-', color='red', label='Validation')
    plt.plot(pofds_train, pods_train, linestyle='-', color='dodgerblue', label='Training')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid()
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path_to_predictions + '/roc_curve_train_val.png',dpi=500)

    
def plot_reliability_diagram(train_pred, train_true, valid_pred, valid_true):
    
    # Load in the truth labels for each dataset
    true_tor_train = train_true[:,:,:,1]
    true_tor_valid = valid_true[:,:,:,1]
    
    # Calculate the observed frequencies and predicted probabilities for binned data
    t_probs_train = train_pred.ravel()
    t_true_train = true_tor_train.ravel()
    prob_true_t_train, prob_pred_t_train = calibration_curve(t_true_train, t_probs_train, n_bins=20)
    t_probs_valid = valid_pred.ravel()
    t_true_valid = true_tor_valid.ravel()
    prob_true_t_valid, prob_pred_t_valid = calibration_curve(t_true_valid, t_probs_valid, n_bins=20)
    
    # Make the validation reliability diagram
    plt.figure()
    plt.plot([0,1], linestyle='--')
    plt.plot(prob_pred_t_valid,prob_true_t_valid, linestyle='-', color='red', label='Validation')
    plt.ylabel("Observed Frequency")
    plt.xlabel("Predicted Probability")
    plt.grid()
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path_to_predictions + '/reliability_diagram_val.png',dpi=500)

    # Make the training reliability diagram
    plt.figure()
    plt.plot([0,1], linestyle='--')
    plt.plot(prob_pred_t_train,prob_true_t_train, linestyle='-', color='dodgerblue', label='Training')
    plt.ylabel("Observed Frequency")
    plt.xlabel("Predicted Probability")
    plt.grid()
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path_to_predictions + '/reliability_diagram_train.png',dpi=500)
    
    # Make the training and validation reliability diagram
    plt.figure()
    plt.plot([0,1], linestyle='--')
    plt.plot(prob_pred_t_valid,prob_true_t_valid, linestyle='-', color='red',  label='Validation')
    plt.plot(prob_pred_t_train,prob_true_t_train, linestyle='-', color='dodgerblue', label='Training')
    plt.ylabel("Observed Frequency")
    plt.xlabel("Predicted Probability")
    plt.grid()
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path_to_predictions + '/reliability_diagram_train_val.png',dpi=500)
    
    
def plot_performance_diagram(train_pred, train_true, valid_pred, valid_true):
    
    # Load in the truth and predictions for each dataset
    true_tor_train = train_true[:,:,:,1]
    y_preds_train = train_pred.ravel()
    t_true_train = true_tor_train.ravel()
    true_tor_val = valid_true[:,:,:,1]
    y_preds_val = valid_pred.ravel()
    t_true_val = true_tor_val.ravel()
    
    # Define the threholds we will use, this will make 20 evenly spaced points between 0 and 1. 
    threshs = np.linspace(0,1,21)

    #pre-allocate a vector full of 0s to fill with our results 
    pods_train = np.zeros(len(threshs))
    srs_train = np.zeros(len(threshs))
    csis_train = np.zeros(len(threshs))
    pods_val = np.zeros(len(threshs))
    srs_val = np.zeros(len(threshs))
    csis_val = np.zeros(len(threshs))

    # Loop through all the thresholds
    for i,t in enumerate(tqdm.tqdm(threshs)):

        # Make a dummy binary array full of 0s
        y_preds_bi_train = np.zeros(y_preds_train.shape,dtype=int)
        y_preds_bi_val = np.zeros(y_preds_val.shape,dtype=int)
        
        #find where the prediction is greater than or equal to the threshold
        idx_train = np.where(y_preds_train >= t)
        idx_val = np.where(y_preds_val >= t)

        #set those indices to 1
        y_preds_bi_train[idx_train] = 1
        y_preds_bi_val[idx_val] = 1

        #get the contingency table again 
        table_train = gewitter_functions.get_contingency_table(y_preds_bi_train,t_true_train)
        table_val = gewitter_functions.get_contingency_table(y_preds_bi_val,t_true_val)

        #calculate pod, sr and csi 
        pods_train[i] = get_pod(table_train)
        srs_train[i] = get_sr(table_train)
        csis_train[i] = csi_from_sr_and_pod(srs_train[i],pods_train[i])
        pods_val[i] = get_pod(table_val)
        srs_val[i] = get_sr(table_val)
        csis_val[i] = csi_from_sr_and_pod(srs_val[i],pods_val[i])

    # Plot the validation performance diagram
    plt.figure()
    ax = gewitter_functions.make_performance_diagram_axis()
    ax.plot(srs_val,pods_val,'-',color='red',markerfacecolor='w',lw=2,label='Validation');
    for i,t in enumerate(threshs):
        text = np.char.ljust(str(np.round(t,2)),width=4,fillchar='0')
        ax.text(srs_val[i]+0.02,pods_val[i]+0.02,text,fontsize=9,color='white')
    plt.legend(loc="best")
    plt.savefig(path_to_predictions + '/performance_diagram_val.png',dpi=500)
    
    # Plot the training performance diagram
    plt.figure()
    ax = gewitter_functions.make_performance_diagram_axis()
    ax.plot(srs_train,pods_train,'-',color='dodgerblue',markerfacecolor='w',lw=2,label="Training");
    for i,t in enumerate(threshs):
        text = np.char.ljust(str(np.round(t,2)),width=4,fillchar='0')
        ax.text(srs_train[i]+0.02,pods_train[i]+0.02,text,fontsize=9,color='white')
    plt.legend(loc="best")
    plt.savefig(path_to_predictions + '/performance_diagram_train.png',dpi=500)
    
    # Plot the training and validation performance diagram
    plt.figure()
    ax = gewitter_functions.make_performance_diagram_axis()
    ax.plot(srs_train,pods_train,'-',color='dodgerblue',markerfacecolor='w',lw=2,label='Training');
    ax.plot(srs_val,pods_val,'-',color='red',markerfacecolor='w',lw=2,label='Validation');
    # for i,t in enumerate(threshs):
        # text = np.char.ljust(str(np.round(t,2)),width=4,fillchar='0')
        # ax.text(srs[i]+0.02,pods[i]+0.02,text,fontsize=9,color='white')
    plt.legend(loc="best")
    plt.savefig(path_to_predictions + '/performance_diagram_train_val.png',dpi=500)







def main():

    # Get the inputs from the .sh file
    get_arguments()  

    #Load in the dataset
    print("Loading Training")

    # Define the shape of the training data
    x_tensor_shape = (None, 32, 32, 12)
    y_tensor_shape = (None, 32, 32, 2)
    elem_spec = (tf.TensorSpec(shape=x_tensor_shape, dtype=tf.float64), tf.TensorSpec(shape=y_tensor_shape, dtype=tf.float32))

    # Load in the training dataset:
    ds_train = tf.data.experimental.load(path_to_training_data, elem_spec)
    ds_train = ds_train#.unbatch()

    print("Loading Validation")
    # Define the shape of the validation data
    x_tensor_shape_val = (32, 32, 12)
    y_tensor_shape_val = (32, 32, 2)
    elem_spec_val = (tf.TensorSpec(shape=x_tensor_shape_val, dtype=tf.float64), tf.TensorSpec(shape=y_tensor_shape_val, dtype=tf.float32))

    # Load in the validation data
    ds_validate = tf.data.experimental.load(path_to_validation_data, elem_spec_val)
    ds_validate = ds_validate.batch(256)

    # Extract the input and the labels from training and validation
    x_train = np.concatenate([x for x,y in ds_train], axis=0)
    y_train = np.concatenate([y for x,y in ds_train], axis=0)
    x_valid = np.concatenate([x for x,y in ds_validate], axis=0)
    y_valid = np.concatenate([y for x,y in ds_validate], axis=0)


    # Load in the ML model
    fss = make_fractions_skill_score(2, 2, c=1.0, cutoff=0.5, want_hard_discretization=False)
    model = keras.models.load_model(checkpoint_path, 
                    custom_objects={'fractions_skill_score': fss, 
                                    'MaxCriticalSuccessIndex': MaxCriticalSuccessIndex})
        
    # Evaluate the unet on the training data
    y_hat = model.predict(x_train)

    # Pull out training predictions
    predicted_tor_train = y_hat[:,:,:,1]

    #evaluate the unet on the validation data
    y_hat = model.predict(x_valid)

    # Pull out validation predictions
    predicted_tor_valid = y_hat[:,:,:,1]

    # Make the ROC Curve
    print('Making ROC Curve')
    plot_roc_curve(train_pred=predicted_tor_train, train_true=y_train, valid_pred=predicted_tor_valid, valid_true=y_valid)

    # Make the Reliability Diagram
    print('Making Reliability Diagram')
    plot_reliability_diagram(train_pred=predicted_tor_train, train_true=y_train, valid_pred=predicted_tor_valid, valid_true=y_valid)

    # Make the Performance Diagram
    print('Making Performance Diagram')
    plot_performance_diagram(train_pred=predicted_tor_train, train_true=y_train, valid_pred=predicted_tor_valid, valid_true=y_valid)



if __name__ == "__main__":
    main()