import numpy as np
import argparse
import tensorflow as tf 
from tensorflow import keras
import tqdm
import sys
import glob
import xarray as xr
from sklearn.calibration import calibration_curve

sys.path.append("/home/lydiaks2/tornado_project/WAF_ML_Tutorial_Part1/scripts/")
import gewitter_functions
from gewitter_functions import get_pod,get_sr,csi_from_sr_and_pod

sys.path.append("/home/lydiaks2/tornado_project/")
from custom_losses import make_fractions_skill_score
from custom_metrics import MaxCriticalSuccessIndex

def get_arguments():
    #Define the strings that explain what each input variable means
    MODELS_DIR_HELP_STRING = 'Path to models is the location of the models to compare. This directory should start from the root directory \'/\'.'
    DATASETS_DIR_HELP_STRING = 'Path to tf datasets is the location of the training and both validation datasets. This directory should start from the root directory \'/\'.'
    OUTPUT_PATH_HELP_STRING = 'Path to output is the location we will store the output of this script. This directory should start from the root directory \'/\'.'
    PATCHES_TYPE_HELP_STRING = 'dataset_patches_type should be a list containing the patch types we want to include in training and 50/50 validation sets \'n\' for nontor patches, \'t\' for tornadic patches, \'s\' for sigtor patches'
    LABELS_TYPE_HELP_STRING = 'dataset_labels_type should be a string, either \'int\', for integer labels, or \'onehot\', for onehot vector labels'
    

    #Tell python what arguments to expect from the .sh file
    INPUT_ARG_PARSER = argparse.ArgumentParser()
        
    INPUT_ARG_PARSER.add_argument(
        '--path_to_models', type=str, required=True,
        help=MODELS_DIR_HELP_STRING)
        
    INPUT_ARG_PARSER.add_argument(
        '--path_to_tf_datasets', type=str, required=True,
        help=DATASETS_DIR_HELP_STRING)
        
    INPUT_ARG_PARSER.add_argument(
        '--path_to_output', type=str, required=True,
        help=OUTPUT_PATH_HELP_STRING)
        
    INPUT_ARG_PARSER.add_argument(
        '--dataset_patches_type', type=str, required=True,
        help=PATCHES_TYPE_HELP_STRING)

    INPUT_ARG_PARSER.add_argument(
        '--dataset_labels_type', type=str, required=True,
        help=LABELS_TYPE_HELP_STRING)

    #Pull out all of the input strings from the .sh file
    args = INPUT_ARG_PARSER.parse_args()

    # Path to models is the location of the models trained in tensorboard that we will compare
    global path_to_models
    path_to_models = getattr(args, 'path_to_models')
    
    # Path to tf datasets is the location of the training and validation datasets
    global path_to_tf_datasets
    path_to_tf_datasets = getattr(args, 'path_to_tf_datasets')
    
    # Path to output is the location to put the script output
    global path_to_output
    path_to_output = getattr(args, 'path_to_output')
    
    # Dataset_patches_type indicates which kinds of patches go into the training and 50/50 validation sets
    global dataset_patches_type
    dataset_patches_type = getattr(args, 'dataset_patches_type')
    if dataset_patches_type not in ['nts', 'nst', 'tns', 'tsn', 'stn', 'snt', 'tn', 'nt', 'st', 'ts', 't']:
        raise NameError('dataset_patches_type is invalid')
    
    # Dataset_labels_type indicates whether we will save the labels as onehot vectors or as integers
    global dataset_labels_type
    dataset_labels_type = getattr(args, 'dataset_labels_type')
    if dataset_labels_type not in ['int', 'onehot']:
        raise NameError('dataset_labels_type is invalid')


def load_datasets():

    # Make the path to open the datasets
    if 's' in dataset_patches_type and 't' in dataset_patches_type and 'n' in dataset_patches_type:
        patches_type_path = 'nontor_tor_sigtor'
    elif 't' in dataset_patches_type and 's' in dataset_patches_type:
        patches_type_path = 'tor_sigtor'
    elif 't' in dataset_patches_type and 'n' in dataset_patches_type:
        patches_type_path = 'nontor_tor'
    elif 't' in dataset_patches_type:
        patches_type_path = 'tor'
    else:
        print('dataset_patches_type =', dataset_patches_type)
        raise NameError('dataset_patches_type invalid')

    # Define the location of the datasets
    path_to_training_data = path_to_tf_datasets + '/training_'+ dataset_labels_type + '_' + patches_type_path + '/training_ZH_only.tf'
    path_to_validation_data = path_to_tf_datasets + '/validation_'+ dataset_labels_type + '_' + patches_type_path + '/validation1_ZH_only.tf'
    path_to_natural_validation_data = path_to_tf_datasets + '/validation_'+ dataset_labels_type + '_' + patches_type_path + '/natural_validation_ZH_only.tf'
    
    # Load in the dataset
    print("Loading Training")

    # Define the shape of our data
    x_tensor_shape = (None, 32, 32, 12)
    y_tensor_shape = (None, 32, 32, 2)
    elem_spec = (tf.TensorSpec(shape=x_tensor_shape, dtype=tf.float64), tf.TensorSpec(shape=y_tensor_shape, dtype=tf.float32))

    # Load in the training dataset:
    ds_train = tf.data.experimental.load(path_to_training_data, elem_spec)
    ds_train = ds_train

    print("Loading Validation 50/50")
    # Define the shape of our data
    x_tensor_shape_val = (32, 32, 36)
    y_tensor_shape_val = (32, 32, 2)
    elem_spec_val = (tf.TensorSpec(shape=x_tensor_shape_val, dtype=tf.float64), tf.TensorSpec(shape=y_tensor_shape_val, dtype=tf.float32))

    # Load in the training dataset:
    ds_validate = tf.data.experimental.load(path_to_validation_data, elem_spec_val)
    ds_validate = ds_validate.batch(256)

    print("Loading Validation Natural")
    # Define the shape of our data
    x_tensor_shape_natval = (32, 32, 36)
    y_tensor_shape_natval = (32, 32, 2)
    elem_spec_natval = (tf.TensorSpec(shape=x_tensor_shape_natval, dtype=tf.float64), tf.TensorSpec(shape=y_tensor_shape_natval, dtype=tf.float32))

    # Load in the training dataset:
    ds_natural_validate = tf.data.experimental.load(path_to_natural_validation_data, elem_spec_natval)
    ds_natural_validate = ds_natural_validate.batch(256)

    # Return the datasets
    return ds_train, ds_validate, ds_natural_validate


def evaluate_model(model_path, ds_train, ds_validate, ds_natural_validate):
    
    model_name = model_path.Split('/')[-1]

    # input_array = np.concatenate([zh, vor, div], axis=3)
    x_train = np.concatenate([x for x,y in ds_train], axis=0)
    y_train = np.concatenate([y for x,y in ds_train], axis=0)
    x_valid = np.concatenate([x for x,y in ds_validate], axis=0)
    y_valid = np.concatenate([y for x,y in ds_validate], axis=0)
    x_natvalid = np.concatenate([x for x,y in ds_natural_validate], axis=0)
    y_natvalid = np.concatenate([y for x,y in ds_natural_validate], axis=0)

    # Load in the ML model with our custom objects
    fss = make_fractions_skill_score(2, 2, c=1.0, cutoff=0.5, want_hard_discretization=False)
    model = keras.models.load_model(model_path, 
                    custom_objects={'fractions_skill_score': fss, 
                                    'MaxCriticalSuccessIndex': MaxCriticalSuccessIndex})
        
    #evaluate the unet on each dataset
    y_hat_train = model.predict(x_train)
    y_hat_val = model.predict(x_valid)
    y_hat_natval = model.predict(x_natvalid)
    
    # Load in the truth and predictions for each dataset
    t_true_train = x_train[:,:,:,1].ravel()
    y_preds_train = y_hat_train[:,:,:,1].ravel()

    t_true_val = x_valid[:,:,:,1].ravel()
    y_preds_val = y_hat_val[:,:,:,1].ravel()

    t_true_natval = x_natvalid[:,:,:,1].ravel()
    y_preds_natval = y_hat_natval[:,:,:,1].ravel()
    
    # Define the threholds we will use, this will make 20 evenly spaced points between 0 and 1. 
    threshs = np.linspace(0,1,21)

    #pre-allocate a vector full of 0s to fill with our results 
    pods_train = np.zeros((1,len(threshs)))
    srs_train = np.zeros((1,len(threshs)))
    csis_train = np.zeros((1,len(threshs)))
    pods_val = np.zeros((1,len(threshs)))
    srs_val = np.zeros((1,len(threshs)))
    csis_val = np.zeros((1,len(threshs)))
    pods_natval = np.zeros((1,len(threshs)))
    srs_natval = np.zeros((1,len(threshs)))
    csis_natval = np.zeros((1,len(threshs)))

    # Loop through all the thresholds
    for i,t in enumerate(tqdm.tqdm(threshs)):

        # Make a dummy binary array full of 0s
        y_preds_bi_train = np.zeros(y_preds_train.shape,dtype=int)
        y_preds_bi_val = np.zeros(y_preds_val.shape,dtype=int)
        y_preds_bi_natval = np.zeros(y_preds_natval.shape,dtype=int)
        
        #find where the prediction is greater than or equal to the threshold
        idx_train = np.where(y_preds_train >= t)
        idx_val = np.where(y_preds_val >= t)
        idx_natval = np.where(y_preds_natval >= t)

        #set those indices to 1
        y_preds_bi_train[idx_train] = 1
        y_preds_bi_val[idx_val] = 1
        y_preds_bi_natval[idx_natval] = 1

        #get the contingency table again 
        table_train = gewitter_functions.get_contingency_table(y_preds_bi_train,t_true_train)
        table_val = gewitter_functions.get_contingency_table(y_preds_bi_val,t_true_val)
        table_natval = gewitter_functions.get_contingency_table(y_preds_bi_natval,t_true_natval)

        #calculate pod, sr and csi 
        pods_train[0,i] = get_pod(table_train)
        srs_train[0,i] = get_sr(table_train)
        csis_train[0,i] = csi_from_sr_and_pod(srs_train[i],pods_train[i])
        pods_val[0,i] = get_pod(table_val)
        srs_val[0,i] = get_sr(table_val)
        csis_val[0,i] = csi_from_sr_and_pod(srs_val[i],pods_val[i])
        pods_natval[0,i] = get_pod(table_natval)
        srs_natval[0,i] = get_sr(table_natval)
        csis_natval[0,i] = csi_from_sr_and_pod(srs_natval[i],pods_natval[i])

     #make a dataset of the true and predicted patch data with the metadata
    ds_return = xr.Dataset(data_vars=dict(pods_train = (['model', 'thresh'], pods_train),
                                srs_train = (['model', 'thresh'], srs_train),
                                csis_train = (['model', 'thresh'], csis_train),
                                pods_val = (['model', 'thresh'], pods_val),
                                srs_val = (['model', 'thresh'], srs_val),
                                csis_val = (['model', 'thresh'], csis_val),
                                pods_natval = (['model', 'thresh'], pods_natval),
                                srs_natval = (['model', 'thresh'], srs_natval),
                                csis_natval = (['model', 'thresh'], csis_natval)),
                        coords=dict(model = model_name,
                                thresh = threshs))

    return ds_return





    
def main():

    # Get all the arguments to the .sh script
    get_arguments()

    # Load in the training, validation, and natural_validation datasets
    training, validation, natural_validation = load_datasets()

    # Load in all the models to compare
    all_models = glob.glob(path_to_models + '/*.h5')

    # Loop through all the models, calculate the desired metrics for each one
    models_output = []
    for model in all_models:
        this_model_output = evaluate_model(model, training, validation, natural_validation)
        models_output.append(this_model_output)

    # Combine the data from all the models into one dataset
    output = xr.concat(models_output, 'model')

    # Save out the dataset
    output.to_netcdf(path_to_output)



if __name__ == "__main__":
    main()



