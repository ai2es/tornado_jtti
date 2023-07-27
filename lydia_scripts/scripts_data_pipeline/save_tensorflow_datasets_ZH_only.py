import tensorflow as tf
from tensorflow import keras
import xarray as xr
from tensorflow.keras.utils import to_categorical
import numpy as np
import glob
import dask
import argparse
import os


def get_arguments():
    #Define the strings that explain what each input variable means
    INPUT_XARRAY_DIR_HELP_STRING = 'The directory where the xarray patch files are stored. This directory should start from the root directory \'/\'.'
    OUTPUT_PATH_HELP_STRING = 'The directory where the tensorflow datasets will be saved. This directory should start from the root directory \'/\'.'
    BATCH_SIZE_HELP_STRING = 'The batch size we want to use for the tensorflow dataset.'
    PATCHES_TYPE_HELP_STRING = 'dataset_patches_type should be a list containing the patch types we want to include in training and 50/50 validation sets \'n\' for nontor patches, \'t\' for tornadic patches, \'s\' for sigtor patches'
    LABELS_TYPE_HELP_STRING = 'dataset_labels_type should be a string, either \'int\', for integer labels, or \'onehot\', for onehot vector labels'
    METADATA_PATH_HELP_STRING = 'The path to the training metadata file used to train the ML model we are evaluating.'
    
    #Tell python what arguments to expect from the .sh file
    INPUT_ARG_PARSER = argparse.ArgumentParser()
        
    INPUT_ARG_PARSER.add_argument(
        '--input_xarray_path', type=str, required=True,
        help=INPUT_XARRAY_DIR_HELP_STRING)
    
    INPUT_ARG_PARSER.add_argument(
        '--output_path', type=str, required=True,
        help=OUTPUT_PATH_HELP_STRING)
        
    INPUT_ARG_PARSER.add_argument(
        '--batch_size', type=int, required=True,
        help=BATCH_SIZE_HELP_STRING)

    INPUT_ARG_PARSER.add_argument(
        '--training_data_metadata_path', type=str, required=True,
        help=METADATA_PATH_HELP_STRING)
        
    INPUT_ARG_PARSER.add_argument(
        '--dataset_patches_type', type=str, required=True,
        help=PATCHES_TYPE_HELP_STRING)

    INPUT_ARG_PARSER.add_argument(
        '--dataset_labels_type', type=str, required=True,
        help=LABELS_TYPE_HELP_STRING)

    INPUT_ARG_PARSER.add_argument(
        '--dry_run', action='store_true',
        help='For testing. Display output and input file/directory paths')
        
    #Pull out all of the input strings from the .sh file
    args = INPUT_ARG_PARSER.parse_args()
    #Index primer indicates the day of data that we are looking at in this particular run of this code
    global input_xarray_path
    input_xarray_path = getattr(args, 'input_xarray_path')
    # Output path is the path to where the tensorflow datasets will be stored
    global output_path
    output_path = getattr(args, 'output_path')
    #training data metadata path is the file path of the training metadata we will use to normalize the wofs predictions
    global training_data_metadata_path
    training_data_metadata_path = getattr(args, 'training_data_metadata_path')
    # Batch size will be the batch size of the training data
    global batch_size
    batch_size = getattr(args, 'batch_size')
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
    
    global dry_run
    dry_run = getattr(args, 'dry_run')

    print(args)


def sort_patches(all_patches_dirs):

    # We are making datasets for training, 50/50 validation and natural validation sets---initialize these empty arrays
    training_patches = []
    validation_patches = []
    natural_validation_patches = []

    # Loop through all the directories
    for directory in all_patches_dirs:
        # Load all the files from that day
        all_patches_in_day = glob.glob(directory + '/*.nc')

        # We want each day to be in EITHER the validation set OR the training set, but not it both
        # Load in the data for both training and validation, but don't add it until we know which one we want
        to_add_training = []
        to_add_validation = []
        to_add_natural_validation = []

        for patches in all_patches_in_day:
            # If we don't want nontor patches to be in our final dataset per the .sh file, don't add them
            if ('n' not in dataset_patches_type) and ('nontor' in patches):
                continue

            # If this file is a validation file, add it to the correct list based on the filename
            if 'validation' in patches:

                # This patch is natural validation
                if 'natural' in patches:
                    to_add_natural_validation.append(patches)

                # This patch is 50/50 validation
                else:
                    to_add_validation.append(patches)

            # Otherwise, this patch is a training file, so add it to the training list
            else:
                to_add_training.append(patches)

        # If there is something in the validation list, this is a validation day, so don't add to the training dataset
        if to_add_validation != []:

            # The 50/50 validation files are made from the training datasets, so add them
            validation_patches = validation_patches + to_add_training

            # The natural_validation files are made from the natural validation datasets, so add those
            natural_validation_patches = natural_validation_patches + to_add_natural_validation
            to_add_training = []
            to_add_validation = []
            to_add_natural_validation = []
        else:
            # There were no validation files in this day, so we can add the training patches to the training dataset
            training_patches = training_patches + to_add_training
            to_add_training = []
            to_add_validation = []
        
    print('total training patches:',len(training_patches))
    #print(training_patches[0])
    print('total 50/50 validation patches:',len(validation_patches))
    #print(validation_patches[0])
    print('total natural validation patches:',len(natural_validation_patches))
    #print(natural_validation_patches[0])

    # If a patch ends up in both the training and validation dataset, throw an error
    for patch in validation_patches:
        if patch in training_patches:
            print('**************************THIS FILE WAS IN TRAINING AND VALIDATION DATA:***********************************')
            print(patch)
            raise NameError('Patch in both training and validation datasets')

    return training_patches, validation_patches, natural_validation_patches



def save_training_dataset(training_patches, vertical_levels=[1,2,3,4,5,6,7,8,9,10,11,12]):
    if len(training_patches) <= 0: print("list of train files is empty"); return
    
    tmp = xr.load_dataset('/ourdisk/hpc/ai2es/tornado/learning_patches_V2/xarray/3D_light/size_32/forecast_window_5/2013/20130129/patches_nontor_20130130_03.nc')
    print(tmp)

    # Load in the training data, combine into one array, and make sure we only have the 12 desired vertical levels
    print('Opening Training Data')
    training_array = xr.open_mfdataset(training_patches, concat_dim='patch',
                                       combine='nested', parallel=True, engine='netcdf4')
    training_array = training_array.sel(z=vertical_levels)
    print("Training Array Number of Gigabytes:", training_array.nbytes / 1e9)

    # Calculate the bulk statistics of the training dataset
    print("Calculating Training Metadata")
    mean_train_ZH = np.nanmean(training_array.ZH.values)
    std_train_ZH = np.nanstd(training_array.ZH.values)
    
    # Make the path to save the output files
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

    # Make sure the directory structure exists
    #out_dir = output_path + '/training_'+ dataset_labels_type + '_' + patches_type_path
    out_dir = os.path.join(output_path, f'training_{dataset_labels_type}_{patches_type_path}')
    '''if not os.path.exists(out_dir):
        print(f"Make dir {out_dir} [dry_run={dry_run}]")
        if not dry_run:
            os.mkdir(out_dir)
    '''
    #tf_dir = out_dir + '/training_ZH_only.tf'
    tf_dir = os.path.join(out_dir, 'training_ZH_only.tf')
    if os.path.exists(output_path) and not os.path.exists(tf_dir):
        print(f"Make dir {tf_dir} [dry_run={dry_run}]")
        if not dry_run:
            #os.mkdir(tf_dir)
            os.makedirs(tf_dir)


    #Save out the metadata of training data
    print("Saving Training Metadata")
    training_array_metadata = training_array.drop(['ZH', 'VOR', 'DIV', 'labels'])
    training_array_metadata['ZH_mean'] = ([], mean_train_ZH)
    training_array_metadata['ZH_std'] = ([], std_train_ZH)
    train_meta_fname = output_path + '/training_'+ dataset_labels_type + '_' + patches_type_path + '/training_metadata_ZH_only.nc'
    print(f"Making meta data file {train_meta_fname} [dry_run={dry_run}]")
    if not dry_run: 
        training_array_metadata.to_netcdf(train_meta_fname)
    training_array_metadata.close()


    # Normalize the arrays to a mean of 0 and std of 1
    training_array.ZH.values = ((training_array.ZH - mean_train_ZH)/std_train_ZH).compute()

    # Make the input data for training---combine all variable fields
    print("Making x_train")
    x_train = training_array.ZH.values

    # Make the labels for training, given the label type
    print("Making y_train")
    # We want int labels
    if dataset_labels_type == 'int':
        labels = np.where(training_array.labels.values > 0, 1, 0)
        y_train = labels.reshape(labels.shape + (1,))
        
    # We want onehot vector labels
    else:
        labels = np.where(training_array.labels.values > 0, 1, 0)
        y_train = keras.utils.to_categorical(labels, num_classes=2)
        

    print("Making Tensorflow dataset")
    #Make the tensorflow dataset
    ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))

    #Shuffle and batch the data
    ds_train = ds_train.shuffle(x_train.shape[0], seed=24).batch(batch_size)
    print("Training Dataset Elem Spec:", ds_train.element_spec)

    # Save out the dataset
    ds_fname = os.path.join(output_path, f'training_{dataset_labels_type}_{patches_type_path}', 'training_ZH_only.tf')
    print(f"[train ds] Saving tf Dataset {ds_fname} [dry_run={dry_run}]")
    #tf.data.experimental.save(ds_train, output_path + '/training_'+ dataset_labels_type + '_' + patches_type_path + '/training_ZH_only.tf')
    if not dry_run:
        ds_train.save(ds_fname)
    training_array.close()

    return


def save_validation_dataset(validation_patches, natural, mean_train_ZH, std_train_ZH, gridrad_levels=[1,2,3,4,5,6,7,8,9,10,11,12]):
    if len(validation_patches) <= 0: print("list of val files is empty"); return

    # Open the validation dataset
    validation_array = xr.open_mfdataset(validation_patches, concat_dim='patch',combine='nested', parallel=True, engine='netcdf4', coords='minimal')
    # Make sure we have selected the vertical levels we want
    validation_array = validation_array.sel(z=gridrad_levels)

    print("Validation Dataset Number of Megabytes:", validation_array.nbytes / 1e6)


    #calculate the metadata of the validation dataset
    print('Calculating validation metadata')
    mean_validate_ZH = np.nanmean(validation_array.ZH.values)
    std_validate_ZH = np.nanstd(validation_array.ZH.values)
    

    # Make the path to save the output files
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

    # Make sure the directory structure exists
    dirname = output_path + '/validation_'+ dataset_labels_type + '_' + patches_type_path
    if not os.path.exists(dirname):
        print(f"Make dir {dirname} [dry_run={dry_run}]")
        if not dry_run:
            os.mkdir(dirname)
    
    # Make the natural dataset path
    output_filename = None
    if natural:
        val_dir = output_path + '/validation_'+ dataset_labels_type + '_' + patches_type_path + '/natural_validation_ZH_only.tf'
        if not os.path.exists(val_dir):
            print(f"Make dir {val_dir} [dry_run={dry_run}]")
            if not dry_run:
                os.mkdir(val_dir)
        output_filename = val_dir

    # Make the 50/50 dataset path
    else:
        val_dir = output_path + '/validation_'+ dataset_labels_type + '_' + patches_type_path + '/validation1_ZH_only.tf'
        if not os.path.exists(val_dir):
            print(f"Make dir {val_dir} [dry_run={dry_run}]")
            if not dry_run:
                os.mkdir(val_dir)
        output_filename = val_dir

    
    # Save out the metadata of validation data
    validation_array_metadata = validation_array.drop(['ZH', 'VOR', 'DIV', 'labels'])
    validation_array_metadata['ZH_mean'] = ([], mean_validate_ZH)
    validation_array_metadata['ZH_std'] = ([], std_validate_ZH)
    #validation_array_metadata.to_netcdf(output_path + '/validation_'+ dataset_labels_type + '_' + patches_type_path + '/validation1_metadata_ZH_only.nc')
    val_fname = output_path + f'/validation_{dataset_labels_type}_{patches_type_path}/validation1_metadata_ZH_only.nc'
    print(f"[val] Save {val_fname} [dry_run={dry_run}]")
    if not dry_run:
        validation_array_metadata.to_netcdf(val_fname)
    validation_array_metadata.close()


    #Normalizing x_validate using the means and stds from the TRAINING dataset
    print("Normalizing x_validate")
    validation_array.ZH.values = ((validation_array.ZH - mean_train_ZH) / std_train_ZH).compute()
    

    # Construct the ML inputs---combine all variable fields
    x_validate = validation_array.ZH.values
    
    #Load in the y_train
    print("Making y_validate")
    # We want int labels
    if dataset_labels_type == 'int':
        labels = np.where(validation_array.labels.values > 0, 1, 0)
        y_validate = labels.reshape(labels.shape + (1,))
        
    # We want onehot vector labels
    else:
        labels = np.where(validation_array.labels.values > 0, 1, 0)
        y_validate = keras.utils.to_categorical(labels, num_classes=2)

    #Make the tensorflow dataset
    print("Making tensorflow dataset")
    ds_validate = tf.data.Dataset.from_tensor_slices((x_validate, y_validate))
    #tf.data.experimental.save(ds_validate, output_filename)
    print(f"[val ds] Save {output_filename} [dry_run={dry_run}]")
    if not dry_run:
        ds_validate.save(output_filename)
    validation_array.close()

    #return


def create_dataset(patches, dsettype='train', vertical_levels=[1,2,3,4,5,6,7,8,9,10,11,12]):
    if len(patches) <= 0: print("list of files is empty"); return

    # Load data, combine into one array, and extract desired vertical levels
    print(f'Opening {dsettype} Data') #print('Opening Training Data')
    data_array = xr.load_mfdataset(patches, concat_dim='patch', combine='nested', parallel=True, engine='netcdf4')
    data_array = data_array.sel(z=vertical_levels)
    print(f"{dsettype} Array Number of Gigabytes:", data_array.nbytes/1e9) #print("Training Array Number of Gigabytes:", data_array.nbytes/1e9)

    # Calculate the bulk statistics of the training dataset
    print(f"Calculating {dsettype} Metadata") #print("Calculating Training Metadata")
    mean_ZH = np.nanmean(data_array.ZH.values)
    std_ZH = np.nanstd(data_array.ZH.values)
    
    # Identify storm type from file name
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

    # Make the path to save the output files
    #TODO {dsettype}
    # Make sure the directory structure exists
    #fbasename = output_path + '/training_'+ dataset_labels_type + '_' + patches_type_path
    fbasename = os.path.join(output_path, f'training_{dataset_labels_type}_{patches_type_path}')
    '''
    if not os.path.exists(fbasename):
        print(f"Make dir {fbasename} [dry_run={dry_run}]")
        if not dry_run:
            #os.mkdir(output_path + '/training_'+ dataset_labels_type + '_' + patches_type_path)
            os.mkdir(fbasename)
    '''
    tf_fname = os.path.join(fbasename, 'training_ZH_only.tf')
    if not os.path.exists(tf_fname):
        print(f"Make dir {tf_fname} [dry_run={dry_run}]")
        if not dry_run:
            #os.mkdir(output_path + '/training_'+ dataset_labels_type + '_' + patches_type_path + '/training_ZH_only.tf')
            #os.mkdir(tf_fname)
            os.makedirs(tf_fname)


    # Save metadata of training data
    print("Saving Training Metadata")
    training_array_metadata = data_array.drop(['ZH', 'VOR', 'DIV', 'labels'])
    training_array_metadata['ZH_mean'] = ([], mean_ZH)
    training_array_metadata['ZH_std'] = ([], std_ZH)
    #training_array_metadata.to_netcdf(output_path + '/training_'+ dataset_labels_type + '_' + patches_type_path + '/training_metadata_ZH_only.nc')
    meta_fname = os.path.join(output_path, f'{dsettype}_{dataset_labels_type}_{patches_type_path}', f'{dsettype}_metadata_ZH_only.nc')
    print(f"Save train meta data {meta_fname} [dry_run={dry_run}]")
    if not dry_run:
        training_array_metadata.to_netcdf(meta_fname)
    training_array_metadata.close()


    # Normalize the arrays to a mean of 0 and std of 1
    data_array.ZH.values = ((data_array.ZH - mean_ZH) / std_ZH).compute()

    # Make the input data for training---combine all variable fields
    print(f"Making X {dsettype}")
    X = data_array.ZH.values

    # Make the labels for training, given the label type
    print(f"Making Y {dsettype}")
    # We want int labels
    if dataset_labels_type == 'int':
        labels = np.where(data_array.labels.values > 0, 1, 0)
        n_tors = np.sum(labels)
        n_nonors = labels.size - n_tors
        print(f" # tors {n_tors} ({n_tors/labels.size}%)")
        print(f" # non tors {n_nonors} ({n_nonors/labels.size}%)")
        Y = labels.reshape(labels.shape + (1,))
        
    # We want onehot vector labels
    else:
        labels = np.where(data_array.labels.values > 0, 1, 0)
        Y = keras.utils.to_categorical(labels, num_classes=2)
        

    print("Making Tensorflow dataset")
    # TODO elem spec float32
    ds = tf.data.Dataset.from_tensor_slices((X, Y)) #ds_train

    # Shuffle and batch the data
    ds = ds.shuffle(X.shape[0], seed=24).batch(batch_size)
    print("Training Dataset Elem Spec:", ds.element_spec)

    # Save tf dataset
    #tf.data.experimental.save(ds_train, output_path + '/training_'+ dataset_labels_type + '_' + patches_type_path + '/training_ZH_only.tf')
    ds_fname = os.path.join(output_path, f'{dsettype}_{dataset_labels_type}_{patches_type_path}/{dsettype}_ZH_only.tf')
    print(f"Save tf dataset {ds_fname} [dry_run={dry_run}]")
    if not dry_run:
        ds.save(ds_fname)
    data_array.close()

    return ds


def main():

    #get the inputs from the .sh file
    get_arguments()  

    # Load in all the directories that will be included in the tensorflow datasets
    print("Loading in all directories")
    all_patches_dirs = glob.glob(input_xarray_path + '2013/*') + glob.glob(input_xarray_path + '2016/*')
    #all_patches_dirs = glob.glob(input_xarray_path + '/*/*')
    all_patches_dirs.sort()

    # Pull in all the patches files and sort them into validation, natural validation, and training
    training_patches, validation_patches, natural_validation_patches = sort_patches(all_patches_dirs)

    print(f"n train {len(training_patches)}; {training_patches[0]}")
    print(f"n val {len(validation_patches)}")
    print(f"n val nat {len(natural_validation_patches)}")

    # Create the training dataset
    save_training_dataset(training_patches)

    training_metadata = xr.open_dataset(training_data_metadata_path)
    global mean_train_ZH
    mean_train_ZH = float(training_metadata.ZH_mean.values)
    global std_train_ZH
    std_train_ZH = float(training_metadata.ZH_std.values)
    training_metadata.close()
    del training_metadata

    # Create the 50/50 validation dataset
    natural = False
    save_validation_dataset(validation_patches, natural, mean_train_ZH, std_train_ZH)

    for patches in natural_validation_patches:
        ds = xr.load_mfdataset([patches]) #xr.open_mfdataset([patches])
        if 'DIV' not in ds:
            print(patches)
            print(ds)

    # Create the natural validation dataset
    natural = True
    save_validation_dataset(natural_validation_patches, natural, mean_train_ZH, std_train_ZH)


if __name__ == "__main__":
    main()


