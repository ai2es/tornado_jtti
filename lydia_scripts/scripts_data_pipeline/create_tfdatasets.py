"""
Tensorflow datasets
"""

import tensorflow as tf
from tensorflow import keras
import xarray as xr
from tensorflow.keras.utils import to_categorical
import numpy as np
import glob
import dask
import argparse
import os, re


def get_arguments():    
    #Tell python what arguments to expect from the .sh file
    INPUT_ARG_PARSER = argparse.ArgumentParser()
        
    INPUT_ARG_PARSER.add_argument(
        '--input_xarray_path', type=str, required=True,
        help="The directory where the xarray patch files are stored. This directory should start from the root directory '/'.")
    
    INPUT_ARG_PARSER.add_argument(
        '--output_path', type=str, required=True,
        help="The directory where the tensorflow datasets will be saved. This directory should start from the root directory '/'.")
        
    INPUT_ARG_PARSER.add_argument(
        '--batch_size', type=int, required=True,
        help='The batch size we want to use for the tensorflow dataset.')

    INPUT_ARG_PARSER.add_argument(
        '--training_data_metadata_path', type=str, required=True,
        help='The path to the training metadata file used to train the ML model we are evaluating.')
        
    INPUT_ARG_PARSER.add_argument(
        '--dataset_patches_type', type=str, required=True,
        help="dataset_patches_type should be a list containing the patch types we want to include in training and 50/50 validation sets 'n' for nontor patches, 't' for tornadic patches, 's' for sigtor patches")

    INPUT_ARG_PARSER.add_argument(
        '--dataset_labels_type', type=str, required=True,
        help='dataset_labels_type should be a string, either \'int\', for integer labels, or \'onehot\', for onehot vector labels')

    INPUT_ARG_PARSER.add_argument(
        '-y', '--years', type=str, nargs='+', required=True,
        help='Space delimited list of all the years of data available')

    INPUT_ARG_PARSER.add_argument(
        '-Z', '--ZH_only', action='store_true',
        help='Only save the reflectivity in the Tensorflow Dataset')

    INPUT_ARG_PARSER.add_argument(
        '--dry_run', action='store_true',
        help='For testing. Display output and input file/directory paths')
        
    #Pull out all of the input strings from the .sh file
    args = INPUT_ARG_PARSER.parse_args()

    dataset_patches_type = getattr(args, 'dataset_patches_type')
    if dataset_patches_type not in ['nts', 'nst', 'tns', 'tsn', 'stn', 'snt', 'tn', 'nt', 'st', 'ts', 't']:
        raise NameError('dataset_patches_type is invalid')

    dataset_labels_type = getattr(args, 'dataset_labels_type')
    if dataset_labels_type not in ['int', 'onehot']:
        raise NameError('dataset_labels_type is invalid')
    
    #ZH_only = getattr(args, 'ZH_only')
    
    #dry_run = getattr(args, 'dry_run')

    return args


def save_training_dataset(training_patches, vertical_levels=[1,2,3,4,5,6,7,8,9,10,11,12]):

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
    dataset_patches_type = args.dataset_patches_type
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
    output_path = args.output_path
    dataset_labels_type = args.dataset_labels_type
    if not os.path.exists(output_path + '/training_'+ dataset_labels_type + '_' + patches_type_path):
        os.mkdir(output_path + '/training_'+ dataset_labels_type + '_' + patches_type_path)
    if not os.path.exists(output_path + '/training_'+ dataset_labels_type + '_' + patches_type_path + '/training_ZH_only.tf'):
        os.mkdir(output_path + '/training_'+ dataset_labels_type + '_' + patches_type_path + '/training_ZH_only.tf')


    #Save out the metadata of training data
    print("Saving Training Metadata")
    training_array_metadata = training_array.drop(['ZH', 'VOR', 'DIV', 'labels'])
    training_array_metadata['ZH_mean'] = ([], mean_train_ZH)
    training_array_metadata['ZH_std'] = ([], std_train_ZH)
    training_array_metadata.to_netcdf(output_path + '/training_'+ dataset_labels_type + '_' + patches_type_path + '/training_metadata_ZH_only.nc')
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
    ds_train = ds_train.shuffle(x_train.shape[0], seed=24).batch(args.batch_size)
    print("Training Dataset Elem Spec:", ds_train.element_spec)

    # Save out the dataset
    #tf.data.experimental.save(ds_train, output_path + '/training_'+ dataset_labels_type + '_' + patches_type_path + '/training_ZH_only.tf')
    ds_train.save(output_path + '/training_'+ dataset_labels_type + '_' + patches_type_path + '/training_ZH_only.tf')
    training_array.close()

    return


def save_validation_dataset(validation_patches, natural, mean_train_ZH, std_train_ZH, gridrad_levels=[1,2,3,4,5,6,7,8,9,10,11,12]):
    
    # Open the validation dataset
    validation_array = xr.open_mfdataset(validation_patches, concat_dim='patch',combine='nested', parallel=True, engine='netcdf4', coords='minimal')
    # Make sure we have selected the vertical levels we want
    validation_array = validation_array.sel(z=gridrad_levels)

    print("Validation Dataset Number of Megabytes:", validation_array.nbytes/1e6)


    #calculate the metadata of the validation dataset
    print('Calculating validation metadata')
    mean_validate_ZH = np.nanmean(validation_array.ZH.values)
    std_validate_ZH = np.nanstd(validation_array.ZH.values)
    

    # Make the path to save the output files
    dataset_patches_type = args.dataset_patches_type
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
    output_path = args.output_path
    dataset_labels_type = args.dataset_labels_type
    if not os.path.exists(output_path + '/validation_'+ dataset_labels_type + '_' + patches_type_path):
        os.mkdir(output_path + '/validation_'+ dataset_labels_type + '_' + patches_type_path)
    
    # Make the natural dataset path
    output_filename = None
    if natural:
        val_dir = output_path + '/validation_'+ dataset_labels_type + '_' + patches_type_path + '/natural_validation_ZH_only.tf'
        if not os.path.exists(val_dir):
            if not args.dry_run:
                os.mkdir(val_dir)
        output_filename = val_dir

    # Make the 50/50 dataset path
    else:
        val_dir = output_path + '/validation_'+ dataset_labels_type + '_' + patches_type_path + '/validation1_ZH_only.tf'
        if not os.path.exists(val_dir):
            if not args.dry_run:
                os.mkdir(val_dir)
        output_filename = val_dir

    
    # Save out the metadata of validation data
    validation_array_metadata = validation_array.drop(['ZH', 'VOR', 'DIV', 'labels'])
    validation_array_metadata['ZH_mean'] = ([], mean_validate_ZH)
    validation_array_metadata['ZH_std'] = ([], std_validate_ZH)
    #validation_array_metadata.to_netcdf(output_path + '/validation_'+ dataset_labels_type + '_' + patches_type_path + '/validation1_metadata_ZH_only.nc')
    val_fname = output_path + f'/validation_{dataset_labels_type}_{patches_type_path}/validation1_metadata_ZH_only.nc'
    if not args.dry_run:
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
    ds_validate.save(output_filename)
    validation_array.close()

    return


def create_dataset(args, all_patchfiles, dsettype='train', vertical_levels=[1,2,3,4,5,6,7,8,9,10,11,12], years=['2013'], tfweights=[0.9, 0.1]):
    '''
    Create Tensorflow Datasets from the data. Save subsets of the data as either training, validation, or testing
    
    :param all_patchfiles: list of strings for the files containing the patches
    :param dsettype: string for the dataset type one of: train, val, or test
    :param vertical_levels: list of ints for the vertical levels to select
    :param years: list of strings for the years to use for the dataset
    '''
    #yrs_pattern = "(2013|2014|2015|2016|2017|2018|2019)"
    yrs_joined = "|".join(years)
    yrs_pattern = f"({yrs_joined})"
    patches = [patchfile for patchfile in all_patchfiles if not re.search(yrs_pattern, patchfile) is None]

    patches_notor = [fname for fname in patches if 'nontor' in fname]
    patches_tor = [fname for fname in patches if '_tor_' in fname]
    n_ntfiles = len(patches_notor)
    n_tfiles = len(patches_tor)
    n_files = n_ntfiles + n_tfiles
    print(f"[{dsettype}] {n_ntfiles} nontor files ({n_ntfiles / n_files})")
    print(f"[{dsettype}] {n_tfiles} tor files ({n_tfiles / n_files})")

    # Load data, combine into one array, and extract desired vertical levels
    print(f'Opening {dsettype} data') #print('Opening Training Data')
    print(years)
    data_array = xr.open_mfdataset(patches, concat_dim='patch', combine='nested', 
                                   parallel=True, engine='netcdf4') #autoclose=True
    print(data_array)
    data_array = data_array.sel(z=vertical_levels)
    print(f"{dsettype} Array Number of Gigabytes:", data_array.nbytes / 1e9)

    # Identify storm type from file name
    dataset_patches_type = args.dataset_patches_type
    if 's' in dataset_patches_type and 't' in dataset_patches_type and 'n' in dataset_patches_type:
        patches_type_path = 'nontor_tor_sigtor'
    elif 't' in dataset_patches_type and 's' in dataset_patches_type:
        patches_type_path = 'tor_sigtor'
    elif 't' in dataset_patches_type and 'n' in dataset_patches_type:
        patches_type_path = 'nontor_tor'
    elif 't' in dataset_patches_type:
        patches_type_path = 'tor'
    else:
        print(f'dataset_patches_type = {dataset_patches_type}')
        raise NameError('dataset_patches_type invalid')
    
    # Calculate or load mean and std of training set
    dataset_labels_type = args.dataset_labels_type
    output_path = args.output_path
    meta_fname = os.path.join(output_path, f'train_{dataset_labels_type}_{patches_type_path}', f'train_metadata_ZH_only.nc')

    training_array_metadata = None
    if dsettype == 'train' or not os.path.exists(meta_fname):
        print(f"Calculating Training Metadata") 
        mean_ZH = np.nanmean(data_array.ZH.values)
        std_ZH = np.nanstd(data_array.ZH.values)

        # Save metadata
        print(f"Saving Training Metadata {meta_fname} [dry_run={args.dry_run}]")
        training_array_metadata = data_array.drop(['ZH', 'VOR', 'DIV', 'labels'])
        training_array_metadata['ZH_mean'] = ([], mean_ZH)
        training_array_metadata['ZH_std'] = ([], std_ZH)
        if not args.dry_run:
            training_array_metadata.to_netcdf(meta_fname)
        training_array_metadata.close()
    else:
        # load metadata with the mean and std of the train set
        print(f"Loading mean and std of train set {meta_fname}")
        training_array_metadata = xr.load_dataset(meta_fname)
        
        mean_ZH = training_array_metadata.ZH_mean#.values
        std_ZH = training_array_metadata.ZH_std#.values

    # Make the path to save the output files
    # Make sure the directory structure exists
    #fbasename = output_path + '/training_'+ dataset_labels_type + '_' + patches_type_path
    fdirname = os.path.join(output_path, f'{dsettype}_{dataset_labels_type}_{patches_type_path}')
    '''
    if not os.path.exists(fdirname):
        print(f"Make dir {fdirname} [dry_run={args.dry_run}]")
        if not args.dry_run:
            os.mkdir(fdirname)
    '''
    fname = os.path.join(fdirname, f'{dsettype}_ZH_only.tf')
    if os.path.exists(output_path) and not os.path.exists(fname): #fdirname + '/training_ZH_only.tf'
        print(f"Make dir {fname} [dry_run={args.dry_run}]")
        if not args.dry_run:
            #os.mkdir(output_path + '/training_'+ dataset_labels_type + '_' + patches_type_path + '/training_ZH_only.tf')
            #os.mkdir(fname)
            os.makedirs(fname)


    # Normalize the arrays to a mean of 0 and std of 1
    data_array.ZH.values = ((data_array.ZH - mean_ZH) / std_ZH).compute()

    # Make the input data for training---combine all variable fields
    X = data_array.ZH.values
    print(f"Making X {dsettype} shape={X.shape}")

    # Make the labels for training, given the label type
    print(f"Making Y {dsettype}")
    labels = np.where(data_array.labels.values > 0, 1, 0)
    print(f"             labels shape={labels.shape}")
    n_tors = np.sum(labels)
    n_notors = labels.size - n_tors
    print(f"number of non tors: {n_notors} ({n_notors/labels.size}%)")
    print(f"number of tors: {n_tors} ({n_tors/labels.size}%)")

    #ds_storm_types = []
    #for storm_type in labels.unique():
    #    pass
    
    # Vector of int labels
    if dataset_labels_type == 'int':
        Y = labels.reshape(labels.shape + (1,))
    # Onehot encodings of the labels
    else:
        #labels = np.where(data_array.labels.values > 0, 1, 0)
        Y = keras.utils.to_categorical(labels, num_classes=2)
    print(f"                  Y shape={Y.shape}")

    print("Making Tensorflow dataset")
    #ds = tf.data.Dataset.sample_from_datasets(
    #[ds_notor, ds_tor], weights=tfweights)
    #p = .5
    #q = 1 - p
    #resampled_ds = tf.data.Dataset.sample_from_datasets([pos_ds, neg_ds], weights=[p. q])
    #resampled_ds = resampled_ds.batch(BATCH_SIZE).prefetch(2)
    ds = tf.data.Dataset.from_tensor_slices((X, Y)) #ds_train

    # Shuffle and batch the data
    ds = ds.shuffle(X.shape[0], seed=24).batch(args.batch_size)
    print(f"{dsettype} Dataset Elem Spec:", ds.element_spec)
    print(ds)

    # Save tf dataset
    #tf.data.experimental.save(ds_train, output_path + '/training_'+ dataset_labels_type + '_' + patches_type_path + '/training_ZH_only.tf')
    #ds_fname = output_path + f'/{dsettype}_{dataset_labels_type}_{patches_type_path}/{dsettype}_ZH_only.tf'
    ds_fname = os.path.join(output_path, f'{dsettype}_{dataset_labels_type}_{patches_type_path}', f'{dsettype}_ZH_only.tf')
    print(f"Saving tf.Dataset {ds_fname} [dry_run={args.dry_run}]")
    if not args.dry_run:
        ds.save(ds_fname)
    data_array.close()

    return ds


if __name__ == "__main__":
    #get the inputs from the .sh file
    args = get_arguments()  

    if "CUDA_VISIBLE_DEVICES" in os.environ.keys():
        # Fetch list of allocated logical GPUs; numbered 0, 1, …
        devices = tf.config.get_visible_devices('GPU')
        ndevices = len(devices)
        devices_logical = tf.config.list_logical_devices('GPU')
        print(f'We have {ndevices} GPUs. Logical devices {len(devices_logical)} {devices_logical}\n')

        # Set memory growth for each
        try:
            for device in devices:
                tf.config.experimental.set_memory_growth(device, True)
        except Exception as err:
            print(err)
    else:
        # No allocated GPUs: do not delete this case!                                                                	 
        tf.config.set_visible_devices([], 'GPU')

    print("GPUs Available: ", tf.config.list_physical_devices('GPU'))

    # Load in all the directories that will be included in the tensorflow datasets
    print("Loading in all directories")
    #all_patches_dirs = glob.glob(input_xarray_path + '/2013/*') + glob.glob(input_xarray_path + '/2016/*')
    all_patchfiles = glob.glob(args.input_xarray_path + '/*/*/*.nc')
    all_patchfiles.sort()

    # [1,2,3,4,5,6,7,8,9,10,11,12]
    vertical_levels = np.arange(1, 13, dtype=int).tolist()

    # Data set years list
    print(args.years)
    all_years = ['2013', '2014', '2015', '2016', '2017', '2018'] # TODO: 2019
    nfolds = len(all_years)
    # rotation index
    rot = 0 
    train_end = (nfolds - 2 + rot) % nfolds
    val_sel = (nfolds - 2 + rot) % nfolds
    test_sel = (nfolds - 1 + rot) % nfolds
    trainset_yrs = all_years[:train_end]
    valset_yrs = all_years[val_sel]
    testset_yrs = all_years[test_sel]

    # Create the training dataset
    ds_train = create_dataset(args, all_patchfiles, dsettype='train', vertical_levels=vertical_levels, years=trainset_yrs)

    # Create the validation dataset
    ds_val = create_dataset(args, all_patchfiles, dsettype='val', vertical_levels=vertical_levels, years=valset_yrs)

    # Create the test dataset
    ds_test = create_dataset(args, all_patchfiles, dsettype='test', vertical_levels=vertical_levels, years=testset_yrs)

    #training_metadata = xr.open_dataset(args.training_data_metadata_path)
    #mean_train_ZH = float(training_metadata.ZH_mean.values)
    #std_train_ZH = float(training_metadata.ZH_std.values)
    #training_metadata.close()
    #del training_metadata

    # Create the 50/50 validation dataset
    # natural = False
    #>save_validation_dataset(validation_patches, natural, mean_train_ZH, std_train_ZH)

    # Create the natural validation dataset
    #:natural = True
    #:save_validation_dataset(natural_validation_patches, natural, mean_train_ZH, std_train_ZH)


