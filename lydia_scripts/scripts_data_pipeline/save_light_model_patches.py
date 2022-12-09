import xarray as xr
import numpy as np
import glob
import argparse
import os
import multiprocessing as mp
import tqdm



def get_arguments():

    #Define the strings that explain what each input variable means
    INPUT_XARRAY_DIR_HELP_STRING = 'The directory where the xarray patch files are stored. This directory should start from the root directory \'/\'.'
    OUTPUT_PATH_HELP_STRING = 'The directory where the 3D light datasets will be saved. This directory should start from the root directory \'/\'.'

    #Tell python what arguments to expect from the .sh file
    INPUT_ARG_PARSER = argparse.ArgumentParser()
        
    INPUT_ARG_PARSER.add_argument(
        '--input_xarray_path', type=str, required=True,
        help=INPUT_XARRAY_DIR_HELP_STRING)
        
    INPUT_ARG_PARSER.add_argument(
        '--output_path', type=str, required=True,
        help=OUTPUT_PATH_HELP_STRING)
        
    #Pull out all of the input strings from the .sh file
    args = INPUT_ARG_PARSER.parse_args()
    #Index primer indicates the day of data that we are looking at in this particuar run of this code
    global input_xarray_path
    input_xarray_path = getattr(args, 'input_xarray_path')
    global output_path
    output_path = getattr(args, 'output_path')


def transfer_patch(patches_file):


    path_length = len(input_xarray_path)  

    # Make the output directory structure
    if not os.path.exists(output_path + patches_file[path_length:path_length+4]):
        try:
            os.mkdir(output_path + patches_file[path_length:path_length+4])
        except:
            pass
    if not os.path.exists(output_path + '/' + patches_file[path_length:path_length+4] + '/' + patches_file[path_length+5:path_length+13]):
        try:
            os.mkdir(output_path + '/' + patches_file[path_length:path_length+4] + '/' + patches_file[path_length+5:path_length+13])
        except:
            pass
            
    # We want to save out the patches based on their file type
    if '_nontor_' in patches_file:
        
        # Transfer only if the "light" file doesn't exists
        if not os.path.exists(output_path + patches_file[path_length:]):

            # Keep only convective patches
            ds = xr.open_dataset(patches_file)
            ds = ds.where(ds.n_convective_pixels > 0).dropna(dim='patch')
            
            # Select 50 random patches from all the convective patches
            random_idxs = list(np.random.randint(low=0, high=ds.patch.values.shape[0], size=50))
            ds = ds.isel(patch=random_idxs)

            # Keep only the 12 desired vertical levels (each km agl)
            ds = ds.sel(z=[1,2,3,4,5,6,7,8,9,10,11,12])

            # Save and close the dataset
            ds.to_netcdf(output_path + patches_file[path_length:])
            ds.close()
            del ds
            return
        
    # Transfer the tornadic patches
    elif 'tor_' in patches_file:
        
        # If the "light" file doesn't already exist
        if not os.path.exists(output_path + patches_file[path_length:]):

            # Open the dataset
            ds = xr.open_dataset(patches_file)
            
            # Select a random 50 patches from the hour of data
            random_idxs = list(np.random.randint(low=0, high=ds.patch.values.shape[0], size=50))
            ds = ds.isel(patch=random_idxs)

            # Select only the vertical levels we want for training
            ds = ds.sel(z=[1,2,3,4,5,6,7,8,9,10,11,12])

            # Save and close the dataset
            ds.to_netcdf(output_path + patches_file[path_length:])
            ds.close()
            del ds
            
            return
        
    # Transfer the validation data
    elif 'validation' in patches_file:

        # If this is a natural distribution file, we want to keep all the patches
        if 'natural' in patches_file:

            # Transfer the data if it hasn't been done already
            if not os.path.exists(output_path + patches_file[path_length:]):

                # Open the dataset
                ds = xr.open_dataset(patches_file)

                # Select the vertical levels we want for the ML model
                ds = ds.sel(z=[1,2,3,4,5,6,7,8,9,10,11,12])

                # Save and close dataset
                ds.to_netcdf(output_path + patches_file[path_length:])
                ds.close()
                del ds
                
                return

        # This dataset is a 50/50 validation file
        else:

            # Transfer if the data hasn't been transfered already
            if not os.path.exists(output_path + patches_file[path_length:]):

                # Open the file
                ds = xr.open_dataset(patches_file)

                # Select the vertical levels we want
                ds = ds.sel(z=[1,2,3,4,5,6,7,8,9,10,11,12])

                # Save out the data
                ds.to_netcdf(output_path + patches_file[path_length:])
                
                # If there is not also a super light version of this validation file, make one
                if not os.path.exists(output_path + 'light_' + patches_file[path_length + 1:]):
                    
                    # Select 50 random patches from the file
                    random_idxs = list(np.random.randint(low=0, high=ds.patch.values.shape[0], size=50))
                    ds = ds.isel(patch=random_idxs)
                    
                    # Save the dataset
                    ds.to_netcdf(output_path + patches_file[path_length:path_length+13] + '/light_' + patches_file[path_length+14:])
                
                # Close the dataset
                ds.close()
                del ds
                return
                    
            # If the validation file has been transfered, but a super light validation file hasn't been made, make one
            elif not os.path.exists(output_path + 'light_' + patches_file[path_length + 1:]):
            
                # Open the dataset
                ds = xr.open_dataset(patches_file)

                # Select the vertical levels for this Ml model
                ds = ds.sel(z=[1,2,3,4,5,6,7,8,9,10,11,12])               
                
                # Select 50 random patches from this time
                random_idxs = list(np.random.randint(low=0, high=ds.patch.values.shape[0], size=50))
                ds = ds.isel(patch=random_idxs)
                
                # Save and close the dataset
                ds.to_netcdf(output_path + patches_file[path_length:path_length+13] + '/light_' + patches_file[path_length+14:])
                ds.close()
                del ds
                return

    else:
        print('This patch file did not follow the file naming conventions and hasn\'t been transfered:')
        print(patches_file)
        return
    
        
      
def main():

    #get the inputs from the .sh file
    get_arguments()  
      
    # Find the filepaths for all the patches we want to reduce
    all_patches = glob.glob(input_xarray_path + '*/*/*.nc')
    all_patches.sort()

    # Transfer patches in parallel
    with mp.Pool(processes=20) as p:
        tqdm.tqdm(p.map(transfer_patch, all_patches), total=len(all_patches))



if __name__ == "__main__":
    main()



