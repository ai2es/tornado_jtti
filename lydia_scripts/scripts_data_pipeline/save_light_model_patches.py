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
        
    INPUT_ARG_PARSER.add_argument(
        '--dry_run', action='store_true',
        help="For testing to verify input and output paths")
        
    #Pull out all of the input strings from the .sh file
    args = INPUT_ARG_PARSER.parse_args()
    #Index primer indicates the day of data that we are looking at in this particuar run of this code
    global input_xarray_path
    input_xarray_path = getattr(args, 'input_xarray_path')
    global output_path
    output_path = getattr(args, 'output_path')
    global dry_run
    dry_run = getattr(args, 'dry_run')


def transfer_patch(patches_file, vertical_levels=[1,2,3,4,5,6,7,8,9,10,11,12]):


    path_length = len(input_xarray_path)  

    # Make the output directory structure
    outpath = output_path + patches_file[path_length:path_length+4]
    '''
    print(f"Making dir outpath={outpath} [dry_run={dry_run}]")
    if not os.path.exists(outpath):
        try:
            #os.mkdir(output_path + patches_file[path_length:path_length+4])
            if not dry_run:
                print(f"Making directory {outpath} [dry_run={dry_run}]")
                os.mkdir(outpath)
        except Exception as err:
            print(err)
    '''

    #outpath2 = output_path + '/' + patches_file[path_length:path_length+4] + '/' + patches_file[path_length+5:path_length+13]
    outpath2 = os.path.join(output_path, patches_file[path_length:path_length+4], patches_file[path_length+5:path_length+13])
    print(f"Making dir outpath2={outpath2} [dry_run={dry_run}]")
    if not os.path.exists(outpath2):
        try:
            #os.mkdir(output_path + '/' + patches_file[path_length:path_length+4] + '/' + patches_file[path_length+5:path_length+13])
            if not dry_run:
                print(f"Making directory {outpath2} [dry_run={dry_run}]")
                os.makedirs(outpath2) #os.mkdir(outpath2)
        except Exception as err:
            print(err)
    
    # We want to save out the patches based on their file type
    if '_nontor_' in patches_file:
        
        # Transfer only if the "light" file doesn't exists
        ds_fname = output_path + patches_file[path_length:]
        print(f"nontor outfile={ds_fname}\n")
        if not os.path.exists(ds_fname):

            # Keep only convective patches
            ds = xr.load_dataset(patches_file) #xr.open_dataset(patches_file)
            ds = ds.where(ds.n_convective_pixels > 0).dropna(dim='patch')
            
            # Select 50 random patches from all the convective patches
            random_idxs = list(np.random.randint(low=0, high=ds.patch.values.shape[0], size=50))
            ds = ds.isel(patch=random_idxs)

            # Keep only the 12 desired vertical levels (each km agl)
            #ds = ds.sel(z=[1,2,3,4,5,6,7,8,9,10,11,12])
            ds = ds.sel(z=vertical_levels)

            # Save and close the dataset
            #ds_fname = output_path + patches_file[path_length:]
            if dry_run: 
                print(f"[nontor] ds to save = {ds_fname} [dry_run={dry_run}]")
            else:
                print(f"[nontor] Saving xr.DS {ds_fname} [dry_run={dry_run}]")
                ds.to_netcdf(ds_fname)
            #ds.close()
            del ds
            return
        
    # Transfer the tornadic patches
    elif 'tor_' in patches_file:
        
        # If the "light" file doesn't already exist
        ds_fname = output_path + patches_file[path_length:]
        print(f"tor outfile={ds_fname}\n")
        if not os.path.exists(ds_fname):

            # Open the dataset
            ds = xr.load_dataset(patches_file) #xr.open_dataset(patches_file)
            
            # Select a random 50 patches from the hour of data
            random_idxs = list(np.random.randint(low=0, high=ds.patch.values.shape[0], size=50))
            ds = ds.isel(patch=random_idxs)

            # Select only the vertical levels we want for training
            #ds = ds.sel(z=[1,2,3,4,5,6,7,8,9,10,11,12])
            ds = ds.sel(z=vertical_levels)

            # Save and close the dataset
            if dry_run: 
                print(f"[tor] ds to save = {ds_fname} [dry_run={dry_run}]")
            else:
                print(f"[tor] Saving xr.DS {ds_fname} [dry_run={dry_run}]")
                #ds.to_netcdf(output_path + patches_file[path_length:])
                ds.to_netcdf(ds_fname)
            #ds.close()
            del ds
            
            return
        
    # Transfer the validation data
    elif 'validation' in patches_file:

        # If this is a natural distribution file, we want to keep all the patches
        if 'natural' in patches_file:

            # Transfer the data if it hasn't been done already
            ds_fname = output_path + patches_file[path_length:]
            print(f"val natural outfile={ds_fname}\n")
            if not os.path.exists(ds_fname):

                # Open the dataset
                ds = xr.load_dataset(patches_file) #xr.open_dataset(patches_file)

                # Select the vertical levels we want for the ML model
                #ds = ds.sel(z=[1,2,3,4,5,6,7,8,9,10,11,12])
                ds = ds.sel(z=vertical_levels)

                # Save and close dataset
                if dry_run: 
                    print(f"[val natural] ds to save = {ds_fname} [dry_run={dry_run}]")
                else:
                    print(f"[val natural] Saving xr.DS {ds_fname} [dry_run={dry_run}]")
                    #ds.to_netcdf(output_path + patches_file[path_length:])
                    ds.to_netcdf(ds_fname)
                #ds.close()
                del ds
                
                return

        # This dataset is a 50/50 validation file
        else:

            # Transfer if the data hasn't been transfered already
            ds_fname = output_path + patches_file[path_length:]
            print(f"val 50/50 outfile={ds_fname}\n")
            if not os.path.exists(ds_fname):

                # Open the file
                ds = xr.open_dataset(patches_file)

                # Select the vertical levels we want
                #ds = ds.sel(z=[1,2,3,4,5,6,7,8,9,10,11,12])
                ds = ds.sel(z=vertical_levels)

                # Save out the data
                if dry_run:
                    print(f"[val 50/50] ds to save = {ds_fname} [dry_run={dry_run}]")
                else:
                    print(f"[val 50/50] Saving xr.DS {ds_fname} [dry_run={dry_run}]")
                    #ds.to_netcdf(output_path + patches_file[path_length:])
                    ds.to_netcdf(ds_fname)
                
                # If there is not also a super light version of this validation file, make one
                ds_light_dir = output_path + 'light_' + patches_file[path_length + 1:]
                if not os.path.exists(ds_light_dir):
                    
                    # Select 50 random patches from the file
                    random_idxs = list(np.random.randint(low=0, high=ds.patch.values.shape[0], size=50))
                    ds = ds.isel(patch=random_idxs)
                    
                    # Save the dataset
                    light_fname = output_path + patches_file[path_length:path_length+13] + '/light_' + patches_file[path_length+14:]
                    if dry_run:
                        print(f"[val 50/50 light] ds to save = {light_fname} [dry_run={dry_run}]")
                    else:
                        print(f"[val 50/50 light] Saving xr.DS {light_fname} [dry_run={dry_run}]")
                        ds.to_netcdf(light_fname)
                
                # Close the dataset
                #ds.close()
                del ds
                return
                    
            # If the validation file has been transfered, but a super light validation file hasn't been made, make one
            elif not os.path.exists(output_path + 'light_' + patches_file[path_length + 1:]):
            
                # Open the dataset
                ds = xr.open_dataset(patches_file)

                # Select the vertical levels for this Ml model
                #ds = ds.sel(z=[1,2,3,4,5,6,7,8,9,10,11,12])
                ds = ds.sel(z=vertical_levels)
                
                # Select 50 random patches from this time
                random_idxs = list(np.random.randint(low=0, high=ds.patch.values.shape[0], size=50))
                ds = ds.isel(patch=random_idxs)
                
                # Save and close the dataset
                light_fname = output_path + patches_file[path_length:path_length+13] + '/light_' + patches_file[path_length+14:]
                if dry_run:
                    print(f"[val 50/50 light2] ds to save = {light_fname} [dry_run={dry_run}]")
                else:
                    print(f"[val 50/50 light2] Saving xr.DS {light_fname} [dry_run={dry_run}]")
                    ds.to_netcdf(light_fname)
                #ds.close()
                del ds
                return

    else:
        print('This patch file did not follow the file naming conventions and hasn\'t been transfered:')
        print(patches_file)
        return
    
        
      
def main():

    # Get command line arguments
    get_arguments()  
      
    # Find the filepaths for all the patches we want to reduce
    all_patches = glob.glob(input_xarray_path + '*/*/*.nc')
    all_patches.sort()
    print(input_xarray_path + '*/*/*.nc')
    print(len(all_patches), all_patches)
    print(" ")

    # Transfer patches in parallel
    with mp.Pool(processes=20) as p:
        #tqdm.tqdm(p.map(transfer_patch, all_patches), total=len(all_patches))
        p.map(transfer_patch, all_patches)



if __name__ == "__main__":
    main()



