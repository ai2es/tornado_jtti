# Steps to process the GridRad data for ML 

This guide is to help others run the code originally written by Ryan Lagerquist during his time at OU (GewitterGefahr). There are a number of steps to follow here, but they are named in their order. If you have questions, please reach out to me (Randy Chase) and we can see if we can solve your issue. 

## Pre-Reqs 

In order to run the scripts in this directory, you need to install Ryan's github package. Now his original package was not intended to run on the size of data we now have. As part of the begining of my postdoc, I went in a tried my best to adapt his code with fixes. Depending on your time of running this, there are likely new bugs that have shown up. Anyways, go install my fork of it 

``` git clone https://github.com/dopplerchase/GewitterGefahr.git``` 

please know where you are installing this repo, you will need it to tell the scripts where your gewitter is. Okay, now that you have the code, install a new env with the required python stuff. 

``` cd GewitterGefahr``` 

``` conda env create -f environment.yml ``` 

This will make an environment named gewitter. Activate it and install the repo so that file paths and things are built. 

``` pip install . ``` 

Note, if you run the tests, some will probably fail with my new changes. I haven't gone in and fixed these. Now you have gewitter, so you can continue on to running the processing scripts. 

## Step 1: Decompress and QC GridRad 

The GridRad severe data are kept in sparse arrays, meaning that we only store 'real' observations. This is done to save disk space, no need to save all those 0.0 dBZ if we don't need to. The only issue with this is that Ryan's code expects the data to be gridded and QC'ed already. Thus, I have added a function called `gridrad_from_sparse_to_grid.py` which opens the raw GridRad data, QC's it and then saves it out to a netCDF file. To accomplish this step, run the script named Step_01.sh

Things you will need to update in the script: 

Line 7: ``#SBATCH --chdir=/ourdisk/hpc/ai2es/randychase/GewitterGefahr/gewittergefahr/scripts/`` 
- this line points to my gewitter, please change this to yours 

Line 8: ``#SBATCH --job-name="Step1_2013"`` 
- Depending on the year you are running make sure this matches line 17. 

Line 9: ``#SBATCH --mail-user=randychase@ou.edu`` 
- change to your email. 

Line 12-13: ``#SBATCH --output=/home/randychase/slurmouts/R-%x.%j.out`` 
- change to your dir 

Line 14: ``#SBATCH --array=2%4``

- this array line will run jobs in parallel (different nodes will get different days to process). This takes a bit of trial and error, but since there is so much reading and writing with this script, don't change the 4 (this means there will be 4 jobs running at once). 
- Depending on the year, you *do* need to find out how many days are in the file on line 17. This will determine how many array jobs you need to do. For example, if you have 100 days in 2013, then the array line would be ``#SBATCH --array=0-99%4`` which will process 4 days at a time within 2013. 

Line 17: ``mapfile -t SPC_DATE_STRINGS < /ourdisk/hpc/ai2es/tornado/tornado_jtti/scripts/process_gridrad_scripts/orderofprocessing_files/orderofprocessing_paths_2013.txt`` 
-  this line has some text files with the paths to each day to allow for it to autodetemine a bunch of files to process. You will need to update this to the year you want to run. 

Ok. Now that you made all the changes, it should run. It will place the netcdf files where you tell it on line 35. 


## Step 2: Convert to MYRORSS

Originally, Ryan had is code running on a radar dataset called MYRORSS, which I believe is the legacy MRMS product. Thus, for his datapipeline to work, we need to now conver the new gridrad files we just made to this new format. 

There are 3 seperate scripts in Step2 (a,b and c). These scripts are all in the same step because they do not rely on each other to run. Meaning you could run them at the same time if you wanted to, but keep in mind the reading/writing issues of multiple jobs all reading/writing multiple files. 

#### Step_02a.sh 

- this script calcualtes the column max reflectivity for the data and saves it in the myrorss format. This field is used for tracking storms. 

#### Step_02b.sh 

- this script calculates the 40 dBZ echotop height and saves it to the myrorss format. this field is used for tracking storms. 

#### Step_02c.sh

- this script determines which echoes are 'convective'. This field is used for tracking storms. 

Please be mindful and change similar lines to Step_01.sh 

