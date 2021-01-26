# GL-IDSS-ADSB
Applied Data Science Bootcamp: MIT-IDSS from Great Learning October 2020-cohort

# Dataset Folder
1. This folder shall contain the the Training and Testing datasets as original folders or softlinks to these folders
OR
2. This folder shall contain the HDF5 files for Training_150.h5 / Testing _150.h5 files where 
each file is a HDF5 file of training or testing images created by using 
BTCHelper.DataUtils.convertToHdf5() function

# create softlinks (mac/linux)
In main project folder [GL-IDSS-ADSB]:
[GL-IDSS-ADSB]$ mkdir DatasetBrainTumor
[GL-IDSS-ADSB]$ cd DatasetBrainTumor
[DatasetBrainTumor]$

#1 for training and testing directories
DatasetBrainTumor$ ln -s [location-of-DataSetBrainTumor/Training] Training
DatasetBrainTumor$ ln -s [location-of-DataSetBrainTumor/Testing] Testing

#2 for HDF5 files
[DatasetBrainTumor]$ ln -s [location-of-DataSetBrainTumor/Training_256.h5] Training_256.h5
[DatasetBrainTumor]$ ln -s [location-of-DataSetBrainTumor/Testing_256.h5] Testing_256.h5

# Output directory:
is automatically created in the notebook location by code
