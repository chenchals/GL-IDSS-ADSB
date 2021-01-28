# GL-IDSS-ADSB
Applied Data Science Bootcamp: MIT-IDSS from Great Learning October 2020-cohort

# Brain Tumor Classification 

Build a CNN model for classifying MRI scans as belonging to one of the following classes - glioma_tumor, meningioma_tumor, pituitary_tumor or no_tumor.

To run the notebook:
1. Ensure you have created proper data directory locations in the notebook
2. Ensure you have created HDF5 files for Trainign and Testing data (See BTCHelper.DataUtils)
3. Create output directories output/figures, output/models and point variables to correct locations
4. Running on Google colab with TPU is much faster than on GPUs


# Other comments and discussion form Kaggle:
This dataset may be similar to the one on Kaggle:
https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri

This discussion link talks about source of the dataset:
https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri/discussion/154260

Comments on notebooks:

1. https://www.kaggle.com/taha07/brain-tumor-classification-97-5
This notebook claims a 97.5%, upon closely following the data, you can see that
the x_test, y_test is used as validation data for training as well. 
Hence, the final results are test data appears as  07.5%, which imho is incorrect.

