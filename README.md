# caribbean_disaster_risk

workflow

#use polygons to crop images from master tifs, then rotate them to horizontal/vertical
#does not resize, usues small buffer to increase polygon size
01_data_prep_rasterio.py

#Needs QC before use
#earlier version, crops only
01_data_prep_rasterio.py

#crops, resizes, and pads out images using BORDER_REFLECT_101
02_data_crop_concat_20191025.ipynb

#crops, resizes, pad using BORDER
***may need debugging
02_1_data_crop_concat_20191027.ipynb

#submission 2
03_1_fastai_train_resnet_20191026.ipynb

#qc of results
03_1_1_result_qc.ipynb

#sharpen blurred images
04_image_reprocessing.ipynb

#as 03_1_fastai_train_resnet_20191026.ipynb but with sharpened images
#slightly worse result
05_1_fastai_train_resnet_20191027.ipynb

#augmented to even out class distribution
04_1_image_repro_even_up_class_numbers.ipynb
NB this is run with data_dir/f'train/rotated/clipped/256' and data_dir/f'train/rotated/wrap/256' data paddings
TODO - combine all the augmented and different padded data for a simgle train and test ds




