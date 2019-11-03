# caribbean_disaster_risk

### workflow

use polygons to crop images from master tifs, then rotate them to horizontal/vertical
does not resize, usues small buffer to increase polygon size

<pre>
01_data_prep_rasterio.py
</pre>

Needs QC before use
earlier version, crops only
<pre>
01_data_prep_rasterio.py
</pre>

crops, resizes, and pads out images using BORDER_REFLECT_101
<pre>
02_data_crop_concat_20191025.ipynb
</pre>

crops, resizes, pad using BORDER
***may need debugging
<pre>
02_1_data_crop_concat_20191027.ipynb
</pre>

submission 2
<pre>
03_1_fastai_train_resnet_20191026.ipynb
</pre>

qc of results
<pre>
03_1_1_result_qc.ipynb
</pre>

sharpen blurred images
<pre>
04_image_reprocessing.ipynb
</pre>

as 03_1_fastai_train_resnet_20191026.ipynb but with sharpened images
slightly worse result
<pre>
05_1_fastai_train_resnet_20191027.ipynb
</pre>

augmented to even out class distribution
<pre>
04_1_image_repro_even_up_class_numbers.ipynb
</pre>

NB this is run with data_dir/f'train/rotated/clipped/256' and data_dir/f'train/rotated/wrap/256' data paddings

TODO - combine all the augmented and different padded data for a simgle train and test ds




