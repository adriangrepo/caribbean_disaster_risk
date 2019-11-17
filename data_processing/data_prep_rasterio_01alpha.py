#!/usr/bin/env python
# coding: utf-8

# # Data prep
# 
# 

# # Pre-Processing
# 
# use conda env 'solaris'
# 
#crops to polygon but no post crop rotation, keeps white background



import time
import solaris as sol
import numpy as np
import geopandas as gpd
from matplotlib import pyplot as plt
from pathlib import Path
import rasterio
import os
import sys
import skimage
from tqdm import tqdm
import pandas as pd

import fiona
import numpy as np

from rasterstats.io import Raster
from PIL import Image

from rio_tiler import main as rt_main
# import mercantile
import rasterio as rio
from rasterio.plot import show
from rasterio.mask import mask
from shapely.geometry import mapping
from rasterio.transform import from_bounds
from shapely.geometry import Polygon as shapely_polygon
from shapely.ops import cascaded_union
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

tile_size = 256
BUFFER = 0.2

colombia_rural = Path('data/stac/colombia/borde_rural')
colombia_soacha = Path('data/stac/colombia/borde_soacha')

guatemala_mixco1 = Path('data/stac/guatemala/mixco_1_and_ebenezer')
guatemala_mixco3 = Path('data/stac/guatemala/mixco_3')

st_lucia_castries = Path('data/stac/st_lucia/castries')
st_lucia_dennery = Path('data/stac/st_lucia/dennery')
st_lucia_gros_islet = Path('data/stac/st_lucia/gros_islet')

data_dir = Path('data')
data_dir.mkdir(exist_ok=True)

path_list= [colombia_rural, colombia_soacha, guatemala_mixco1, guatemala_mixco3,
         st_lucia_castries, st_lucia_dennery, st_lucia_gros_islet]
json_list=['train-borde_rural.geojson','train-borde_soacha.geojson','train-mixco_1_and_ebenezer.geojson','train-mixco_3.geojson',
          'train-castries.geojson','train-dennery.geojson','train-gros_islet.geojson']
country_list=['colombia','colombia','guatemala','guatemala','st_lucia','st_lucia','st_lucia']
region_list=['borde_rural','borde_soacha','mixco_1_and_ebenezer','mixco_3','castries','dennery','gros_islet']


def mask_raster(df, tif_filename, country, region, save_path, buffer=2):
    with rasterio.open(tif_filename) as src:
        for index, row in df.iterrows():
            id = row["id"]
            poly = df.iloc[[index]]['geometry']
            poly = poly.buffer(distance=buffer, cap_style=2, join_style=2)
            out_image, out_transform = rasterio.mask.mask(src, poly, crop=True, nodata= 255)
            out_meta = src.meta
            out_meta.update({"driver": "GTiff",
                             "height": out_image.shape[1],
                             "width": out_image.shape[2],
                             "transform": out_transform})
            path_out = data_dir / f'{country}_{region}/cropped/{save_path}/{id}.tif'
            with rasterio.open(path_out, "w", **out_meta) as dest:
                dest.write(out_image)

def mask_images(df, df_test, data_tif, test_exists, country, region, buffer):
    mask_raster(df, data_tif, country, region, save_path='train', buffer=buffer)

    if test_exists:
        mask_raster(df_test, data_tif, country, region, save_path='test', buffer=buffer)

def format_all(data_path, country, region):

    DATASET = f'{country}_{region}'

    #no test data for these
    test_exists=True
    if region in ['castries','gros_islet']:
        test_exists=False

    DATA_TIFF = data_path/f'{region}_ortho-cog.tif'
    TRAIN_JSON = f'train-{region}.geojson'
    TRAIN_TRN_JSON = f'train-{region}_trn.geojson'
    TRAIN_VAL_JSON = f'train-{region}_val.geojson'
    TEST_JSON = f'test-{region}.geojson'
    PNG_THUMBNAL = data_path/f'{region}_ortho-cog-thumbnail.png'

    # load geojson colombia rural training data

    label_df = gpd.read_file(data_path/TRAIN_JSON)
    label_df = label_df[label_df['geometry'].isna() != True] # remove empty rows
    epsg=str(rasterio.open(DATA_TIFF).meta['crs']).split(':')[1]
    label_df = label_df.to_crs({'init': f'epsg:{epsg}'})


    # load geojson test data
    if test_exists:
        label_df_test = gpd.read_file(data_path/TEST_JSON)
        label_df_test = label_df_test[label_df_test['geometry'].isna() != True] # remove empty rows
        print(label_df_test.crs)
        label_df_test = label_df_test.to_crs({'init': f'epsg:{epsg}'})
        print(label_df_test.crs)
    else:
        label_df_test=None


    # ### RasterIO
    poly_path = data_dir/f'{country}_{region}'
    poly_path.mkdir(exist_ok=True)
    cropped = data_dir/f'{country}_{region}/w_cropped'
    cropped.mkdir(exist_ok=True)
    cropped_trn= data_dir/f'{country}_{region}/w_cropped/train'
    cropped_trn.mkdir(exist_ok=True)
    if test_exists:
        cropped_test= data_dir/f'{country}_{region}/w_cropped/test'
        cropped_test.mkdir(exist_ok=True)
    return label_df, label_df_test, DATA_TIFF, test_exists



def workflow():
    for data_path, country, region in zip(path_list, country_list, region_list):
        print(f'{data_path} {country} {region}')
        start=time.time()
        df, df_test, data_tif, test_exists = format_all(data_path, country, region)
        mask_images(df, df_test, data_tif, test_exists, country, region, buffer=BUFFER)
        end = time.time()
        print(f'elapsed: {end-start}')

if __name__ == "__main__":
    workflow()















