#!/usr/bin/env python
# coding: utf-8

# # Data prep
# 
# 

# # Pre-Processing
# 
# use conda env 'solaris'
# 

#uses transparent mask, rotates image, clips as much empty mask as possible

# for bleeding edge version of solaris:
# !pip install git+https://github.com/CosmiQ/solaris/@dev


from os import listdir
from os.path import isfile, join
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
from shapely.geometry import mapping, MultiPolygon
from rasterio.transform import from_bounds
from rasterio.plot import reshape_as_raster, reshape_as_image
from shapely.geometry import Polygon as shapely_polygon
from shapely.ops import cascaded_union
from shapely import affinity
import cv2

import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

tile_size = 512
BUFFER = 0.1

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

def oddTuples(aTup):
    return aTup[0::2]

def evenTuples(aTup):
    return aTup[1::2]

def plot_poly(x, y, title='polygon'):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y, color='#6699cc', alpha=0.7,
            linewidth=3, solid_capstyle='round', zorder=2)
    ax.set_title(title)
    plt.savefig(title)
    fig.show()

def get_angle(poly):
    '''returns the angle from hx of first line in the polygon'''
    theta = 0
    if len(poly)>0:
        if isinstance(poly.iloc[0], MultiPolygon):
            x,y = poly.convex_hull.exterior.coords.xy
            #c=list(lr.coords)
            poly=cascaded_union(poly)
            #may still be mitipolygon
            if poly.geom_type == 'MultiPolygon':
                # extract polygons out of multipolygon and keep biggest one
                max_area=0
                index=0
                for i,polygon in enumerate(poly):
                    if polygon.area>max_area:
                        max_area=polygon.area
                        index=i
                poly=poly[index]
                c=list(poly.exterior.coords)
        else:
            lr=poly.exterior.values[0]
            c=list(lr.coords)
        #flatten the list of tuples
        c=list(sum(c, ()))
        cx=oddTuples(c)
        cy = evenTuples(c)
        if len(cy)>1 and len(cx)>1:
            dy=cy[1]-cy[0]
            dx=cx[1]-cx[0]
            try:
                theta=np.arctan(dy/dx)
            except ZeroDivisionError as e:
                theta=0
    return np.degrees(theta)

def prep_poly(poly):
    '''returns lists of x, y from polygon for plotting'''
    lr_rot = poly.exterior.values[0]
    c = list(lr_rot.coords)
    c = list(sum(c, ()))
    x = oddTuples(c)
    y = evenTuples(c)
    return x, y

def mask_raster(df, tif_filename, country, region, save_path, buffer=2):
    angles={}
    with rasterio.open(tif_filename) as src:
        for index, row in df.iterrows():
            id = row["id"]
            poly = df.iloc[[index]]['geometry']
            theta=get_angle(poly)
            angles[id]=theta
            poly = poly.buffer(distance=buffer, cap_style=2, join_style=2)

            #can't rotate image before mask as image is 5GB

            #out_image, out_transform = rasterio.mask.mask(src, poly, crop=True, nodata= 255)
            #transparent outside mark
            out_image, out_transform = rasterio.mask.mask(src, poly, crop=True)
            out_meta = src.meta
            out_meta.update({"driver": "GTiff",
                             "height": out_image.shape[1],
                             "width": out_image.shape[2],
                             "transform": out_transform})
            path_out = data_dir / f'{country}_{region}/cropped/{save_path}/{id}.tif'
            with rasterio.open(path_out, "w", **out_meta) as dest:
                dest.write(out_image)

            std_image=reshape_as_image(out_image)
            plt.imshow(std_image)
            #rotated image
            rimage = Image.fromarray(std_image)
            rotated = rimage.rotate(theta*-1)
            rot_out = data_dir / f'{country}_{region}/cropped/{save_path}/rotated/{id}.tif'
            rotated.save(rot_out)

def crop_non_transparent(dir, out_path):
    onlyfiles = [f for f in listdir(dir) if isfile(join(dir, f))]
    for f in onlyfiles:
        assert(os.path.isfile(dir/f'{f}'))
        im = cv2.imread(str(dir/f'{f}'), cv2.IMREAD_UNCHANGED)
        # axis 0 is the row(y) and axis(x) 1 is the column
        y, x = im[:, :, 3].nonzero()  # get the nonzero alpha coordinates
        minx = np.min(x)
        miny = np.min(y)
        maxx = np.max(x)
        maxy = np.max(y)
        cropImg = im[miny:maxy, minx:maxx]
        cv2.imwrite(str(out_path/f'{f}'), cropImg)
        #cv2.imshow("cropped", cropImg)
        #cv2.waitKey(0)

def mask_images(df, df_test, data_tif, test_exists, country, region, buffer):
    mask_raster(df, data_tif, country, region, save_path='train', buffer=buffer)

    if test_exists:
        mask_raster(df_test, data_tif, country, region, save_path='test', buffer=buffer)

def format_all(data_path, country, region):

    DATASET = f'{country}_{region}'

    #no test data for these
    test_exists=True
    if region.strip() in ['castries','gros_islet']:
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
        print(f'test data exists for {country}_{region}, new CRS: {label_df_test.crs}')
    else:
        print(f'no test data exists for {country}_{region}')
        label_df_test=None

    # ### RasterIO
    poly_path = data_dir/f'{country}_{region}'
    poly_path.mkdir(exist_ok=True)
    cropped = data_dir/f'{country}_{region}/cropped'
    cropped.mkdir(exist_ok=True)
    cropped_trn= data_dir/f'{country}_{region}/cropped/train'
    cropped_trn.mkdir(exist_ok=True)
    if test_exists:
        cropped_test= data_dir/f'{country}_{region}/cropped/test'
        cropped_test.mkdir(exist_ok=True)
        rot_dir = data_dir / f'{country}_{region}/cropped/test/rotated'
        rot_dir.mkdir(exist_ok=True)
        rot_clipped_dir = data_dir / f'{country}_{region}/cropped/test/rotated/clipped'
        rot_clipped_dir.mkdir(exist_ok=True)
    rot_dir = data_dir / f'{country}_{region}/cropped/train/rotated'
    rot_dir.mkdir(exist_ok=True)
    rot_clipped_dir = data_dir / f'{country}_{region}/cropped/train/rotated/clipped'
    rot_clipped_dir.mkdir(exist_ok=True)
    return label_df, label_df_test, DATA_TIFF, test_exists



def workflow():
    for data_path, country, region in zip(path_list, country_list, region_list):
        print(f'{data_path} {country} {region}')
        start=time.time()
        df, df_test, data_tif, test_exists = format_all(data_path, country, region)
        mask_images(df, df_test, data_tif, test_exists, country, region, buffer=BUFFER)
        crop_non_transparent(data_dir/f'{country}_{region}/cropped/train/rotated', data_dir/f'{country}_{region}/cropped/train/rotated/clipped')
        if test_exists:
            crop_non_transparent(data_dir / f'{country}_{region}/cropped/test/rotated',
                                 data_dir / f'{country}_{region}/cropped/test/rotated/clipped')
        end = time.time()
        print(f'elapsed: {end-start}')

if __name__ == "__main__":
    workflow()















