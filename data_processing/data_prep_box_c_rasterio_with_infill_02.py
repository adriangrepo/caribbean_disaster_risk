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

import matplotlib
#matplotlib.rcParams['backend'] = "TkAgg"
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
import decimal
from shapely.geometry import shape
import shapely
import skimage
from tqdm import tqdm
import pandas as pd

import fiona
import numpy as np
from itertools import product
from rasterstats.io import Raster
from PIL import Image

from rio_tiler import main as rt_main
# import mercantile
import rasterio as rio
from rasterio.plot import show
from rasterio.mask import mask
from rasterio.mask import raster_geometry_mask
from shapely.geometry import mapping, MultiPolygon
from rasterio.transform import from_bounds
from rasterio.plot import reshape_as_raster, reshape_as_image
from shapely.geometry import Polygon as shapely_polygon
from shapely.ops import cascaded_union
from shapely import affinity
from shapely.geometry import MultiPoint
import cv2

import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

'''
Spent a lot of time getting masks to work as outline for pasting, but ended up could paste with pillow using alpha 
and not requiring masks, mask code superfluos

Padding code takes original image, roates it , makes 4 copes, shifts to corners, uses as bg and puts origial over the top
'''

BUFFER = 0.1
PAD = False
#if True then dont wite any files
DEBUG = False
SHOW_PLOTS=False
SAVE_CENTROIDS=True

d = Path().resolve().parent

#data_src = 'data'
data_src = 'data_03'
data_orig = 'data'

colombia_rural = Path(d/f'{data_src}/stac/colombia/borde_rural')
colombia_soacha = Path(d/f'{data_src}/stac/colombia/borde_soacha')

guatemala_mixco1 = Path(d/f'{data_src}/stac/guatemala/mixco_1_and_ebenezer')
guatemala_mixco3 = Path(d/f'{data_src}/stac/guatemala/mixco_3')

st_lucia_castries = Path(d/f'{data_src}/stac/st_lucia/castries')
st_lucia_dennery = Path(d/f'{data_src}/stac/st_lucia/dennery')
st_lucia_gros_islet = Path(d/f'{data_src}/stac/st_lucia/gros_islet')

data_dir = Path(d/f'{data_src}')
print(f'data_dir: {str(data_dir)}')

path_list= [colombia_rural, colombia_soacha, guatemala_mixco1, guatemala_mixco3,
         st_lucia_castries, st_lucia_dennery, st_lucia_gros_islet]
json_list=['train-borde_rural.geojson','train-borde_soacha.geojson','train-mixco_1_and_ebenezer.geojson','train-mixco_3.geojson',
          'train-castries.geojson','train-dennery.geojson','train-gros_islet.geojson']
country_list=['colombia','colombia','guatemala','guatemala','st_lucia','st_lucia','st_lucia']
region_list=['borde_rural','borde_soacha','mixco_1_and_ebenezer','mixco_3','castries','dennery','gros_islet']

def create_paths(country, region):
    test_exists=True
    if region.strip() in ['castries','gros_islet']:
        test_exists=False
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
        pad_clipped_dir = data_dir / f'{country}_{region}/cropped/test/rotated/clipped/padded'
        pad_clipped_dir.mkdir(exist_ok=True)
    rot_dir = data_dir / f'{country}_{region}/cropped/train/rotated'
    rot_dir.mkdir(exist_ok=True)
    rot_clipped_dir = data_dir / f'{country}_{region}/cropped/train/rotated/clipped'
    rot_clipped_dir.mkdir(exist_ok=True)
    pad_clipped_dir = data_dir / f'{country}_{region}/cropped/train/rotated/clipped/padded'
    pad_clipped_dir.mkdir(exist_ok=True)

def mpl_plot(np_img, fn, title, id):
    if SHOW_PLOTS:
        plt.imshow(np_img, interpolation='nearest')
        plt.title(f'{fn} {title}')
        plt.axis('off')
        plt.show(block=False)
        plt.close()
    else:
        pass

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
    is_multi=False
    area=0
    polygon=None
    if len(poly)>0:
        if isinstance(poly.iloc[0], MultiPolygon):
            is_multi=True
            poly=cascaded_union(poly)
            #may still be multipolygon
            if poly.geom_type == 'MultiPolygon':
                # extract polygons out of multipolygon and keep biggest one
                max_area=0
                index=0
                for i,polygon in enumerate(poly):
                    if polygon.area>max_area:
                        max_area=polygon.area
                        index=i
                area=max_area
                poly=poly[index]
                polygon=poly
                c=list(poly.exterior.coords)
        else:
            area=poly.area
            polygon = poly
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
    return np.degrees(theta), is_multi, area, polygon

def prep_poly(poly):
    '''returns lists of x, y from polygon for plotting'''
    lr_rot = poly.exterior.values[0]
    c = list(lr_rot.coords)
    c = list(sum(c, ()))
    x = oddTuples(c)
    y = evenTuples(c)
    return x, y

def to_rgb_to_np(np_in):
    rgb_img = Image.fromarray(np_in).convert('RGB')
    np_back = np.array(rgb_img)
    return np_back

def to_rgba_to_np(np_in):
    rgba_img = Image.fromarray(np_in).convert('RGBA')
    np_back = np.array(rgba_img)
    return np_back

def crop_TL(img, height, width):
    assert isinstance(height, int)
    assert isinstance(width, int)
    upper = int(height / 4)
    left = int(width / 4)
    right=width
    lower=height
    #The crop rectangle (left, upper, right, lower)-tuple.
    box = (left, upper, right, lower)
    img_TL=img.crop(box)
    img_pad = Image.new('RGBA', (width, height), 0)
    # pixels must be integers - here we round to nearest
    n_x = round(decimal.Decimal(0))
    n_y = round(decimal.Decimal(0))
    pbox = (n_x, n_y, img_TL.width, img_TL.height)
    #print(f'crop_TL img_pad: {img_pad.size}, img_TL: {img_TL.size}, pbox: {pbox}')
    img_pad.paste(img_TL, pbox)
    img_TL = img_pad
    return img_TL

def mask_TL(height, width, shape_mask):
    assert isinstance(height, int)
    assert isinstance(width, int)
    upper = int(height / 4)
    left = int(width / 4)
    c=3
    #create boolean index array
    np_TL = np.ones(shape=(height, width, c), dtype=np.uint8)
    np_TL = np_TL * 255
    mask_TL = shape_mask[upper:, left:, :]
    #print(f'mask_TL: {mask_TL.shape}, np_TL: {np_TL.shape}')
    assert height-upper==mask_TL.shape[0]
    assert width-left == mask_TL.shape[1]
    np_TL[0:height-upper, 0:width-left, :]=mask_TL
    return np_TL

def crop_BL(img, height, width):
    upper = 0
    left = int(width / 4)
    right = width
    lower = height - int(height / 4)
    box = (left, upper, right, lower)
    img_BL = img.crop(box)
    img_pad = Image.new('RGBA', (width, height), 0)
    n_x = round(decimal.Decimal(0))
    n_y = round(decimal.Decimal(int(height / 4)))
    img_pad.paste(img_BL, (n_x, n_y))
    img_BL = img_pad
    return img_BL

def mask_BL(height, width, shape_mask):
    upper = 0
    left = int(width / 4)
    c = 3
    np_BL = np.ones(shape=(height, width, c), dtype=np.uint8)
    np_BL = np_BL * 255
    mask_BL = shape_mask[upper:height-int(height / 4), left:, :]
    #print(f'mask_BL: {mask_BL.shape}, np_BL: {np_BL.shape}')
    np_BL[upper+int(height / 4):height, 0:width-left, :]=mask_BL
    return np_BL

def crop_TR(img, height, width):
    upper = int(height / 4)
    left = 0
    right = width-int(width / 4)
    lower = height
    # The crop rectangle (left, upper, right, lower)-tuple.
    box = (left, upper, right, lower)
    img_TR = img.crop(box)
    img_pad = Image.new('RGBA', (width, height), 0)
    n_x = round(decimal.Decimal(int(width / 4)))
    n_y = round(decimal.Decimal(0))
    img_pad.paste(img_TR, (n_x, n_y))
    img_TR = img_pad
    return img_TR

def mask_TR(height, width, shape_mask):
    upper = int(height / 4)
    left = 0
    c = 3
    # create boolean index array
    np_TR = np.ones(shape=(height, width, c), dtype=np.uint8)
    np_TR = np_TR * 255
    mask_TR = shape_mask[upper:, left:width-int(width / 4), :]
    #print(f'mask_TR: {mask_TR.shape}, np_TR: {np_TR.shape}')
    np_TR[0:height - upper, left+int(width / 4):width, :] = mask_TR
    return np_TR

def crop_BR(img, height, width):
    upper = 0
    left = 0
    right = width-int(width / 4)
    lower = height-int(height / 4)
    box = (left, upper, right, lower)
    img_BR = img.crop(box)
    img_pad = Image.new('RGBA', (width, height), 0)
    n_x = round(decimal.Decimal(int(width / 4)))
    n_y = round(decimal.Decimal(int(height / 4)))
    img_pad.paste(img_BR, (n_x, n_y))
    img_BR = img_pad
    return img_BR

def mask_BR(height, width, shape_mask):
    upper = 0
    left = 0
    c = 3
    np_BR = np.ones(shape=(height, width, c), dtype=np.uint8)
    np_BR = np_BR * 255
    mask_BR = shape_mask[upper:height-int(height / 4), left:width-int(width / 4), :]
    #print(f'mask_BR: {mask_BR.shape}, np_BR: {np_BR.shape}')
    np_BR[upper+int(height / 4):height, int(width / 4):width, :] = mask_BR
    return np_BR

def crop_about_c(img_np, country, region, tt_path, id):
    ''' Shift original image by 1/4 image h and w in 4 directions with crop and pad for each
    Ditto for original image mask
    Combine all where mask=true keep to create rich background
    Paste original image over the top of bg.
    Not using masks'''
    #assert img_np.shape[0]==mask_np.shape[0]
    #assert img_np.shape[1] == mask_np.shape[1]
    assert img_np.shape[2] == 4
    # RGB
    #assert mask_np.shape[2] == 3
    height = img_np.shape[0]
    width = img_np.shape[1]

    img = Image.fromarray(img_np, 'RGBA')
    img_TL = crop_TL(img, height, width)
    img_BL = crop_BL(img, height, width)
    img_TR = crop_TR(img, height, width)
    img_BR = crop_BR(img, height, width)

    #np_TL = mask_TL(height, width, mask_np)
    #np_BL = mask_BL(height, width, mask_np)
    #np_TR = mask_TR(height, width, mask_np)
    #np_BR = mask_BR(height, width, mask_np)

    repeated_images = [img_TL,img_BL,img_TR,img_BR]
    #repeated_masks = [np_TL, np_BL, np_TR, np_BR]
    return repeated_images

def raster_pts(img, p):
    '''get points inside polygon
    "height": out_image.shape[1],
    "width": out_image.shape[2]'''
    #https://stackoverflow.com/questions/44399749/get-all-lattice-points-lying-inside-a-shapely-polygon
    idx = p.index.values[-1]
    xmin, ymin, xmax, ymax = p[idx].bounds
    points = MultiPoint(list(product(range(img.shape[2]), range(img.shape[1]))))
    #print(f'{idx},{xmin},{ymin},{xmax},{ymax},{img.shape[2]},{img.shape[1]}')
    result = points.intersection(p[idx])
    return result

def combine_background(img_list, std_np):
    # Put logo in ROI and modify the main image
    '''see https://docs.opencv.org/trunk/d0/d86/tutorial_py_image_arithmetics.html
    img_TL,img_BL,img_TR,img_BR, np_TL, np_BL, np_TR, np_BR
    Note we dont need masks'''
    assert isinstance(std_np, np.ndarray)

    row, col, ch = std_np.shape
    img_TL=np.array(img_list[0])
    img_BL=np.array(img_list[1])
    img_TR=np.array(img_list[2])
    img_BR=np.array(img_list[3])

    std_img = Image.fromarray(std_np).convert('RGBA')
    rgba_TL = Image.fromarray(img_TL).convert('RGBA')
    rgba_BL = Image.fromarray(img_BL).convert('RGBA')
    rgba_TR = Image.fromarray(img_TR).convert('RGBA')
    rgba_BR = Image.fromarray(img_BR).convert('RGBA')
    img_pad = Image.new('RGBA', (col, row), 0)
    position = (0, 0)
    # print(f'position: {position}')
    img_pad.paste(rgba_TL, position)
    img_pad.paste(rgba_BL, position, rgba_BL)
    img_pad.paste(rgba_TR, position, rgba_TR)
    img_pad.paste(rgba_BR, position, rgba_BR)
    # put main image back over the top
    img_pad.paste(std_img, position, std_img)
    #img_pad.show()
    return img_pad

def rasterio_mask(dataset, shapes, all_touched=False, invert=False, nodata=None,
         filled=True, crop=False, pad=False, pad_width=0.5, indexes=None):

    if nodata is None:
        if dataset.nodata is not None:
            nodata = dataset.nodata
        else:
            nodata = 0

    shape_mask, transform, window = raster_geometry_mask(
        dataset, shapes, all_touched=all_touched, invert=invert, crop=crop,
        pad=pad, pad_width=pad_width)

    if indexes is None:
        out_shape = (dataset.count, ) + shape_mask.shape
    elif isinstance(indexes, int):
        out_shape = shape_mask.shape
    else:
        out_shape = (len(indexes), ) + shape_mask.shape

    out_image = dataset.read(
        window=window, out_shape=out_shape, masked=True, indexes=indexes)

    out_image.mask = out_image.mask | shape_mask

    '''
    #We dont need to return mask image anymore
    out_mask=out_image.mask.copy()
    #convert mask into a valid 3 channel image, 0 for roof, bg 255
    copy_image=out_image.copy()
    x=np.ma.getmaskarray(copy_image)
    xi=x.astype(np.int32)
    x_im = np.ma.transpose(xi, [1, 2, 0])
    x_im = x_im[:, :, 0]
    x_im = x_im*255
    #addding a value of 1 for actual roof instead of zero - see rotation
    x_im[x_im == 0] = 1
    x_img = Image.fromarray(x_im).convert('RGB')
    '''

    if filled:
        out_image = out_image.filled(nodata)

    return out_image, transform

def get_info_from_polygon(polygon):
    xy = []
    if isinstance(polygon, shapely.geometry.Polygon):
        x = polygon.centroid.x
        y = polygon.centroid.y
        xy = (x, y)
        area = polygon.area
    elif isinstance(polygon, gpd.GeoSeries):
        try:
            vp = polygon.centroid.values
            if len(vp) == 0:
                return xy
            if len(vp)==1:
                x=vp[0].x
                y=vp[0].y
                xy=[x,y]
            else:
                x=vp[-1].x
                y=vp[-1].y
                xy = (x, y)
        except AttributeError as e:
            print(f'Error, not polygon.centroid.values')
        try:
            if len(polygon)==0:
                area = 0
            #if len(polygon)==1:
            #    area = polygon[0].area
            else:
                idxs=polygon.index.tolist()
                area = polygon[idxs[-1]].area
        except KeyError as e:
            print(e)
    else:
        print(f'Error, unknown type: {type(polygon)}')
    return xy, area

def save_centroids(poly_centroids, poly_areas, poly_bounds, country, region, tt_path):
    print(f'save_centroids  centroids: {len(poly_centroids)} areas: {len(poly_areas)} bounds: {len(poly_bounds)}')
    col_names = ['id', 'centroid', 'area']
    df_data = pd.DataFrame(columns=col_names)
    df_data['id']=poly_centroids.keys()
    df_data['centroid'] = poly_centroids.values()
    df_data['area'] = poly_areas.values()
    #df_data['bbox']=poly_bounds.values()
    '''TODO convert these vales to format that can save
    dict_values([            minx           miny           maxx           maxy
0  593322.014631  503542.106432  593331.550536  503555.329639,             minx           miny           maxx           maxy
1  593314.707898  503544.685727  593324.739175  503557.208689,            minx           miny           maxx           maxy'''

    df_coord = pd.DataFrame(df_data)
    print(df_coord.head())
    df_name=Path(d/f'data/df_{country}_{region}_{tt_path}_centroids.csv')
    df_coord.to_csv(str(df_name), index = False)
    print(f'saved centroids, areas to df_{country}_{region}_{tt_path}_centroids.csv')

def mask_raster(df, tif_filename, country, region, tt_path, buffer=2, debug=False):
    angles={}
    multi_ids=[]
    areas_list=[]
    poly_centroids={}
    poly_areas={}
    poly_bounds={}
    with rasterio.open(tif_filename) as src:
        for index, row in df.iterrows():
            id = row["id"]
            poly = df.iloc[[index]]['geometry']
            theta, is_multi, area, polygon =get_angle(poly)
            if is_multi:
                multi_ids.append(id)
                areas_list.append(area)
            angles[id]=theta

            poly = poly.buffer(distance=buffer, cap_style=2, join_style=2)
            xy_l, area=get_info_from_polygon(polygon)
            poly_bounds[id]=polygon.bounds
            poly_centroids[id]=" ".join([str(i) for i in xy_l])
            poly_areas[id]=area

            out_np, out_transform = rasterio_mask(src, poly, crop=True)
            out_meta = src.meta
            out_meta.update({"driver": "GTiff",
                             "height": out_np.shape[1],
                             "width": out_np.shape[2],
                             "transform": out_transform})

            path_out = data_dir / f'{country}_{region}/cropped/{tt_path}/{id}.tif'
            if not debug:
                with rasterio.open(path_out, "w", **out_meta) as dest:
                    dest.write(out_np)

            # rotate before padding
            # (but can't rotate image before rasterio mask above as image is 5GB)
            std_np=reshape_as_image(out_np)
            std_img = Image.fromarray(std_np, 'RGBA')
            std_img = std_img.rotate(theta * -1, resample=Image.BICUBIC, expand=True)
            std_img_90 = std_img.rotate(90, resample=Image.BICUBIC, expand=True)
            # has RGBA channels
            std_np = np.array(std_img)
            std_np_90 = np.array(std_img_90)

            #Not using masks anymore, not required
            #mask_img = mask_img.rotate(theta * -1, resample=Image.BICUBIC, expand=True)
            #mask_img_90 = mask_img.rotate(90, resample=Image.BICUBIC, expand=True)
            # mask has just L channel
            # when rotated puts 0 in blank areas - fill these with non-image 255
            # ie image after rotate and replace: 255==non roof, 1==roof
            #mask_np_90 = np.array(mask_img_90)
            #mask_np = np.array(mask_img)
            #mask_np[mask_np == 0] = 255
            #mask_np_90[mask_np_90 == 0] = 255

            # create 4x4 background
            repeated_images = crop_about_c(std_np, country, region, tt_path, id)
            rotated = combine_background(repeated_images, std_np)

            repeated_images_90 = crop_about_c(std_np_90, country, region, tt_path, id)
            rotated_90 = combine_background(repeated_images_90, std_np_90)

            if PAD:
                pad_path = data_dir/f'{country}_{region}/cropped/{tt_path}/rotated/bg_padded'
                pad_path.mkdir(exist_ok=True)
                pad_path = data_dir / f'{country}_{region}/cropped/{tt_path}/rotated/padded'
                pad_path.mkdir(exist_ok=True)
                #mask_path = data_dir/f'{country}_{region}/cropped/{tt_path}/rotated/bg_padded/mask'
                #mask_path.mkdir(exist_ok=True)
                pad_rot_out = data_dir/f'{country}_{region}/cropped/{tt_path}/rotated/bg_padded/{id}.tif'
                pad_rot_out_90 = data_dir / f'{country}_{region}/cropped/{tt_path}/rotated/bg_padded/{id}_rot90.tif'
                rot_out = data_dir/f'{country}_{region}/cropped/{tt_path}/rotated/{id}.tif'
                rot_out_90 = data_dir / f'{country}_{region}/cropped/{tt_path}/rotated/{id}_rot90.tif'
                #np_mask_out = mask_path/f'{id}_mask'
                if not debug:
                    #save padded bg, original rotated image and npy mask (cant workout how to savas as propper image) for later use
                    rotated.save(pad_rot_out)
                    std_img.save(rot_out)
                    rotated_90.save(pad_rot_out_90)
                    std_img_90.save(rot_out_90)
                    #np.save(np_mask_out, mask_np)
                    #mask_np_test=np.load(np_mask_out+'.npy')
                    #mask_img_test = Image.fromarray(mask_np)
                    #mask_img_test.show(title='mask_img_test')
                    if index % 1000 == 0:
                        print(f'saved {pad_rot_out} {rot_out}')
            else:
                std_out = data_dir / f'{country}_{region}/cropped/{tt_path}/rotated/{id}.tif'
                std_out_90 = data_dir / f'{country}_{region}/cropped/{tt_path}/rotated/{id}_rot90.tif'
                if not debug:
                    std_img.save(std_out)
                    std_img_90.save(std_out_90)
                    if index % 1000 == 0:
                        print(f'saved {std_out_90} {std_out}')

        if debug and len(multi_ids)>0:
            print(f'multi_ids country: {country}, region: {region}, count: {len(multi_ids)}, ids: {multi_ids}, areas: {areas_list}')

    if SAVE_CENTROIDS:
        save_centroids(poly_centroids, poly_areas, poly_bounds, country, region, tt_path)

def crop_inc_pad_non_transparent(dir, pad_out_path, bg_pad_dir):
    '''crops tranparent areas outside of the roof
    as below but use crop with non padded background to best locate crop then crop the padded image'''
    padfiles = [f for f in listdir(bg_pad_dir) if isfile(join(bg_pad_dir, f))]
    onlyfiles = [f for f in listdir(dir) if isfile(join(dir, f))]
    print(f'--crop_non_transparent() input: {dir}, out path: {pad_out_path}, padded bg: {bg_pad_dir}')
    crop_d={}
    for f in onlyfiles:
        assert(os.path.isfile(dir/f'{f}'))
        im = cv2.imread(str(dir/f'{f}'), cv2.IMREAD_UNCHANGED)
        # axis 0 is the row(y) and axis(x) 1 is the column
        y, x = im[:, :, 3].nonzero()  # get the nonzero alpha coordinates
        minx = np.min(x)
        miny = np.min(y)
        maxx = np.max(x)
        maxy = np.max(y)
        id=f.split('/')[-1].split('.tif')[0]
        crop_d[id]=[miny, maxy, minx, maxx]
    for f in padfiles:
        assert (os.path.isfile(bg_pad_dir / f'{f}'))
        im = cv2.imread(str(bg_pad_dir / f'{f}'), cv2.IMREAD_UNCHANGED)
        id=f.split('/')[-1].split('.tif')[0]
        #total hack to handle errant id's that should have been deleted
        if not id.endswith('_90'):
            coord_list=crop_d[id]
            cropImg = im[coord_list[0]:coord_list[1], coord_list[2]:coord_list[3]]
            cv2.imwrite(str(pad_out_path/f'{f}'), cropImg)

def crop_non_transparent(dir, out_path):
    '''crops tranparent areas outside of the roof'''
    onlyfiles = [f for f in listdir(dir) if isfile(join(dir, f))]
    print(f'--crop_non_transparent() {dir},  {out_path}')
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

def mask_images(df, df_test, data_tif, test_exists, country, region, buffer, debug=False):
    mask_raster(df, data_tif, country, region, tt_path='train', buffer=buffer, debug=debug)

    if test_exists:
        mask_raster(df_test, data_tif, country, region, tt_path='test', buffer=buffer, debug=debug)

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
    create_paths(country, region)
    return label_df, label_df_test, DATA_TIFF, test_exists

def workflow():
    data_postfix = 'cropped/train/rotated'
    clip_postfix = 'cropped/train/rotated/clipped'
    data_test_postfix = 'cropped/test/rotated'
    clip_test_postfix = 'cropped/test/rotated/clipped'

    for data_path, country, region in zip(path_list, country_list, region_list):
        print(f'{data_path} {country} {region}')
        start=time.time()
        df, df_test, data_tif, test_exists = format_all(data_path, country, region)
        mask_images(df, df_test, data_tif, test_exists, country, region, buffer=BUFFER, debug=DEBUG)
        if not DEBUG:
            dir = data_dir / f'{country}_{region}/{data_postfix}'
            out_path = data_dir / f'{country}_{region}/{clip_postfix}'
            test_dir = data_dir / f'{country}_{region}/{data_test_postfix}'
            test_out_path = data_dir / f'{country}_{region}/{clip_test_postfix}'
            if PAD:
                bg_pad_dir = dir/'bg_padded'
                bg_pad_dir.mkdir(exist_ok=True)
                test_pad_dir = test_dir / 'bg_padded'
                pad_out_path = out_path/'padded'
                pad_out_path.mkdir(exist_ok=True)
                test_pad_out_path = test_out_path/'padded'
                crop_inc_pad_non_transparent(dir, pad_out_path, bg_pad_dir)
                if test_exists:
                    test_pad_dir.mkdir(exist_ok=True)
                    test_pad_out_path.mkdir(exist_ok=True)
                    crop_inc_pad_non_transparent(test_dir, test_pad_out_path, test_pad_dir)
            else:
                crop_non_transparent(dir, out_path)
                if test_exists:
                    crop_non_transparent(test_dir, test_out_path)

        end = time.time()
        print(f'elapsed: {end-start}')

if __name__ == "__main__":
    workflow()















