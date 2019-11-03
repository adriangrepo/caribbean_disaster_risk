#!/usr/bin/env python
# coding: utf-8

# # Data prep
# 
# 

# # Pre-Processing
# 
# colombia_soacha
# 
# **Note that the preprocessing section is possible to be done on CPU runtime:**
# 
# Change in menu: Runtime > Change runtime type > Hardware Accelerator = None

# ## Install all the geo things
# 
# `Pip install` the required geodata processing packages we'll be using of, test that their import to Colab works, and create our output data directories.

# !add-apt-repository ppa:ubuntugis/ppa
# !apt-get update
# !apt-get install python-numpy gdal-bin libgdal-dev
# !apt install python3-rtree
# 
# !pip install rasterio
# !pip install geopandas
# !pip install descartes
# !pip install solaris
# !pip install rio-tiler

# In[1]:


# for bleeding edge version of solaris:
# !pip install git+https://github.com/CosmiQ/solaris/@dev


# In[2]:


import solaris as sol
import numpy as np
import geopandas as gpd
from matplotlib import pyplot as plt
from pathlib import Path
import rasterio
import os
import skimage
from tqdm import tqdm
import pandas as pd

import fiona
import numpy as np

from rasterstats.io import Raster
from PIL import Image

from rio_tiler import main as rt_main
# import mercantile
from rasterio.transform import from_bounds
from shapely.geometry import Polygon as shapely_polygon
from shapely.ops import cascaded_union


# In[3]:


tile_size = 256
zoom_level = 21


# ## Preview and load imagery and labels
# 

# In[4]:


colombia_rural = Path('data/stac/colombia/borde_rural')
colombia_soacha = Path('data/stac/colombia/borde_soacha')


# In[5]:


guatemala_mixco1 = Path('data/stac/guatemala/mixco_1_and_ebenezer')
guatemala_mixco3 = Path('data/stac/guatemala/mixco_3')


# In[6]:


st_lucia_castries = Path('data/stac/st_lucia/castries')
st_lucia_dennery = Path('data/stac/st_lucia/dennery')
st_lucia_gros_islet = Path('data/stac/st_lucia/gros_islet')


# In[7]:


data_dir = Path('data')
data_dir.mkdir(exist_ok=True)


# In[8]:


for pth in ['colombia_borde_rural', 'colombia_borde_soacha', 'guatemala_mixco_1', 'guatemala_mixco_3',
         'st_lucia_castries','st_lucia_dennery', 'st_lucia_gros_islet']:
    img_path = data_dir/f'{pth}_images-{tile_size}'
    mask_path = data_dir/f'{pth}_masks-{tile_size}'
    img_path.mkdir(exist_ok=True)
    mask_path.mkdir(exist_ok=True)


# In[9]:


COUNTRY='colombia'
REGION='borde_soacha'


# In[10]:


DATASET = f'{COUNTRY}_{REGION}'


# In[11]:


DATASET_PATH=colombia_soacha


# In[12]:


DATA_TIFF = DATASET_PATH/f'{REGION}_ortho-cog.tif'
TRAIN_JSON = f'train-{REGION}.geojson'
TRAIN_TRN_JSON = f'train-{REGION}_trn.geojson'
TRAIN_VAL_JSON = f'train-{REGION}_val.geojson'
TEST_JSON = f'test-{REGION}.geojson'
PNG_THUMBNAL = DATASET_PATH/f'{REGION}_ortho-cog-thumbnail.png'


# In[13]:


img_path=data_dir/f'{DATASET}_images-{tile_size}'
img_path


# In[14]:


mask_path=data_dir/f'{DATASET}_masks-{tile_size}'


# In[15]:


rasterio.open(DATA_TIFF).meta


# In[16]:


# load geojson colombia rural training data

label_df = gpd.read_file(DATASET_PATH/TRAIN_JSON)
label_df = label_df[label_df['geometry'].isna() != True] # remove empty rows


# In[67]:


label_df.crs


# In[68]:


label_df = label_df.to_crs({'init': 'epsg:32618'}) 


# In[69]:


label_df.crs


# In[70]:


label_df.head()


# In[71]:


label_df.plot(figsize=(10,10))


# In[72]:


# load geojson colombia rural test data

label_df_test = gpd.read_file(DATASET_PATH/TEST_JSON)
label_df_test = label_df_test[label_df_test['geometry'].isna() != True] # remove empty rows


# In[73]:


label_df_test.crs


# In[74]:


label_df_test = label_df_test.to_crs({'init': 'epsg:32618'}) 
label_df_test.crs


# In[75]:


label_df_test.plot(figsize=(10,10))


# In[76]:


from PIL import Image
path = PNG_THUMBNAL
display(Image.open(path))


# In[77]:


label_df.head()


# In[78]:


len(label_df)


# ### Shapely (not working)

# In[26]:


poly_path = data_dir/f'{COUNTRY}_{REGION}'
poly_path.mkdir(exist_ok=True)
polys = data_dir/f'{COUNTRY}_{REGION}/polygons'
polys.mkdir(exist_ok=True)


# In[52]:


from shapely.geometry import mapping
import shapely
#see https://gis.stackexchange.com/questions/297088/clipping-geotiff-with-shapefile
def polygonize(df, tif_filename):
    with Raster(tif_filename, band=1) as raster_obj_1:
        with Raster(tif_filename, band=2) as raster_obj_2:
            with Raster(tif_filename, band=3) as raster_obj_3:
                index = 0
                for row in df.itertuples():
                    id = getattr(row, "id")
                    polygon = getattr(row, "geometry")
                    #print(type(polygon))
                    if isinstance(polygon, shapely.geometry.polygon.Polygon):
                        polygon_geometry= polygon.exterior.coords
                    m_poly=mapping(polygon)
                    print(m_poly)
                    #print(polygon_geometry['coordinates'][0])
                    #polygon = shapely_polygon(polygon_geometry['coordinates'][0])
                    polygon_bounds = polygon.bounds

                    raster_subset_1 = raster_obj_1.read(bounds=polygon_bounds)
                    print(raster_subset_1.shape[0])
                    print(raster_subset_1.shape[1])
                    polygon_mask = rasterio.features.geometry_mask(geometries=[m_poly],
                                                            out_shape=(raster_subset_1.shape[0],raster_subset_1.shape[1]),
                                                            transform=raster_subset_1.affine,
                                                            all_touched=False,
                                                            invert=True)
                    print(f'polygon_mask: {polygon_mask}')

                    raster_subset_2 = raster_obj_2.read(bounds=polygon_bounds)
                    raster_subset_3 = raster_obj_3.read(bounds=polygon_bounds)

                    masked_1 = raster_subset_1.array * polygon_mask
                    masked_2 = raster_subset_2.array * polygon_mask
                    masked_3 = raster_subset_3.array * polygon_mask

                    masked_all = np.dstack([masked_1, masked_2, masked_3])

                    img = Image.fromarray(masked_all[:, :, :].astype('uint8'), 'RGB')
                    img.save(data_dir/f'{COUNTRY}_{REGION}/polygons/{id}.jpg', dpi=(300,300))
                    index += 1


# In[79]:


#polygonize(trn_all, DATA_TIFF)


# ### EarthPy

# In[56]:


import os
import numpy as np
import rasterio as rio
from rasterio.plot import show
from rasterio.mask import mask
from shapely.geometry import mapping
import matplotlib.pyplot as plt
import geopandas as gpd
import earthpy as et
import earthpy.plot as ep
import earthpy.spatial as es
import cartopy as cp

# set home directory and download data
#et.data.get_data("spatial-vector-lidar")
#os.chdir(os.path.join(et.io.HOME, 'earth-analytics'))

# optional - turn off warnings
import warnings
warnings.filterwarnings('ignore')


# In[62]:


def earth_py_proc(staellite_tif):
    # open the tiff
    with rio.open(staellite_tif) as src:
        chm_im = src.read(masked=True)[0]
        extent = rio.plot.plotting_extent(src)
        soap_profile = src.profile

    ep.plot_bands(chm_im,
                   cmap="RdYlGn",
                   extent=extent,
                   title="data",
                   cbar=False);
    return chm_im, extent, soap_profile


# In[63]:


chm_im, extent, soap_profile=earth_py_proc(DATA_TIFF)


# In[80]:


print('poygon crs: ', label_df.crs)
print('tiff crs: ', soap_profile['crs'])


# In[81]:


# plot the data
fig, ax = plt.subplots(figsize = (6, 6))
label_df.plot(ax=ax)
ax.set_title("labels", 
             fontsize = 16)
ax.set_axis_off();


# In[87]:


# plot nth row
fig, ax = plt.subplots(figsize = (6, 6))
label_df.iloc[[2]].plot(ax=ax)
ax.set_title("labels", 
             fontsize = 16)
ax.set_axis_off();


# In[83]:


fig, ax = plt.subplots(figsize=(10, 10))
ep.plot_bands(chm_im,
              extent=extent,
              ax=ax,
              cbar=False)
label_df.plot(ax=ax, alpha=.5, color='b');


# ### crop

# In[88]:


poly_path = data_dir/f'{COUNTRY}_{REGION}'
poly_path.mkdir(exist_ok=True)
polys = data_dir/f'{COUNTRY}_{REGION}/cropped'
polys.mkdir(exist_ok=True)


# In[ ]:


plt.ioff()


# In[111]:


with rio.open(DATA_TIFF) as src:
    crops=[]
    metas=[]
    for index, row in label_df.iterrows():
        id = row["id"]
        chm_crop, data_meta = es.crop_image(src, label_df.iloc[[index]])
        # Update the metadata to have the new shape (x and y and affine information)
        data_meta.update({"driver": "GTiff",
                         "height": chm_crop.shape[0],
                         "width": chm_crop.shape[1],
                         "transform": data_meta["transform"]})

        # generate an extent for the newly cropped object for plotting
        cr_ext = rio.transform.array_bounds(data_meta['height'], 
                                                    data_meta['width'], 
                                                    data_meta['transform'])

        bound_order = [0,2,1,3]
        cr_extent = [cr_ext[b] for b in bound_order]
        # mask the nodata and plot the newly cropped raster layer
        chm_crop_ma = np.ma.masked_equal(chm_crop[0], -9999.0) 
        # Save to disk 
        path_out=data_dir/f'{COUNTRY}_{REGION}/cropped/{id}.tif'
        fig = plt.gcf()
        ep.plot_bands(chm_crop[0], cmap='terrain', cbar=False)
        #if index<5:
        plt.savefig(path_out)
        #plt.show()
        #plt.close(fig)
        
        #with rio.open(path_out, 'w', **data_meta) as ff:
            #ff.write(chm_crop[0], 1)
            #ff.write(chm_crop_ma, 1)
        
        #crops.append(chm_crop)
        #metas.append(data_meta)


# In[ ]:


crops[0]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


#of training tiles reserve 10% for a validation set
trn_tiles, val_tiles = np.split(trn_all.sample(frac=1), [int(.9*len(trn_all))])


# In[ ]:


len(trn_tiles)


# In[105]:


len(val_tiles)


# ### supermercado-burn

# Complicated step here to create X, Y, Z tiles

# In[106]:


trn_tiles['dataset'] = 'trn'
val_tiles['dataset'] = 'val'


# In[107]:


len(val_tiles)


# In[108]:


trn_tiles.to_file(data_dir/f"tiles/{COUNTRY}_{TRAIN_TRN_JSON}", driver='GeoJSON')
val_tiles.to_file(data_dir/f"tiles/{COUNTRY}_{TRAIN_VAL_JSON}", driver='GeoJSON')


# In[109]:


val_tiles.head()


# In[110]:


trn_tiles.head()


# In[111]:


trn_out=Path(f"{data_dir}/tiles/{DATASET}_trn_aoi_z{zoom_level}_tiles.geojson")
val_out=Path(f"{data_dir}/tiles/{DATASET}_val_aoi_z{zoom_level}_tiles.geojson")
trn_out, val_out


# In[112]:


trn_in=f"data/tiles/{COUNTRY}_{TRAIN_TRN_JSON}"
val_in=f"data/tiles/{COUNTRY}_{TRAIN_VAL_JSON}"
trn_in, val_in


# In[113]:


get_ipython().system('pwd')


# In[114]:


get_ipython().system('cat {trn_in} | supermercado burn {zoom_level} | mercantile shapes | fio collect > {trn_out}')
get_ipython().system('cat {val_in} | supermercado burn {zoom_level} | mercantile shapes | fio collect > {val_out}')


# In[115]:


#of training tiles reserve 10% for a validation set
trn_tiles, val_tiles = np.split(trn_all.sample(frac=1), [int(.9*len(trn_all))])


# In[116]:


trn_tiles_burnt = gpd.read_file(trn_out)
val_tiles_burnt = gpd.read_file(val_out)
trn_tiles_burnt['dataset'] = 'trn'
val_tiles_burnt['dataset'] = 'val'


# In[117]:


trn_tiles_burnt.head()


# In[118]:


len(trn_tiles_burnt)


# In[119]:


len(val_tiles_burnt)


# In[120]:


# see if there's overlapping tiles between trn and val
fig, ax = plt.subplots(figsize=(10,10))
trn_tiles.plot(ax=ax, color='grey', alpha=0.5, edgecolor='red')
val_tiles.plot(ax=ax, color='grey', alpha=0.5, edgecolor='blue')


# In[121]:


# see if there's overlapping tiles between trn and val
fig, ax = plt.subplots(figsize=(10,10))
trn_tiles_burnt.plot(ax=ax, color='grey', alpha=0.5, edgecolor='blue')
val_tiles_burnt.plot(ax=ax, color='grey', alpha=0.5, edgecolor='red')


# In[122]:


len(trn_tiles_burnt)


# In[123]:


len(val_tiles_burnt)


# In[124]:


ignore_tiles, val_tiles_burnt = np.split(val_tiles_burnt.sample(frac=1), [int(.9*len(val_tiles_burnt))])


# In[125]:


len(val_tiles_burnt)


# In[126]:


# merge into one gdf to keep all trn tiles while dropping overlapping/duplicate val tiles
tiles_gdf = gpd.GeoDataFrame(pd.concat([val_tiles_burnt, trn_tiles_burnt], ignore_index=True), crs=trn_tiles_burnt.crs)


# In[127]:


tiles_gdf.head()


# In[128]:


tiles_gdf.drop_duplicates(subset=['id'], inplace=True)


# In[129]:


len(tiles_gdf)


# In[130]:


trn_tiles_burnt.crs


# In[131]:


# check that there's no more overlapping tiles between trn and val
fig, ax = plt.subplots(figsize=(10,10))
tiles_gdf[tiles_gdf['dataset'] == 'trn'].plot(ax=ax, color='grey', edgecolor='red', alpha=0.5)
tiles_gdf[tiles_gdf['dataset'] == 'val'].plot(ax=ax, color='grey', edgecolor='blue', alpha=0.5)


# In[132]:


tiles_gdf.head()


# In[133]:


# convert 'id' string to list of ints for z,x,y
def reformat_xyz(tile_gdf):
  tile_gdf['xyz'] = tile_gdf.id.apply(lambda x: x.lstrip('(,)').rstrip('(,)').split(','))
  tile_gdf['xyz'] = [[int(q) for q in p] for p in tile_gdf['xyz']]
  return tile_gdf


# In[134]:


tiles_gdf = reformat_xyz(tiles_gdf)
tiles_gdf.head()


# In[135]:


len(tiles_gdf.loc[tiles_gdf['dataset'] == 'trn'])


# In[136]:


len(tiles_gdf.loc[tiles_gdf['dataset'] == 'val'])


# ## Load slippy map tile image from COG with rio-tiler and corresponding label with geopandas
# 
# Now we'll use  [rio-tiler](https://github.com/cogeotiff/rio-tiler) and the slippy map tile polygons generated by supermercado to test load a single 256x256 pixel tile from our col_borde_rural_001 COG image file. We will also load the col_borde_rural_001 geoJSON labels into a geopandas GeoDataFrame and crop  the building geometries to only those that intersect the bounds of the tile image.
# 
# Here is a great intro to COGs, rio-tiler, and exciting developments in the cloud-native geospatial toolbox by [Vincent Sarago](https://medium.com/@_VincentS_) of [Development Seed](https://developmentseed.org/): https://medium.com/devseed/cog-talk-part-1-whats-new-941facbcd3d1
# 
# We'll then create our corresponding 3-channel RGB mask by passing these cropped geometries to solaris' df_to_px_mask function. Pixel value of 255 in the generated mask: 
# 
# - in the 1st (Red) channel represent building footprints, 
# - in the 2nd (Green) channel represent building boundaries (visually looks yellow on the RGB mask display because the pixels overlap red and green+red=yellow), 
# - and in the 3rd (Blue) channel represent close contact points between adjacent buildings

# In[137]:


idx = 22
tiles_gdf.iloc[idx]['xyz']


# In[138]:


tif_url = DATA_TIFF


# tile_size default is 256

# In[139]:


#inputs required for: tile(address, tile_x, tile_y, tile_z, tilesize=256, **kwargs)


# In[140]:


tile, mask = rt_main.tile(tif_url, *tiles_gdf.iloc[idx]['xyz'], tilesize=tile_size)


# In[141]:


plt.imshow(np.moveaxis(tile,0,2))


# In[142]:


# redisplay our labeled geojson file
label_df.plot(figsize=(10,10))


# In[143]:


# get the geometries from the geodataframe
all_polys = label_df.geometry


# In[144]:


# preemptively fix and merge any invalid or overlapping geoms that would otherwise throw errors during the rasterize step. 
# TODO: probably a better way to do this

# https://gis.stackexchange.com/questions/271733/geopandas-dissolve-overlapping-polygons
# https://nbviewer.jupyter.org/gist/rutgerhofste/6e7c6569616c2550568b9ce9cb4716a3

def explode(gdf):
    """    
    Will explode the geodataframe's muti-part geometries into single 
    geometries. Each row containing a multi-part geometry will be split into
    multiple rows with single geometries, thereby increasing the vertical size
    of the geodataframe. The index of the input geodataframe is no longer
    unique and is replaced with a multi-index. 

    The output geodataframe has an index based on two columns (multi-index) 
    i.e. 'level_0' (index of input geodataframe) and 'level_1' which is a new
    zero-based index for each single part geometry per multi-part geometry
    
    Args:
        gdf (gpd.GeoDataFrame) : input geodataframe with multi-geometries
        
    Returns:
        gdf (gpd.GeoDataFrame) : exploded geodataframe with each single 
                                 geometry as a separate entry in the 
                                 geodataframe. The GeoDataFrame has a multi-
                                 index set to columns level_0 and level_1
        
    """
    gs = gdf.explode()
    gdf2 = gs.reset_index().rename(columns={0: 'geometry'})
    gdf_out = gdf2.merge(gdf.drop('geometry', axis=1), left_on='level_0', right_index=True)
    gdf_out = gdf_out.set_index(['level_0', 'level_1']).set_geometry('geometry')
    gdf_out.crs = gdf.crs
    return gdf_out

def cleanup_invalid_geoms(all_polys):
  all_polys_merged = gpd.GeoDataFrame()
  all_polys_merged['geometry'] = gpd.GeoSeries(cascaded_union([p.buffer(0) for p in all_polys]))

  gdf_out = explode(all_polys_merged)
  gdf_out = gdf_out.reset_index()
  gdf_out.drop(columns=['level_0','level_1'], inplace=True)
  all_polys = gdf_out['geometry']
  return all_polys

all_polys = cleanup_invalid_geoms(all_polys)


# In[145]:


# get the same tile polygon as our tile image above
tile_poly = tiles_gdf.iloc[idx]['geometry']
print(tile_poly.bounds)
tile_poly


# In[146]:


# get affine transformation matrix for this tile using rasterio.transform.from_bounds: https://rasterio.readthedocs.io/en/stable/api/rasterio.transform.html#rasterio.transform.from_bounds
tfm = from_bounds(*tile_poly.bounds, tile_size, tile_size) 
tfm


# In[147]:


DATA_TIFF


# In[148]:


epsg = str(rasterio.open(DATA_TIFF).meta['crs']).split(':')[1]


# In[149]:


#epsg='4326' #WGS84 no zone


# In[150]:


# crop col_borde_rural_001 geometries to what overlaps our tile polygon bounds
cropped_polys = [poly for poly in all_polys if poly.intersects(tile_poly)]
cropped_polys_gdf = gpd.GeoDataFrame(geometry=cropped_polys, crs={'init': f'epsg:{epsg}'})
cropped_polys_gdf.plot()


# In[151]:


# burn a footprint/boundary/contact 3-channel mask with solaris: https://solaris.readthedocs.io/en/latest/tutorials/notebooks/api_masks_tutorial.html

fbc_mask = sol.vector.mask.df_to_px_mask(df=cropped_polys_gdf,
                                         channels=['footprint', 'boundary', 'contact'],
                                         affine_obj=tfm, shape=(tile_size,tile_size),
                                         boundary_width=5, boundary_type='inner', contact_spacing=5, meters=True)


# In[152]:


fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10, 5))
ax1.imshow(np.moveaxis(tile,0,2))
ax2.imshow(fbc_mask)


# In[153]:


fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(10, 5))
ax1.imshow(fbc_mask[:,:,0])
ax2.imshow(fbc_mask[:,:,1])
ax3.imshow(fbc_mask[:,:,2])


# ## Make and save all the image and mask tiles
# 
# Now that we've successfully loaded one tile image from COG with rio-tiler and created its 3-channel RGB mask with solaris, let's generate our full training and validation datasets. 
# 
# We'll write some functions and loops to run through all of our `trn` and `val` tiles at `zoom_level=19` and save them as lossless `png` files in the appropriate folders with a filename schema of `{save_path}/{prefix}{z}_{x}_{y}` so we can easily identify and geolocate what tile each file represents.

# In[ ]:


def save_tile_img(tif_url, xyz, tile_size, save_path='', prefix='', display=False):
  x,y,z = xyz
  tile, mask = rt_main.tile(tif_url, x,y,z, tilesize=tile_size)
  if display: 
    plt.imshow(np.moveaxis(tile,0,2))
    plt.show()
    
  skimage.io.imsave(f'{save_path}/{prefix}{z}_{x}_{y}.png',np.moveaxis(tile,0,2), check_contrast=False) 


# In[ ]:


def save_tile_mask(labels_poly, tile_poly, xyz, tile_size, save_path='', prefix='', display=False):
  x,y,z = xyz
  tfm = from_bounds(*tile_poly.bounds, tile_size, tile_size) 
  
  cropped_polys = [poly for poly in labels_poly if poly.intersects(tile_poly)]
  cropped_polys_gdf = gpd.GeoDataFrame(geometry=cropped_polys, crs={'init': f'epsg:{epsg}'})
  
  fbc_mask = sol.vector.mask.df_to_px_mask(df=cropped_polys_gdf,
                                         channels=['footprint', 'boundary', 'contact'],
                                         affine_obj=tfm, shape=(tile_size,tile_size),
                                         boundary_width=5, boundary_type='inner', contact_spacing=5, meters=True)
  
  if display: plt.imshow(fbc_mask); plt.show()
  
  skimage.io.imsave(f'{save_path}/{prefix}{z}_{x}_{y}_mask.png',fbc_mask, check_contrast=False) 


# In[ ]:


tiles_gdf[tiles_gdf['dataset'] == 'trn'].shape, tiles_gdf[tiles_gdf['dataset'] == 'val'].shape


# In[ ]:


#needs to exist for code below
get_ipython().system('cp {DATA_TIFF} tmp.tif')


# In[ ]:


for idx, tile in tqdm(tiles_gdf.iterrows()):
    dataset = tile['dataset']
    save_tile_img('tmp.tif', tile['xyz'], tile_size, save_path=img_path, prefix=f'{DATASET}_001{dataset}_', display=False)


# In[ ]:


n = 40  #chunk row size
list_df = [tiles_gdf[i:i+n] for i in range(0,tiles_gdf.shape[0],n)]


# In[ ]:


def process_frame(df):
    for idx, tile in tqdm(tiles_gdf.iterrows()):
        dataset = tile['dataset']
        tile_poly = tile['geometry']
        save_tile_mask(all_polys, tile_poly, tile['xyz'], tile_size, save_path=mask_path,prefix=f'{DATASET}_001{dataset}_', display=False)


# In[ ]:


import multiprocessing as mp
pool = mp.Pool(8) # use 8 processes
funclist = []
for df in list_df:
    # process each data frame
    f = pool.apply_async(process_frame,[df])


# In[ ]:


# check that tile images and masks saved correctly
start_idx, end_idx = 200,205
for i,j in zip(sorted(img_path.iterdir())[start_idx:end_idx], sorted(mask_path.iterdir())[start_idx:end_idx]):
  fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,5))
  ax1.imshow(skimage.io.imread(i))
  ax2.imshow(skimage.io.imread(j))
  plt.show()


# In[ ]:


# compress and download
get_ipython().system("tar -czf f'{DATASET}'_001trn.tar.gz data")

