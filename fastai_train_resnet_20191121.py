#!/usr/bin/env python
# coding: utf-8

# ## Prediction 

# In[1]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from fastai.vision import *
import geopandas as gpd
from resizeimage import resizeimage


# In[3]:


data_dir = Path('data')
colombia_rural = Path('data/stac/colombia/borde_rural')
COUNTRY='colombia'
REGION='borde_rural'
DATASET = f'{COUNTRY}_{REGION}'
DATASET_PATH=colombia_rural
path=data_dir/f'{COUNTRY}_{REGION}/cropped/'
TRAIN_JSON = f'train-{REGION}.geojson'
TEST_JSON = f'test-{REGION}.geojson'


# In[4]:


RETRAIN = True
RESIZE_IMAGES = False


# In[5]:


label_df = gpd.read_file(DATASET_PATH/TRAIN_JSON)
label_df.head()


# To put this in a `DataBunch` while using the [data block API](https://docs.fast.ai/data_block.html), we then need to using `ImageList` (and not `ImageDataBunch`). This will make sure the model created has the proper loss function to deal with the multiple classes.

# In[6]:


tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)


# get average image size

# In[7]:


import PIL.Image as pil_image
def img_type_av_sz(fn_list):
    widths = []
    heights = []
    for im in fn_list:
        #100 x faster than open_image(img_f)
        w,h=pil_image.open(im).size
        widths.append(w)
        heights.append(h)
    av_w = sum(widths)/len(widths)
    av_h = sum(heights)/len(heights)
    print(f'avg width: {av_w}, avg height: {av_h}, max w: {max(widths)}, min w: {min(widths)}, max h: {max(heights)}, min h: {min(heights)}')
    return widths, heights


# In[8]:


fnames = get_image_files(path/'train')


# In[9]:


fnames[0]


# In[10]:


test_fnames = get_image_files(path/'test')


# In[11]:


widths, heights=img_type_av_sz(fnames)


# In[12]:


num_bins = 100
n, bins, patches = plt.hist(widths, num_bins, facecolor='blue', alpha=0.5)
plt.show()


# In[13]:


num_bins = 100
n, bins, patches = plt.hist(heights, num_bins, facecolor='red', alpha=0.5)
plt.show()


# In[14]:


def resize_to_max(fn_list, max_w, max_h):
    widths = []
    heights = []
    for im in fn_list:
        with pil_image.open(im) as image:
            w,h=image.size
            if (w > max_w) or (h > max_h):
                res_im = resizeimage.resize_contain(image, [max_w, max_h])
                name=str(im).split('.')[0]
                res_im.save(f'{name}_512.tif', image.format)


# In[15]:


def resize_all(fn_list, max_w, max_h, data_path):
    widths = []
    heights = []
    for im in fn_list:
        with pil_image.open(im) as image:
            w,h=image.size
            res_im = resizeimage.resize_contain(image, [max_w, max_h])
            name=str(im).split('.')[0]
            name=name.split('/')[-1]
            res_im.save(f'{data_path}/{name}.tif', image.format)


# ### resize all images

# In[16]:


if RESIZE_IMAGES:
    resize_to_max(fnames, 512,512)


# In[17]:


if RESIZE_IMAGES:
    resize_all(fnames, 256,256, data_path=f'{path}/train/256')
    resize_all(test_fnames, 256,256, data_path=f'{path}/test/256')


# ### setup dataset

# In[81]:


np.random.seed(42)
dep_var='roof_material'
src = (ImageList.from_df(path=path/'train/256', df=label_df, suffix='.tif')
       .split_by_rand_pct(0.2)
      .label_from_df(cols=dep_var)
      .add_test_folder(path/'test/256'))


# In[19]:


data = (src.transform(tfms, size=256)
        .databunch().normalize(imagenet_stats))


# In[95]:


data.c


# In[43]:


type(data)


# In[20]:


data.loss_func


# `show_batch` still works, and show us the different labels separated by `;`.

# In[21]:


data.show_batch(rows=3, figsize=(12,9))


# In[22]:


arch = models.resnet50


# In[23]:


learn = cnn_learner(data, arch, metrics=error_rate, bn_final=True)


# In[24]:


learn.model = torch.nn.DataParallel(learn.model, device_ids=[0, 1, 2])


# We use the LR Finder to pick a good learning rate.

# ### (Re)train model

# In[25]:


if RETRAIN:
    learn.lr_find()


# In[26]:


if RETRAIN:
    learn.recorder.plot()


# Then we can fit the head of our network.

# In[27]:


if RETRAIN:
    lr = 1e-1


# In[28]:


if RETRAIN:
    learn.fit_one_cycle(1, slice(lr))


# In[29]:


if RETRAIN:
    learn.save('stage-1-rn50')
    #saves in parent of models directory
    learn.export()


# #### Load model

# In[31]:


if RETRAIN:
    learn.load('stage-1-rn50')


# ### inference

# In[77]:


test=ImageList.from_folder(path/'test/256')


# In[44]:


learn = load_learner(path/'train/256', test=test)


# In[45]:


learn.data.loss_func


# In[46]:


type(learn.data)


# In[47]:


type(learn.dl(DatasetType.Test))


# Get number of items in the Valid dataset (in DeviceDataLoader)

# In[48]:


len(learn.dl(DatasetType.Test).dl)


# Required format:
#     
# <pre>
# id	concrete_cement	healthy_metal	incomplete	irregular_metal	other
# 7a4d630a	0.9	0	0	0	0
# 7a4bbbd6	0.9	0	0	0	0
# 7a4ac744	0.9	0	0	0	0
# 7a4881fa	0.9	0	0	0	0
# 7a4aa4a8	0.9	0	0	0	0
# </pre>
# 

# In[76]:


preds,y= learn.get_preds(ds_type=DatasetType.Test)


# In[65]:


labels = np.argmax(preds, 1)


# In[50]:


len(preds)


# In[93]:


preds[0].tolist()


# In[94]:


preds_list=[]
for pred in preds:
    preds_list.append(pred.tolist())


# In[67]:


labels


# In[84]:


test_predictions = [learn.data.classes[int(x)] for x in labels]


# In[85]:


test_predictions[0]


# In[86]:


type(learn.data.test_ds)


# In[74]:


learn.data.test_ds.x.items


# In[90]:


ids=[]
for item in learn.data.test_ds.x.items:
    base, id = os.path.split(item)
    id = id.split('.tif')[0]
    ids.append(id)


# In[ ]:


df = pd.DataFrame(list(zip(ids, preds_list)), 
               columns =['id', 'val']) 


# In[70]:


#1x1 prediction
#img = data.train_ds[0][0]
#returns y, pred, raw_pred
#learn.predict(img)


# ### Re-train

# Required format:
#     
# <pre>
# id	concrete_cement	healthy_metal	incomplete	irregular_metal	other
# 7a4d630a	0.9	0	0	0	0
# 7a4bbbd6	0.9	0	0	0	0
# 7a4ac744	0.9	0	0	0	0
# 7a4881fa	0.9	0	0	0	0
# 7a4aa4a8	0.9	0	0	0	0
# </pre>
# 

# In[56]:


learn.unfreeze()


# In[57]:


learn.lr_find()
learn.recorder.plot()


# In[58]:


learn.fit_one_cycle(5, slice(1e-5, lr/5))


# In[59]:


learn.save('stage-2-rn50')


# In[20]:


learn.load('stage-2-rn50')


# In[60]:


data = (src.transform(tfms, size=256)
        .databunch().normalize(imagenet_stats))

learn.data = data
data.train_ds[0][0].shape


# In[61]:


learn.freeze()


# In[62]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


lr=1e-2/2


# In[ ]:


learn.fit_one_cycle(5, slice(lr))


# In[ ]:


learn.save('stage-1-256-rn50')


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.fit_one_cycle(5, slice(1e-5, lr/5))


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


learn.save('stage-2-256-rn50')


# You won't really know how you're going until you submit to Kaggle, since the leaderboard isn't using the same subset as we have for training. But as a guide, 50th place (out of 938 teams) on the private leaderboard was a score of `0.930`.

# In[ ]:


learn.export()

