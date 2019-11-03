#!/usr/bin/env python
# coding: utf-8

# ## Train and Prediction on all data
# 
# Using rotated to hz + OpenCv border
# 
# Additional transforms

# In[1]:




# In[64]:


from fastai.vision import *
from fastai import *
import geopandas as gpd
from resizeimage import resizeimage
import datetime
import uuid


# In[65]:


#print(fastai.__version__)


# In[3]:


torch.cuda.set_device(1)
torch.cuda.current_device()


# In[4]:


data_dir = Path('data')
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


COUNTRY='colombia'
REGION='borde_rural'
DATASET = f'{COUNTRY}_{REGION}'
DATASET_PATH=colombia_rural
path=data_dir/f'{COUNTRY}_{REGION}/cropped/'
TRAIN_JSON = f'train-{REGION}.geojson'
TEST_JSON = f'test-{REGION}.geojson'


# In[8]:


RETRAIN = True
RESIZE_IMAGES = True


# In[9]:


MODEL_NAME='cv_reflect_101'


# In[10]:


DATE = datetime.datetime.today().strftime('%Y%m%d')
UID=str(uuid.uuid4())[:8]
print(f'UID: {UID}, DATE: {DATE}')                                                 


# In[11]:


SUB_NUM='2'


# In[12]:


img_size=256


# In[13]:


train_images=data_dir/f'train/rotated/clipped/{img_size}'
test_images=data_dir/f'test/rotated/clipped/{img_size}'


# In[14]:


df_all=pd.read_csv(data_dir/'df_train_all.csv')


# In[15]:


df_all.tail()


# In[16]:


df_test=pd.read_csv(data_dir/'df_test_all.csv')


# In[17]:


df_test.tail()


# In[18]:


df_all.loc[df_all['id'] == '7a204ec4']


# In[19]:


len(df_all)


# To put this in a `DataBunch` while using the [data block API](https://docs.fast.ai/data_block.html), we then need to using `ImageList` (and not `ImageDataBunch`). This will make sure the model created has the proper loss function to deal with the multiple classes.

# In[20]:


p_affine = 0.05
#xtra_tfms = [rotate(degrees=(-95, -85), p=p_affine), rotate(degrees=(95, 85), p=p_affine), zoom_crop(scale=(0.9,1.2), do_rand=True, p=0.05]


# In[30]:


#xtra_tfms = [zoom_crop(scale=(0.9,1.2), do_rand=True, p=0.05), cutout(n_holes=(1,4), length=(5, 30), p=.05)]


# In[35]:


xtra_tfms = [cutout(n_holes=(1,4), length=(5, 30), p=.05)]


# In[36]:


tfms = get_transforms(flip_vert=True, max_rotate=180, max_lighting=0.1, max_zoom=1.05, max_warp=0., xtra_tfms=xtra_tfms)


# ### setup dataset

# In[37]:


np.random.seed(42)
dep_var='roof_material'
src = (ImageList.from_df(path=train_images, df=df_all, cols='id', suffix='.tif')
       .split_by_rand_pct(0.2)
      .label_from_df(cols=dep_var)
      .add_test_folder(test_images))


# In[38]:


data = (src.transform(tfms, size=img_size)
        .databunch().normalize(imagenet_stats))


# In[39]:


#to check what params object has
#dir(data)


# In[40]:


data.label_list


# In[41]:


data.loss_func


# `show_batch` still works, and show us the different labels separated by `;`.

# In[42]:


data.show_batch(rows=3, figsize=(12,9))


# ### Model

# In[43]:


arch = models.resnet50
arch_name = 'rn50'


# In[44]:


learn = cnn_learner(data, arch, metrics=error_rate, bn_final=True)


# In[45]:


#learn.model = torch.nn.DataParallel(learn.model, device_ids=[0, 1, 2])


# We use the LR Finder to pick a good learning rate.

# ### (Re)train model

# In[46]:


if RETRAIN:
    learn.lr_find()


# In[47]:


if RETRAIN:
    learn.recorder.plot()


# Then we can fit the head of our network.

# In[48]:


if RETRAIN:
    lr = 1e-1


# In[49]:


if RETRAIN:
    learn.fit_one_cycle(5, slice(lr))


# In[50]:


if RETRAIN:
    learn.save(f'stage-1-{arch_name}-{MODEL_NAME}-{DATE}-{UID}')
    #saves in parent of models directory
    #learn.export()


# #### Load model

# In[51]:


if RETRAIN:
    learn.load(f'stage-1-{arch_name}-{MODEL_NAME}-{DATE}-{UID}')


# In[52]:


interp = ClassificationInterpretation.from_learner(learn)
interp.most_confused(min_val=2)


# In[53]:


interp.plot_confusion_matrix(figsize=(10,10), dpi=60)


# ### Re-train

# In[56]:


learn.unfreeze()


# In[57]:


learn.lr_find()
learn.recorder.plot()


# In[58]:


learn.fit_one_cycle(5, slice(1e-5, lr/5))


# In[54]:


learn.save(f'stage-2-{arch_name}-{MODEL_NAME}-{DATE}-{UID}')


# In[55]:


learn.load(f'stage-2-{arch_name}-{MODEL_NAME}-{DATE}-{UID}')


# In[56]:


interp = ClassificationInterpretation.from_learner(learn)
interp.most_confused(min_val=2)


# In[57]:


interp.plot_confusion_matrix(figsize=(10,10), dpi=60)


# In[59]:


#### more training 


# In[71]:


torch.cuda.empty_cache()


# In[73]:


data = (src.transform(tfms, size=512)
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


learn.save(f'stage-1-512-{arch_name}-{MODEL_NAME}-{DATE}-{UID}')


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.fit_one_cycle(5, slice(1e-5, lr/5))


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


learn.save(f'stage-2-256-{arch_name}-{MODEL_NAME}-{DATE}-{UID})


# You won't really know how you're going until you submit to Kaggle, since the leaderboard isn't using the same subset as we have for training. But as a guide, 50th place (out of 938 teams) on the private leaderboard was a score of `0.930`.

# In[ ]:


learn.export()


# ### inference

# In[ ]:


test=ImageList.from_folder(path/f'test/{img_size}')


# In[ ]:


learn = load_learner(path/f'train/{img_size}', test=test)


# In[ ]:


learn.data.loss_func


# In[ ]:


type(learn.data)


# In[ ]:


type(learn.dl(DatasetType.Test))


# Get number of items in the Valid dataset (in DeviceDataLoader)

# In[ ]:


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

# In[ ]:


preds,y= learn.get_preds(ds_type=DatasetType.Test)


# In[ ]:


labels = np.argmax(preds, 1)


# In[ ]:


len(preds)


# In[ ]:


preds[0].tolist()


# In[ ]:


preds_list=[]
for pred in preds:
    preds_list.append(pred.tolist())


# In[ ]:


len(labels)


# In[ ]:


learn.data.classes


# In[ ]:


data.classes


# In[ ]:


test_predictions = [learn.data.classes[int(x)] for x in labels]


# In[ ]:


test_predictions[0]


# In[ ]:


type(learn.data.test_ds)


# In[ ]:


learn.data.test_ds.x.items


# In[ ]:


ids=[]
for item in learn.data.test_ds.x.items:
    base, id = os.path.split(item)
    id = id.split('.tif')[0]
    ids.append(id)


# In[ ]:


preds_list[0]


# In[ ]:


cols = learn.data.classes.copy()
cols.insert(0,'id')
df = pd.DataFrame(list(zip(ids, preds_list)), 
               columns =['id', 'pred']) 


# In[ ]:


cols


# In[ ]:


df.head()


# In[ ]:


pred_df = pd.DataFrame(df['pred'].values.tolist())


# In[ ]:


pred_df.insert(loc=0, column='id', value=ids)


# In[ ]:


pred_df.columns = cols


# In[ ]:


pred_df.head()


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
