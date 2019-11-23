#!/usr/bin/env python
# coding: utf-8

# ## Regression for area
# 
# Using rotated to hz + OpenCv border
# 
# Extra transforms
# 
# Padded BG data
# 
# added 90 training data

# In[1]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from fastai.vision import *
import geopandas as gpd
from resizeimage import resizeimage
import datetime
import uuid
from os import listdir
from os.path import isfile, join


# In[3]:


torch.cuda.set_device(0)
torch.cuda.current_device()


# In[4]:


data_dir = Path('data_02')
orig_dir = Path('data')


# In[5]:


RETRAIN = True
RESIZE_IMAGES = True


# In[6]:


MODEL_NAME='cv_reflect_101_valid'


# In[7]:


NB_NUM='09_6'


# In[8]:


DATE = datetime.datetime.today().strftime('%Y%m%d')
UID=str(uuid.uuid4())[:8]
print(f'UID: {UID}, DATE: {DATE}')                                                 


# In[9]:


DATE = '20191122'
UID = 'dac74e4a'


# In[10]:


SUB_NUM='20'


# In[11]:


#wrap or reflect
border='reflect'
#padded or padded_bg
pad_type='padded'
img_size=256


# In[12]:


train_images=data_dir/f'train/rotated/clipped/{border}/{img_size}'
test_images=data_dir/f'test/rotated/clipped/{border}/{img_size}'


# In[13]:


test_names = get_image_files(test_images)


# In[14]:


len(test_names)


# In[15]:


#assert len(test_names)==7325


# In[16]:


df_all=pd.read_csv(orig_dir/'df_train_all.csv')


# In[17]:


len(df_all)


# In[18]:


df_sl_cc=pd.read_csv(orig_dir/'st_lucia_castries_gold_concrete_cement.csv')


# In[19]:


df_sl_cc.head()


# In[20]:


df_valid=df_all.loc[df_all['verified'] == True]


# In[21]:


df_valid.head()


# In[22]:


df_sl_cc['roof_material'] = 'concrete_cement'
df_sl_cc['country'] = 'st_lucia'
df_sl_cc['region'] = 'castries'
#pseudo verified
df_sl_cc['verified'] = True


# In[23]:


df_valid=df_valid.append(df_sl_cc, ignore_index=True)


# In[24]:


df_test=pd.read_csv(orig_dir/'df_test_all.csv')


# In[25]:


df_test.tail()


# In[26]:


df_centroids_train=pd.read_csv(orig_dir/'df_centroids_train.csv')
df_centroids_test=pd.read_csv(orig_dir/'df_centroids_test.csv')


# In[27]:


df_centroids_train.head()


# In[28]:


df_centroids_train=df_centroids_train[~df_centroids_train['id'].str.contains("_")]


# In[29]:


df_centroids_train=df_centroids_train.drop_duplicates(subset=['id'])


# In[30]:


len(df_centroids_train)


# In[31]:


len(df_all)


# In[58]:


df_centroids_train.dtypes


# In[32]:


delts_list = list(set(df_all['id'].values.tolist()) - set(df_centroids_train['id'].values.tolist()))


# In[33]:


delts_list


# To put this in a `DataBunch` while using the [data block API](https://docs.fast.ai/data_block.html), we then need to using `ImageList` (and not `ImageDataBunch`). This will make sure the model created has the proper loss function to deal with the multiple classes.

# In[34]:


xtra_tfms=[dihedral(p=0.5), rand_crop(p=0.4), rand_zoom(scale=(1.,1.5),p=0.4)] 
tfms = get_transforms(flip_vert=True, max_lighting=0.2, max_warp=0., xtra_tfms=xtra_tfms)


#tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)


# ### setup dataset

# In[60]:


np.random.seed(42)
dep_var='area'
src = (ImageList.from_df(path=train_images, df=df_centroids_train, cols='id', suffix='.tif')
       .split_by_rand_pct(0.2)
      .label_from_df(cols='area', label_cls=FloatList)
      .add_test_folder(test_images))


# In[61]:


data = (src.transform(tfms, size=img_size)
        .databunch().normalize(imagenet_stats))


# In[48]:


data.loss_func


# `show_batch` still works, and show us the different labels separated by `;`.

# In[49]:


data.show_batch(rows=3, figsize=(12,9))


# ### Model

# In[50]:


arch = models.resnet50
arch_name = 'rn50'


# In[51]:


learn = cnn_learner(data, arch, metrics=error_rate, bn_final=True).to_fp16()


# In[52]:


#learn.model = torch.nn.DataParallel(learn.model, device_ids=[0, 1, 2])


# We use the LR Finder to pick a good learning rate.

# ### Train model

# In[53]:


learn.lr_find()


# In[54]:


learn.recorder.plot()


# Then we can fit the head of our network.

# In[56]:


lr = 1e-5


# In[57]:


learn.fit_one_cycle(5, slice(lr))


# In[92]:


learn.fit_one_cycle(5, slice(lr))


# In[63]:


print(f'stage-1-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}')


# In[64]:


learn.save(f'stage-1-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}')
#saves in parent of models directory
#learn.export()


# In[65]:


learn.load(f'stage-1-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}')


# In[66]:


learn.fit_one_cycle(3, slice(lr))


# In[67]:


learn.fit_one_cycle(2, slice(lr))


# In[41]:


learn.save(f'stage-1-1-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}')


# #### Load model

# In[42]:


learn.load(f'stage-1-1-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}')


# In[43]:


interp = ClassificationInterpretation.from_learner(learn)
interp.most_confused(min_val=2)


# In[44]:


interp.plot_confusion_matrix(figsize=(10,10), dpi=60)


# ### Re-train

# In[45]:


learn.unfreeze()


# In[46]:


learn.lr_find()
learn.recorder.plot()


# In[47]:


learn.fit_one_cycle(5, slice(2e-6, lr/5))


# In[48]:


learn.save(f'stage-2-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}')


# In[49]:


learn.load(f'stage-2-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}')


# In[50]:


learn.fit_one_cycle(5, slice(5e-7, lr/5))


# In[51]:


learn.recorder.plot_losses()


# In[52]:


learn.save(f'stage-2-1-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}')


# In[53]:


learn.load(f'stage-2-1-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}')


# In[54]:


learn=learn.to_fp32()


# In[55]:


learn.export()


# ### inference

# In[56]:


#test_images=data_dir/f'test/rotated/clipped/{img_size}'
test_dataset=ImageList.from_folder(test_images)


# In[57]:


len(test_dataset)


# In[58]:


learn = load_learner(path=data_dir/f'train/rotated/clipped/{pad_type}/{border}/{img_size}', test=test_dataset)


# In[59]:


learn.data.loss_func


# In[60]:


type(learn.data)


# In[61]:


type(learn.dl(DatasetType.Test))


# In[62]:


len(learn.dl(DatasetType.Test))


# Get number of items in the Valid dataset (in DeviceDataLoader)

# In[63]:


preds,y= learn.get_preds(ds_type=DatasetType.Test)


# In[64]:


labels = np.argmax(preds, 1)


# In[65]:


len(preds)


# In[66]:


preds[0].tolist()


# In[67]:


preds_list=[]
for pred in preds:
    preds_list.append(pred.tolist())


# In[68]:


len(labels)


# In[69]:


learn.data.classes


# In[70]:


data.classes


# In[71]:


test_predictions = [learn.data.classes[int(x)] for x in labels]


# In[72]:


test_predictions[0]


# In[73]:


type(learn.data.test_ds)


# In[74]:


learn.data.test_ds.x.items


# In[75]:


ids=[]
for item in learn.data.test_ds.x.items:
    base, id = os.path.split(item)
    id = id.split('.tif')[0]
    ids.append(id)


# In[76]:


preds_list[0]


# In[77]:


cols = learn.data.classes.copy()
cols.insert(0,'id')
df = pd.DataFrame(list(zip(ids, preds_list)), 
               columns =['id', 'pred']) 


# In[78]:


cols


# In[79]:


df.head()


# In[80]:


pred_df = pd.DataFrame(df['pred'].values.tolist())


# In[81]:


pred_df.insert(loc=0, column='id', value=ids)


# In[82]:


pred_df.columns = cols


# In[83]:


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

# In[84]:


pred_ids=pred_df['id'].values.tolist()


# In[85]:


df_baseline = pd.read_csv(data_dir/f'submissions/mean_baseline.csv')


# In[ ]:


df_baseline.head()


# In[ ]:


baseline_ids=df_baseline['id'].values.tolist()


# In[ ]:


assert set(pred_ids)==set(baseline_ids)


# #### sort by baseline ids

# In[ ]:


pred_df['id_cat'] = pd.Categorical(
    pred_df['id'], 
    categories=baseline_ids, 
    ordered=True
)


# In[ ]:


pred_df.head()


# In[ ]:


pred_df=pred_df.sort_values('id_cat')


# In[ ]:


pred_df.head()


# In[ ]:


pred_df.drop(columns=['id_cat'],inplace=True)


# In[ ]:


pred_df.to_csv(data_dir/f'submissions/stage-2-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}.csv', index=False)


# In[ ]:


### Submission 2: 0.4687


# In[ ]:


arch_name = 'rn50'
pred_df=pd.read_csv(data_dir/f'submissions/stage-2-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}.csv')


# In[ ]:


pred_df.drop(columns=['id'],inplace=True)
classes=pred_df.idxmax(axis=1)
pd.value_counts(classes).plot(kind="bar")


# In[ ]:




