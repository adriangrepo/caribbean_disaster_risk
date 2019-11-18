#!/usr/bin/env python
# coding: utf-8

# ## Train and Prediction on all data
# 
# Using rotated to hz + OpenCv border
# 
# Basic default transforms

# In[2]:



# In[3]:


from fastai.vision import *
import geopandas as gpd
from resizeimage import resizeimage
import datetime
import uuid
from os import listdir
from os.path import isfile, join


# In[4]:


torch.cuda.set_device(0)
torch.cuda.current_device()


# In[5]:


data_dir = Path('data')


# In[6]:


RETRAIN = True
RESIZE_IMAGES = True


# In[7]:


MODEL_NAME='cv_reflect_101_valid_cf_verified'


# In[8]:


NB_NUM='03_12'


# In[18]:


border='reflect'


# In[9]:


DATE = datetime.datetime.today().strftime('%Y%m%d')
UID=str(uuid.uuid4())[:8]
print(f'UID: {UID}, DATE: {DATE}')                                                 


# In[10]:


#DATE = '20191109'
#UID = '123cca5f'


# In[11]:


SUB_NUM='17'


# In[12]:


img_size=256


# In[13]:


train_images=data_dir/f'train/rotated/clipped/reflect/{img_size}'
test_images=data_dir/f'test/rotated/clipped/reflect/{img_size}'


# In[14]:


test_names = get_image_files(test_images)


# In[15]:


assert len(test_names)==7325


# In[16]:


df_all=pd.read_csv(data_dir/'df_train_all.csv')


# In[17]:


df_valid=df_all.loc[df_all['verified'] == True]


# In[ ]:


df_invalid=df_all.loc[df_all['verified'] == False]


# In[18]:


df_test=pd.read_csv(data_dir/'df_test_all.csv')


# In[19]:


df_test.tail()


# In[20]:


assert len(df_test)==7325


# In[21]:


df_valid.loc[df_valid['id'] == '7a204ec4']


# In[22]:


len(df_valid)


# To put this in a `DataBunch` while using the [data block API](https://docs.fast.ai/data_block.html), we then need to using `ImageList` (and not `ImageDataBunch`). This will make sure the model created has the proper loss function to deal with the multiple classes.

# In[23]:


tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)


# ### setup dataset

# In[24]:


np.random.seed(42)
dep_var='roof_material'
src = (ImageList.from_df(path=train_images, df=df_valid, cols='id', suffix='.tif')
       .split_by_rand_pct(0.2)
      .label_from_df(cols=dep_var)
      .add_test_folder(test_images))


# In[25]:


data = (src.transform(tfms, size=img_size)
        .databunch().normalize(imagenet_stats))


# In[26]:


#to check what params object has
#dir(data)


# In[27]:


data.label_list


# In[28]:


data.loss_func


# `show_batch` still works, and show us the different labels separated by `;`.

# In[29]:


data.show_batch(rows=3, figsize=(12,9))


# ### Model 
# 
# Load a pre-trained model using only valid data, then predict on non verified images and keep those that match

# In[30]:


arch = models.resnet50
arch_name = 'rn50'


# In[31]:


learn = cnn_learner(data, arch, metrics=error_rate, bn_final=True)


# In[ ]:


P_DATE = '20191109'
P_UID = '123cca5f'
P_NB_NUM='03_11'
P_MODEL_NAME='cv_reflect_101_valid'
P_arch_name = 'rn50'


# In[ ]:


learn.load(f'stage-2-{P_arch_name}-{P_NB_NUM}-{P_MODEL_NAME}-{P_DATE}-{P_UID}')


# In[ ]:





# ### inference

# In[ ]:


test_dataset=ImageList.from_df(df=df_invalid, cols='id', folder=train_images, suffix:str='', **kwargs) → ItemList


# In[64]:


#test_images=data_dir/f'test/rotated/clipped/{img_size}'
test_dataset=ImageList.from_folder(test_images)


# In[65]:


len(test_dataset)


# In[66]:


learn = load_learner(path=data_dir/f'train/rotated/clipped/{img_size}', test=test_dataset)


# In[67]:


#learn = load_learner(path=data_dir/f'train/rotated/clipped/{img_size}', file=f'stage-2-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}.pkl', test=test_dataset)


# In[68]:


learn.data.loss_func


# In[69]:


type(learn.data)


# In[70]:


type(learn.dl(DatasetType.Test))


# In[71]:


len(learn.dl(DatasetType.Test))


# Get number of items in the Valid dataset (in DeviceDataLoader)

# In[72]:


#assert len(learn.dl(DatasetType.Test).dl)==7325


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

# In[73]:


preds,y= learn.get_preds(ds_type=DatasetType.Test)


# In[74]:


labels = np.argmax(preds, 1)


# In[75]:


len(preds)


# In[76]:


preds[0].tolist()


# In[77]:


preds_list=[]
for pred in preds:
    preds_list.append(pred.tolist())


# In[78]:


len(labels)


# In[79]:


learn.data.classes


# In[80]:


data.classes


# In[81]:


test_predictions = [learn.data.classes[int(x)] for x in labels]


# In[82]:


test_predictions[0]


# In[83]:


type(learn.data.test_ds)


# In[84]:


learn.data.test_ds.x.items


# In[85]:


ids=[]
for item in learn.data.test_ds.x.items:
    base, id = os.path.split(item)
    id = id.split('.tif')[0]
    ids.append(id)


# In[86]:


preds_list[0]


# In[87]:


cols = learn.data.classes.copy()
cols.insert(0,'id')
df = pd.DataFrame(list(zip(ids, preds_list)), 
               columns =['id', 'pred']) 


# In[88]:


cols


# In[89]:


df.head()


# In[90]:


pred_df = pd.DataFrame(df['pred'].values.tolist())


# In[91]:


pred_df.insert(loc=0, column='id', value=ids)


# In[92]:


pred_df.columns = cols


# In[93]:


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

# In[94]:


pred_ids=pred_df['id'].values.tolist()


# In[95]:


df_baseline = pd.read_csv(data_dir/f'submissions/mean_baseline.csv')


# In[96]:


df_baseline.head()


# In[97]:


baseline_ids=df_baseline['id'].values.tolist()


# In[98]:


assert set(pred_ids)==set(baseline_ids)


# #### sort by baseline ids

# In[99]:


pred_df['id_cat'] = pd.Categorical(
    pred_df['id'], 
    categories=baseline_ids, 
    ordered=True
)


# In[100]:


pred_df.head()


# In[101]:


pred_df=pred_df.sort_values('id_cat')


# In[102]:


pred_df.head()


# In[103]:


pred_df.drop(columns=['id_cat'],inplace=True)


# In[104]:


pred_df.to_csv(data_dir/f'submissions/stage-2-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}.csv', index=False)


# In[105]:


### Submission 2: 0.4687


# In[106]:


arch_name = 'rn50'
pred_df=pd.read_csv(data_dir/f'submissions/stage-2-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}.csv')


# In[107]:


pred_df.drop(columns=['id'],inplace=True)
classes=pred_df.idxmax(axis=1)
pd.value_counts(classes).plot(kind="bar")


# In[ ]:




