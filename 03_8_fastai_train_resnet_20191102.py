#!/usr/bin/env python
# coding: utf-8

# ## Ricap
# 
# Using rotated to hz + OpenCv border
# 
# Basic default transforms

# In[4]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


from fastai.vision import *
import geopandas as gpd
from resizeimage import resizeimage
import datetime
import uuid
from os import listdir
from os.path import isfile, join
import image_data_aug


# In[6]:


torch.cuda.set_device(0)
torch.cuda.current_device()


# In[7]:


data_dir = Path('data')
colombia_rural = Path('data/stac/colombia/borde_rural')
colombia_soacha = Path('data/stac/colombia/borde_soacha')


# In[8]:


guatemala_mixco1 = Path('data/stac/guatemala/mixco_1_and_ebenezer')
guatemala_mixco3 = Path('data/stac/guatemala/mixco_3')


# In[9]:


st_lucia_castries = Path('data/stac/st_lucia/castries')
st_lucia_dennery = Path('data/stac/st_lucia/dennery')
st_lucia_gros_islet = Path('data/stac/st_lucia/gros_islet')


# In[10]:


COUNTRY='colombia'
REGION='borde_rural'
DATASET = f'{COUNTRY}_{REGION}'
DATASET_PATH=colombia_rural
path=data_dir/f'{COUNTRY}_{REGION}/cropped/'
TRAIN_JSON = f'train-{REGION}.geojson'
TEST_JSON = f'test-{REGION}.geojson'


# In[11]:


RETRAIN = True
RESIZE_IMAGES = True


# In[50]:


MODEL_NAME='cv_reflect_101_ricap'


# In[52]:


NB_NUM='03_8_3'


# In[14]:


DATE = datetime.datetime.today().strftime('%Y%m%d')
UID=str(uuid.uuid4())[:8]
print(f'UID: {UID}, DATE: {DATE}')                                                 


# In[15]:


#DATE = '20191026'
#UID = '1964891c'


# In[16]:


SUB_NUM='9'


# In[17]:


img_size=256
bs = 64


# In[18]:


train_images=data_dir/f'train/rotated/clipped/{img_size}'
test_images=data_dir/f'test/rotated/clipped/{img_size}'


# In[19]:


test_names = get_image_files(test_images)


# In[20]:


assert len(test_names)==7325


# In[21]:


df_all=pd.read_csv(data_dir/'df_train_all.csv')


# In[22]:


df_all.tail()


# In[23]:


df_test=pd.read_csv(data_dir/'df_test_all.csv')


# In[24]:


df_test.tail()


# In[25]:


assert len(df_test)==7325


# In[26]:


df_all.loc[df_all['id'] == '7a204ec4']


# In[27]:


len(df_all)


# To put this in a `DataBunch` while using the [data block API](https://docs.fast.ai/data_block.html), we then need to using `ImageList` (and not `ImageDataBunch`). This will make sure the model created has the proper loss function to deal with the multiple classes.

# In[28]:


tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)


# ### setup dataset

# In[29]:


np.random.seed(42)
dep_var='roof_material'
src = (ImageList.from_df(path=train_images, df=df_all, cols='id', suffix='.tif')
       .split_by_rand_pct(0.2)
      .label_from_df(cols=dep_var)
      .add_test_folder(test_images))


# In[30]:


data = (src.transform(tfms, size=img_size)
        .databunch(bs=bs).normalize(imagenet_stats))


# In[31]:


#to check what params object has
#dir(data)


# In[32]:


data.label_list


# In[33]:


data.loss_func


# `show_batch` still works, and show us the different labels separated by `;`.

# In[34]:


data.show_batch(rows=3, figsize=(12,9))


# ### Model

# In[35]:


arch = models.resnet50
arch_name = 'rn50'


# Without mixup:
# <pre>
# epoch 	train_loss 	valid_loss 	error_rate 	time
# 0 	0.508273 	1.170024 	0.339911 	01:16
# 1 	0.471699 	0.494651 	0.166297 	01:22
# 2 	0.439157 	0.417086 	0.152328 	01:23
# 3 	0.402381 	0.370002 	0.127938 	01:23
# 4 	0.355355 	0.360440 	0.126608 	01:22
#     </pre>

# #### Ricap

# In[46]:


learn = cnn_learner(data, arch, metrics=error_rate, bn_final=True).to_fp16().ricap()


# In[42]:


learn.lr_find()


# In[43]:


learn.recorder.plot()


# In[47]:


lr = 1e-2


# In[48]:


learn.fit_one_cycle(5, slice(lr))


# In[53]:


print(f'stage-1-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}')


# In[54]:


learn.save(f'stage-1-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}')


# #### Load model

# In[55]:


learn.load(f'stage-1-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}')


# In[62]:


learn.callbacks


# In[56]:


interp = ClassificationInterpretation.from_learner(learn)
interp.most_confused(min_val=2)


# In[57]:


interp.plot_confusion_matrix(figsize=(10,10), dpi=60)


# ### Re-train

# In[58]:


learn.unfreeze()


# In[59]:


learn.lr_find()
learn.recorder.plot()


# In[60]:


learn.fit_one_cycle(5, slice(5e-6, lr/5))


# In[73]:


learn.save(f'stage-2-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}')


# In[32]:


learn.load(f'stage-2-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}')


# In[33]:


learn.fit_one_cycle(5, slice(1e-5, lr/5))


# In[34]:


learn.recorder.plot_losses()


# In[35]:


learn.save(f'stage-2-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}')


# In[39]:


learn.export(f'stage-2-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}.pkl')


# In[72]:


learn.export()


# ### Larger size images

# In[45]:


data = (src.transform(tfms, size=256)
        .databunch().normalize(imagenet_stats))

learn.data = data
data.train_ds[0][0].shape


# In[46]:


learn.freeze()


# In[47]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


lr=1e-2/2


# In[ ]:


learn.fit_one_cycle(5, slice(lr))


# In[ ]:


learn.save(f'stage-1-256-{arch_name}-{MODEL_NAME}-{DATE}-{UID}')


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

# In[73]:


#test_images=data_dir/f'test/rotated/clipped/{img_size}'
test_dataset=ImageList.from_folder(test_images)


# In[74]:


len(test_dataset)


# In[88]:


learn = load_learner(path=data_dir/f'train/rotated/clipped/{img_size}', test=test_dataset)


# In[89]:


#learn = load_learner(path=data_dir/f'train/rotated/clipped/{img_size}', file=f'stage-2-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}.pkl', test=test_dataset)


# In[90]:


learn.data.loss_func


# In[91]:


type(learn.data)


# In[92]:


type(learn.dl(DatasetType.Test))


# In[94]:


len(learn.dl(DatasetType.Test))


# Get number of items in the Valid dataset (in DeviceDataLoader)

# In[95]:


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

# In[96]:


preds,y= learn.get_preds(ds_type=DatasetType.Test)


# In[97]:


labels = np.argmax(preds, 1)


# In[98]:


len(preds)


# In[99]:


preds[0].tolist()


# In[100]:


preds_list=[]
for pred in preds:
    preds_list.append(pred.tolist())


# In[101]:


len(labels)


# In[102]:


learn.data.classes


# In[103]:


data.classes


# In[104]:


test_predictions = [learn.data.classes[int(x)] for x in labels]


# In[105]:


test_predictions[0]


# In[106]:


type(learn.data.test_ds)


# In[107]:


learn.data.test_ds.x.items


# In[108]:


ids=[]
for item in learn.data.test_ds.x.items:
    base, id = os.path.split(item)
    id = id.split('.tif')[0]
    ids.append(id)


# In[109]:


preds_list[0]


# In[110]:


cols = learn.data.classes.copy()
cols.insert(0,'id')
df = pd.DataFrame(list(zip(ids, preds_list)), 
               columns =['id', 'pred']) 


# In[111]:


cols


# In[112]:


df.head()


# In[113]:


pred_df = pd.DataFrame(df['pred'].values.tolist())


# In[114]:


pred_df.insert(loc=0, column='id', value=ids)


# In[115]:


pred_df.columns = cols


# In[116]:


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

# In[118]:





# In[138]:


pred_df = pd.read_csv(data_dir/f'submissions/stage-2-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}.csv')


# In[139]:


pred_df.head()


# In[149]:


pred_ids=pred_df['id'].values.tolist()


# In[119]:


df_baseline = pd.read_csv(data_dir/f'submissions/mean_baseline.csv')


# In[129]:


df_baseline.head()


# In[146]:


baseline_ids=df_baseline['id'].values.tolist()


# In[147]:


baseline_ids


# In[150]:


assert set(pred_ids)==set(baseline_ids)


# #### sort by baseline ids

# In[151]:


pred_df['id_cat'] = pd.Categorical(
    pred_df['id'], 
    categories=baseline_ids, 
    ordered=True
)


# In[152]:


pred_df.head()


# In[153]:


pred_df=pred_df.sort_values('id_cat')


# In[154]:


pred_df.head()


# In[156]:


pred_df.drop(columns=['id_cat'],inplace=True)


# In[157]:


pred_df.to_csv(data_dir/f'submissions/stage-2-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}.csv', index=False)


# In[1]:


### Submission 2: 0.4461


# In[16]:


arch_name = 'rn50'
pred_df=pd.read_csv(data_dir/f'submissions/stage-2-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}.csv')


# In[17]:


pred_df.drop(columns=['id'],inplace=True)
classes=pred_df.idxmax(axis=1)
pd.value_counts(classes).plot(kind="bar")


# In[ ]:




