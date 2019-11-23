#!/usr/bin/env python
# coding: utf-8

# ## Train and Prediction on all data
# 
# Using rotated to hz + OpenCv border
# 
# Basic default transforms

# In[1]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from fastai.vision import *
from fastai import *
import geopandas as gpd
from resizeimage import resizeimage
import datetime
import uuid
from os import listdir
from os.path import isfile, join


# In[3]:


torch.cuda.set_device(0)
torch.cuda.current_device()


# In[89]:


data_dir = Path('data')
data_dir_02 = Path('data_02')


# In[5]:


RETRAIN = True
RESIZE_IMAGES = True


# In[6]:


MODEL_NAME='cv_reflect_101_valid'


# In[7]:


NB_NUM='09_3'


# In[8]:


DATE = datetime.datetime.today().strftime('%Y%m%d')
UID=str(uuid.uuid4())[:8]
print(f'UID: {UID}, DATE: {DATE}')                                                 


# In[9]:


#DATE = '20191109'
#UID = '123cca5f'


# In[10]:


SUB_NUM='20'


# In[11]:


img_size=256
border='reflect'


# In[12]:


train_images=data_dir_02/f'train/rotated/clipped/{border}/{img_size}'
test_images=data_dir_02/f'test/rotated/clipped/{border}/{img_size}'


# In[13]:


test_names = get_image_files(test_images)


# In[15]:


len(test_names)


# In[16]:


df_all=pd.read_csv(data_dir/'df_train_all.csv')


# In[17]:


len(df_all)


# In[18]:


df_valid=df_all.loc[df_all['verified'] == True]


# In[19]:


df_sl_cc=pd.read_csv(data_dir/'st_lucia_castries_gold_concrete_cement.csv')


# In[20]:


df_sl_cc['roof_material'] = 'concrete_cement'
df_sl_cc['country'] = 'st_lucia'
df_sl_cc['region'] = 'castries'
#pseudo verified
df_sl_cc['verified'] = True


# In[21]:


df_valid=df_valid.append(df_sl_cc, ignore_index=True)


# In[22]:


df_test=pd.read_csv(data_dir/'df_test_all.csv')


# In[23]:


df_test.tail()


# In[24]:


assert len(df_test)==7325


# In[25]:


len(df_valid)


# In[62]:


df_colombia_borde_rural_train_xy=pd.read_csv(data_dir/f'df_colombia_borde_rural_train_centroids.csv')
df_colombia_borde_rural_train_xy.head()
df_colombia_borde_rural_train_xy['country'] = 'colombia'
df_colombia_borde_rural_train_xy['region'] = 'borde_rural'


# In[63]:


df_colombia_borde_rural_test_xy=pd.read_csv(data_dir/f'df_colombia_borde_rural_test_centroids.csv')
df_colombia_borde_rural_test_xy.head()
df_colombia_borde_rural_test_xy['country'] = 'colombia'
df_colombia_borde_rural_test_xy['region'] = 'borde_rural'


# In[64]:


df_colombia_borde_soacha_train_xy=pd.read_csv(data_dir/f'df_colombia_borde_soacha_train_centroids.csv')
df_colombia_borde_soacha_test_xy=pd.read_csv(data_dir/f'df_colombia_borde_soacha_test_centroids.csv')
df_colombia_borde_soacha_train_xy['country'] = 'colombia'
df_colombia_borde_soacha_train_xy['region'] = 'borde_soacha'
df_colombia_borde_soacha_test_xy['country'] = 'colombia'
df_colombia_borde_soacha_test_xy['region'] = 'borde_soacha'


# In[65]:


df_guatemala_mixco_1_and_ebenezer_train_xy=pd.read_csv(data_dir/f'df_guatemala_mixco_1_and_ebenezer_train_centroids.csv')
df_guatemala_mixco_1_and_ebenezer_test_xy=pd.read_csv(data_dir/f'df_guatemala_mixco_1_and_ebenezer_test_centroids.csv')
df_guatemala_mixco_1_and_ebenezer_train_xy['country'] = 'guatemala'
df_guatemala_mixco_1_and_ebenezer_train_xy['region'] = 'mixco_1_and_ebenezer'
df_guatemala_mixco_1_and_ebenezer_test_xy['country'] = 'guatemala'
df_guatemala_mixco_1_and_ebenezer_test_xy['region'] = 'mixco_1_and_ebenezer'


# In[66]:


df_guatemala_mixco_3_train_xy=pd.read_csv(data_dir/f'df_guatemala_mixco_3_test_centroids.csv')
df_guatemala_mixco_3_test_xy=pd.read_csv(data_dir/f'df_guatemala_mixco_3_test_centroids.csv')
df_guatemala_mixco_3_train_xy['country'] = 'guatemala'
df_guatemala_mixco_3_train_xy['region'] = 'mixco_3'
df_guatemala_mixco_3_test_xy['country'] = 'guatemala'
df_guatemala_mixco_3_test_xy['region'] = 'mixco_3'


# In[67]:


df_st_lucia_castries_train_xy=pd.read_csv(data_dir/f'df_st_lucia_castries_train_centroids.csv')
df_st_lucia_castries_train_xy['country'] = 'st_lucia'
df_st_lucia_castries_train_xy['region'] = 'castries'


# In[68]:


df_st_lucia_dennery_train_xy=pd.read_csv(data_dir/f'df_st_lucia_dennery_train_centroids.csv')
df_st_lucia_dennery_test_xy=pd.read_csv(data_dir/f'df_st_lucia_dennery_test_centroids.csv')
df_st_lucia_dennery_train_xy['country'] = 'st_lucia'
df_st_lucia_dennery_train_xy['region'] = 'dennery'
df_st_lucia_dennery_test_xy['country'] = 'st_lucia'
df_st_lucia_dennery_test_xy['region'] = 'dennery'


# In[69]:


df_st_lucia_gros_islet_train_xy=pd.read_csv(data_dir/f'df_st_lucia_gros_islet_train_centroids.csv')
df_st_lucia_gros_islet_train_xy['country'] = 'st_lucia'
df_st_lucia_gros_islet_train_xy['region'] = 'gros_islet'


# In[70]:


df_centroids_train=df_colombia_borde_rural_train_xy.copy()
df_centroids_test=df_colombia_borde_rural_test_xy.copy()


# In[71]:


train_data_list=[df_colombia_borde_soacha_train_xy, df_guatemala_mixco_1_and_ebenezer_train_xy, df_guatemala_mixco_3_train_xy, df_st_lucia_castries_train_xy, df_st_lucia_dennery_train_xy, df_st_lucia_gros_islet_train_xy]


# In[72]:


test_data_list=[df_colombia_borde_soacha_test_xy, df_guatemala_mixco_1_and_ebenezer_test_xy, df_guatemala_mixco_3_test_xy, df_st_lucia_dennery_test_xy]


# In[73]:


df_centroids_train.append(train_data_list, ignore_index=True)
df_centroids_test.append(test_data_list, ignore_index=True)


# In[ ]:


assert len(df_centroids_train)=7325


# In[91]:


df_centroids_train.to_csv(data_dir/'df_centroids_train.csv', index=False)


# In[92]:


df_centroids_test.to_csv(data_dir/'df_centroids_test.csv', index=False)


# To put this in a `DataBunch` while using the [data block API](https://docs.fast.ai/data_block.html), we then need to using `ImageList` (and not `ImageDataBunch`). This will make sure the model created has the proper loss function to deal with the multiple classes.

# In[74]:


tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)


# ### setup dataset

# In[75]:


bs=32


# In[77]:


#src = ImageDataBunch.from_df(path=data_dir_02, df=train_images, csv_labels=data_dir/f'df_colombia_borde_rural_train_centroids.csv', bs=bs, tfms = tfms, continuous=True)


# In[99]:


np.random.seed(42)
dep_var='centroid'
#src = (ImageList.from_csv(path=train_images, delimiter=' ', csv_name='df_centroids_train.csv', fn_col=0, cols=1, suffix='.tif')
#       .split_by_rand_pct(0.1)
#      .label_from_df(cols=dep_var)
#      .add_test_folder(test_images))


# In[ ]:


def get_y_func(pathStr):
    name = Path(pathStr).name
    coords = path2coords[name]
    return coords


# In[ ]:


src = (Data.PointsItemList.from_df(
             df = df_centroids_train,
             path = train_images,
         )
 .random_split_by_pct()
 .label_from_func(get_y_func)
 .transform(Transform.get_transforms(), tfm_y=True, size=(224,224))
 .databunch())


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

# In[30]:


arch = models.resnet50
arch_name = 'rn50'


# In[31]:


learn = cnn_learner(data, arch, metrics=error_rate, bn_final=True)


# In[32]:


#learn.model = torch.nn.DataParallel(learn.model, device_ids=[0, 1, 2])


# We use the LR Finder to pick a good learning rate.

# ### Train model

# In[33]:


learn.lr_find()


# In[34]:


learn.recorder.plot()


# Then we can fit the head of our network.

# In[35]:


lr = 1e-2


# In[36]:


learn.fit_one_cycle(5, slice(lr))


# In[37]:


print(f'stage-1-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}')


# In[38]:


learn.save(f'stage-1-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}')
#saves in parent of models directory
#learn.export()


# In[39]:


learn.load(f'stage-1-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}')


# In[40]:


learn.fit_one_cycle(5, slice(lr))


# In[41]:


learn.save(f'stage-1-1-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}')


# #### Load model

# In[42]:


learn.load(f'stage-1-1-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}')


# In[41]:


interp = ClassificationInterpretation.from_learner(learn)
interp.most_confused(min_val=2)


# In[42]:


interp.plot_confusion_matrix(figsize=(10,10), dpi=60)


# ### Re-train

# In[45]:


learn.unfreeze()


# In[46]:


learn.lr_find()
learn.recorder.plot()


# In[57]:


learn.fit_one_cycle(5, slice(5e-6, lr/5))


# In[58]:


learn.save(f'stage-2-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}')


# In[59]:


learn.load(f'stage-2-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}')


# In[60]:


learn.fit_one_cycle(5, slice(5e-7, lr/5))


# In[61]:


learn.recorder.plot_losses()


# In[ ]:


learn.save(f'stage-2-1-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}')


# In[62]:


learn.load(f'stage-2-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}')


# In[63]:


learn.export()


# ### inference

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




