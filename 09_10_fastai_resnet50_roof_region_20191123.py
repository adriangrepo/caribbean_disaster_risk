#!/usr/bin/env python
# coding: utf-8

# ## Predict roof and country, all data
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
import geopandas as gpd
from resizeimage import resizeimage
import datetime
import uuid
from os import listdir
from os.path import isfile, join
from ipyexperiments import *


# In[3]:


torch.cuda.set_device(0)
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


RESIZE_IMAGES = True


# In[9]:


MODEL_NAME='cv_reflect_101'


# In[10]:


NB_NUM='09_10'


# In[11]:


DATE = datetime.datetime.today().strftime('%Y%m%d')
UID=str(uuid.uuid4())[:8]
print(f'UID: {UID}, DATE: {DATE}')                                                 


# In[12]:


#DATE = '20191123'
#UID = 'b4030283'


# In[13]:


SUB_NUM='2'


# In[14]:


img_size=256
bs=128


# In[15]:


train_images=data_dir/f'train/rotated/clipped/{img_size}'
test_images=data_dir/f'test/rotated/clipped/{img_size}'


# In[16]:


test_names = get_image_files(test_images)


# In[17]:


assert len(test_names)==7325


# #### useing only gold st_lucia cement

# In[19]:


df_all=pd.read_csv(data_dir/'df_all_repl_st_lucia_castries_gold_concrete_cement.csv')


# In[20]:


df_all.tail()


# In[21]:


df_all.drop(columns=['target'],inplace=True)


# In[22]:


#### Sepate roof types per country


# In[23]:


df_all['target']=df_all['roof_material']+'_'+df_all['region']


# In[24]:


df_all.head()


# In[25]:


df_all.target.value_counts().plot(kind='bar')


# In[26]:


df_test=pd.read_csv(data_dir/'df_test_all.csv')


# In[27]:


df_test.tail()


# In[28]:


assert len(df_test)==7325


# To put this in a `DataBunch` while using the [data block API](https://docs.fast.ai/data_block.html), we then need to using `ImageList` (and not `ImageDataBunch`). This will make sure the model created has the proper loss function to deal with the multiple classes.

# In[33]:


#xtra_tfms=[rand_crop(p=0.4)], 
tfms = get_transforms(flip_vert=True, max_lighting=0.25, max_zoom=1.2, max_warp=0., xtra_tfms=xtra_tfms)

#tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)


# ### setup dataset

# In[34]:


np.random.seed(42)
dep_var='target'
src = (ImageList.from_df(path=train_images, df=df_all, cols='id', suffix='.tif')
       .split_by_rand_pct(0.1)
      .label_from_df(cols=dep_var)
      .add_test_folder(test_images))


# In[35]:


data = (src.transform(tfms, size=img_size)
        .databunch(bs=bs).normalize(imagenet_stats))


# In[32]:


data.loss_func


# `show_batch` still works, and show us the different labels separated by `;`.

# In[ ]:


data.show_batch(rows=3, figsize=(12,9))


# ### Model

# In[ ]:


#exp1 = IPyExperimentsPytorch()


# In[ ]:


arch = models.resnet50
arch_name = 'rn50'


# In[ ]:


#acc_02 = partial(accuracy_thresh, thresh=0.2)
#f_score = partial(fbeta, thresh=0.2)


# In[ ]:


learn = cnn_learner(data, arch, metrics=error_rate, bn_final=True).to_fp16()


# In[ ]:


#learn.model = torch.nn.DataParallel(learn.model, device_ids=[0, 1, 2])


# We use the LR Finder to pick a good learning rate.

# ### Train model

# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# Then we can fit the head of our network.

# In[62]:


lr = 1e-2


# In[63]:


learn.fit_one_cycle(5, slice(lr))


# In[64]:


learn.loss_func


# In[65]:


learn.save(f'stage-1-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}')


# In[66]:


learn.fit_one_cycle(5, slice(lr))


# In[67]:


learn.save(f'stage-1-1-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}')


# #### Load model

# In[68]:


learn.load(f'stage-1-1-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}')


# In[69]:


interp = ClassificationInterpretation.from_learner(learn)
interp.most_confused(min_val=2)


# In[70]:


interp.plot_confusion_matrix(figsize=(10,10), dpi=60)


# ### Re-train

# In[71]:


#exp2 = IPyExperimentsPytorch()


# In[72]:


learn.unfreeze()


# In[73]:


learn.lr_find()
learn.recorder.plot()


# In[74]:


learn.fit_one_cycle(5, slice(1e-5, lr/5))


# In[75]:


learn.save(f'stage-2-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}')


# In[76]:


learn.load(f'stage-2-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}')


# In[77]:


#learn.fit_one_cycle(5, slice(1e-6, lr/5))


# In[78]:


learn.recorder.plot_losses()


# In[79]:


#learn.save(f'stage-2-2-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}')


# In[80]:


learn = learn.to_fp32()


# In[81]:


learn.export(f'stage-2-2-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}.pkl')


# In[82]:


learn.export()


# ### inference

# In[83]:


#test_images=data_dir/f'test/rotated/clipped/{img_size}'
test_dataset=ImageList.from_folder(test_images)


# In[84]:


len(test_dataset)


# In[85]:


learn = load_learner(path=data_dir/f'train/rotated/clipped/{img_size}', test=test_dataset)


# In[86]:


learn.data.loss_func


# In[87]:


type(learn.data)


# In[88]:


type(learn.dl(DatasetType.Test))


# In[89]:


len(learn.dl(DatasetType.Test))


# In[90]:


preds,y= learn.get_preds(ds_type=DatasetType.Test)


# In[91]:


learn.data.c


# In[92]:


learn.data.classes


# In[93]:


preds.shape


# In[94]:


labels = np.argmax(preds, 1)


# In[95]:


len(preds)


# In[96]:


preds[0].tolist()


# In[97]:


preds_list=[]
for pred in preds:
    preds_list.append(pred.tolist())


# In[98]:


learn.data.classes


# In[99]:


test_predictions = [learn.data.classes[int(x)] for x in labels]


# In[100]:


test_predictions[0]


# In[101]:


ids=[]
for item in learn.data.test_ds.x.items:
    base, id = os.path.split(item)
    id = id.split('.tif')[0]
    ids.append(id)


# In[102]:


preds_list[0]


# In[103]:


cols = list(learn.data.classes.copy())
cols.insert(0,'id')
df = pd.DataFrame(list(zip(ids, preds_list)), 
               columns =['id', 'pred']) 


# In[104]:


cols


# In[105]:


df.head()


# In[106]:


pred_df = pd.DataFrame(df['pred'].values.tolist())


# In[107]:


pred_df.insert(loc=0, column='id', value=ids)


# In[108]:


pred_df.columns = cols


# In[109]:


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

# #### combine to just roof material

# In[110]:


pred_df["concrete_cement"] = pred_df[["concrete_cement_colombia", "concrete_cement_guatemala","concrete_cement_st_lucia"]].max(axis=1)


# In[111]:


pred_df["healthy_metal"] = pred_df[["healthy_metal_colombia", "healthy_metal_guatemala","healthy_metal_st_lucia"]].max(axis=1)


# TODO check why

# In[112]:


#pred_df["incomplete"] = pred_df[["incomplete_colombia", "incomplete_guatemala","incomplete_st_lucia"]].max(axis=1)


# In[113]:


pred_df["incomplete"] = pred_df[["incomplete_colombia", "incomplete_st_lucia"]].max(axis=1)


# In[114]:


pred_df["irregular_metal"] = pred_df[["irregular_metal_colombia", "irregular_metal_guatemala","irregular_metal_st_lucia"]].max(axis=1)


# In[115]:


pred_df["other"] = pred_df[["other_colombia", "other_guatemala","other_st_lucia"]].max(axis=1)


# #### drop country

# In[116]:


pred_df.drop(columns=["concrete_cement_colombia", "concrete_cement_guatemala","concrete_cement_st_lucia", "healthy_metal_colombia", "healthy_metal_guatemala","healthy_metal_st_lucia", "incomplete_colombia", "incomplete_st_lucia", "irregular_metal_colombia", "irregular_metal_guatemala","irregular_metal_st_lucia", "other_colombia", "other_guatemala","other_st_lucia"],inplace=True)


# In[117]:


pred_df.head()


# #### Format correctly

# In[118]:


pred_ids=pred_df['id'].values.tolist()


# In[119]:


df_baseline = pd.read_csv(data_dir/f'submissions/mean_baseline.csv')


# In[120]:


df_baseline.head()


# In[121]:


baseline_ids=df_baseline['id'].values.tolist()


# In[122]:


assert set(pred_ids)==set(baseline_ids)


# #### sort by baseline ids

# In[123]:


pred_df['id_cat'] = pd.Categorical(
    pred_df['id'], 
    categories=baseline_ids, 
    ordered=True
)


# In[124]:


pred_df.head()


# In[125]:


pred_df=pred_df.sort_values('id_cat')


# In[126]:


pred_df.head()


# In[127]:


pred_df.drop(columns=['id_cat'],inplace=True)


# In[128]:


arch_name='rn50'


# In[129]:


pred_df=pred_df.drop_duplicates(subset=['id'])


# In[130]:


pred_df.head()


# In[131]:


print(f'submissions/stage-2-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}.csv')


# In[132]:


pred_df.to_csv(data_dir/f'submissions/stage-2-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}.csv', index=False)


# In[ ]:





# In[133]:


### Submission 2: 0.4461


# In[134]:


arch_name = 'rn50'
pred_df=pd.read_csv(data_dir/f'submissions/stage-2-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}.csv')


# In[135]:


pred_df.drop(columns=['id'],inplace=True)
classes=pred_df.idxmax(axis=1)
pd.value_counts(classes).plot(kind="bar")


# In[ ]:





# In[ ]:





# In[ ]:




