#!/usr/bin/env python
# coding: utf-8

# ## Train and Prediction on valid
# 
# Using rotated to hz + OpenCv border
# 
# Basic default transforms
# 
# Densenet
# 
# at bs=32 only using 1.8GB

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
from torchvision.models import densenet121


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


NB_NUM='06_11'


# In[11]:


DATE = datetime.datetime.today().strftime('%Y%m%d')
UID=str(uuid.uuid4())[:8]
print(f'UID: {UID}, DATE: {DATE}')                                                 


# In[12]:


DATE = '20191126'
UID = 'b8154638'


# In[13]:


SUB_NUM='7'


# In[14]:


img_size=256
bs=64


# In[15]:


train_images=data_dir/f'train/rotated/clipped/{img_size}'
test_images=data_dir/f'test/rotated/clipped/{img_size}'


# In[16]:


### add St Lucia pred cement as valid


# In[17]:


df_all=pd.read_csv(data_dir/'df_all_repl_st_lucia_castries_gold_concrete_cement.csv')


# In[18]:


df_all.drop(columns=['target'],inplace=True)


# In[19]:


df_valid=df_all.loc[df_all['verified'] == True]


# In[ ]:





# In[20]:


df_test=pd.read_csv(data_dir/'df_test_all.csv')


# In[21]:


df_test.tail()


# In[22]:


df_all.loc[df_all['id'] == '7a204ec4']


# In[23]:


len(df_all)


# To put this in a `DataBunch` while using the [data block API](https://docs.fast.ai/data_block.html), we then need to using `ImageList` (and not `ImageDataBunch`). This will make sure the model created has the proper loss function to deal with the multiple classes.

# In[24]:


xtra_tfms=[rand_crop(p=0.4)]
tfms = get_transforms(flip_vert=True, max_lighting=0.25, max_zoom=1.2, max_warp=0., xtra_tfms=xtra_tfms)


# ### setup dataset

# In[25]:


np.random.seed(42)
dep_var='roof_material'
src = (ImageList.from_df(path=train_images, df=df_valid, cols='id', suffix='.tif')
       .split_by_rand_pct(0.1)
      .label_from_df(cols=dep_var)
      .add_test_folder(test_images))


# In[26]:


data = (src.transform(tfms, size=256)
        .databunch(bs=bs).normalize(imagenet_stats))


# In[27]:


#to check what params object has
#dir(data)


# In[28]:


data.label_list


# In[29]:


data.loss_func


# `show_batch` still works, and show us the different labels separated by `;`.

# In[30]:


data.show_batch(rows=3, figsize=(12,9))


# In[31]:


from fastai.callbacks import *

# small change to SaveModelCallback() to add printouts
@dataclass
class SaveModelCallbackVerbose(TrackerCallback):
    "A `TrackerCallback` that saves the model when monitored quantity is best."
    every:str='improvement'
    name:str='bestmodel'
    def __post_init__(self):
        if self.every not in ['improvement', 'epoch']:
            warn(f'SaveModel every {self.every} is invalid, falling back to "improvement".')
            self.every = 'improvement'
        super().__post_init__()

    def on_epoch_end(self, epoch, **kwargs:Any)->None:
        if self.every=="epoch": self.learn.save(f'{self.name}_{epoch}')
        else: #every="improvement"
            current = self.get_monitor_value()
            if current is not None and self.operator(current, self.best):
                self.best = current
                self.learn.save(f'{self.name}')
                print(f'saved model at epoch {epoch} with {self.monitor} value: {current}')

    def on_train_end(self, **kwargs):
        if self.every=="improvement": self.learn.load(f'{self.name}')


# ### Model

# In[32]:


arch = models.densenet169
arch_name = 'dn169'


# In[33]:


#FP16


# In[34]:


learn = cnn_learner(data, arch, metrics=error_rate, bn_final=True).to_fp16()


# In[35]:


#learn.model = torch.nn.DataParallel(learn.model, device_ids=[0, 1, 2])


# We use the LR Finder to pick a good learning rate.

# ### Train model

# In[36]:


learn.lr_find()


# In[37]:


learn.recorder.plot()


# Then we can fit the head of our network.

# In[38]:


lr = 1e-2


# In[39]:


'''learn.fit_one_cycle(5, max_lr=lr, 
                    callbacks=[
                        SaveModelCallbackVerbose(learn,
                                                 monitor='dice',
                                                 mode='max',
                                                 name=f'{DATE}-{ARCH_NAME}-{MODEL_NAME}-comboloss-best-{UID}')
                    ]
                   )'''


# In[40]:


learn.fit_one_cycle(5, slice(lr),  callbacks=[
                        SaveModelCallbackVerbose(learn,
                                                 monitor='error_rate',
                                                 mode='min',
                                                 name=f'stage-1-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}')
])


# In[ ]:


learn.fit_one_cycle(5, slice(lr),  callbacks=[
                        SaveModelCallbackVerbose(learn,
                                                 monitor='error_rate',
                                                 mode='min',
                                                 name=f'stage-1-1-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}')


# In[ ]:


learn.fit_one_cycle(5, slice(lr),  callbacks=[
                        SaveModelCallbackVerbose(learn,
                                                 monitor='error_rate',
                                                 mode='min',
                                                 name=f'stage-1-2-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}')


# #### Load model

# In[ ]:


learn.load(f'stage-1-1-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}')


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
interp.most_confused(min_val=2)


# In[ ]:


interp.plot_confusion_matrix(figsize=(10,10), dpi=60)


# ### Re-train

# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(5, slice(1e-6, lr/5),  callbacks=[
                        SaveModelCallbackVerbose(learn,
                                                 monitor='error_rate',
                                                 mode='min',
                                                 name=f'stage-2-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}')


# In[ ]:


learn.fit_one_cycle(5, slice(1e-6, lr/5),  callbacks=[
                        SaveModelCallbackVerbose(learn,
                                                 monitor='error_rate',
                                                 mode='min',
                                                 name=f'stage-2-2-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}')


# In[ ]:


learn.fit_one_cycle(5, slice(1e-6, lr/5),  callbacks=[
                        SaveModelCallbackVerbose(learn,
                                                 monitor='error_rate',
                                                 mode='min',
                                                 name=f'stage-2-3-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}')


# In[ ]:


learn.recorder.plot_losses()


# ### Load model and export for inference

# In[32]:


learn.load(f'stage-2b-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}')


# In[33]:


learn = learn.to_fp32()


# In[34]:


learn.export()


# In[ ]:





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

# In[38]:


test_dataset=ImageList.from_folder(test_images)


# In[39]:


learn = load_learner(path=data_dir/f'train/rotated/clipped/{img_size}', test=test_dataset)


# In[40]:


learn.data.loss_func


# In[41]:


type(learn.data)


# In[42]:


type(learn.dl(DatasetType.Test))


# Get number of items in the Valid dataset (in DeviceDataLoader)

# In[43]:


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

# In[44]:


preds,y= learn.get_preds(ds_type=DatasetType.Test)


# In[45]:


labels = np.argmax(preds, 1)


# In[46]:


len(preds)


# In[47]:


preds[0].tolist()


# In[48]:


preds_list=[]
for pred in preds:
    preds_list.append(pred.tolist())


# In[49]:


len(labels)


# In[50]:


learn.data.classes


# In[51]:


data.classes


# In[52]:


test_predictions = [learn.data.classes[int(x)] for x in labels]


# In[53]:


test_predictions[0]


# In[54]:


type(learn.data.test_ds)


# In[55]:


learn.data.test_ds.x.items


# In[56]:


ids=[]
for item in learn.data.test_ds.x.items:
    base, id = os.path.split(item)
    id = id.split('.tif')[0]
    ids.append(id)


# In[57]:


preds_list[0]


# In[58]:


cols = learn.data.classes.copy()
cols.insert(0,'id')
df = pd.DataFrame(list(zip(ids, preds_list)), 
               columns =['id', 'pred']) 


# In[59]:


cols


# In[60]:


df.head()


# In[61]:


pred_df = pd.DataFrame(df['pred'].values.tolist())


# In[62]:


pred_df.insert(loc=0, column='id', value=ids)


# In[63]:


pred_df.columns = cols


# In[64]:


pred_df.head()


# In[65]:


pred_ids=pred_df['id'].values.tolist()


# In[66]:


df_baseline = pd.read_csv(data_dir/f'submissions/mean_baseline.csv')


# In[67]:


baseline_ids=df_baseline['id'].values.tolist()


# In[68]:


pred_df['id_cat'] = pd.Categorical(
    pred_df['id'], 
    categories=baseline_ids, 
    ordered=True
)


# In[69]:


pred_df=pred_df.sort_values('id_cat')


# In[70]:


pred_df.drop(columns=['id_cat'],inplace=True)


# In[71]:


pred_df=pred_df.drop_duplicates(subset=['id'])


# In[72]:


pred_df.to_csv(data_dir/f'submissions/stage-2-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}.csv', index=False)


# #### Submission result

# sub 8: 0.4947 

# In[74]:


pred_df=pd.read_csv(data_dir/f'submissions/stage-2-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}.csv')


# In[75]:


pred_df.drop(columns=['id'],inplace=True)
classes=pred_df.idxmax(axis=1)
pd.value_counts(classes).plot(kind="bar")


# In[ ]:





# In[ ]:





# In[ ]:




