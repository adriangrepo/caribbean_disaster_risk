#!/usr/bin/env python
# coding: utf-8

# ## Train and Prediction on all data
# 
# Using rotated to hz + OpenCv border
# 
# Basic default transforms

# In[1]:



# In[2]:


from fastai.vision import *
import geopandas as gpd
from resizeimage import resizeimage
import datetime
import uuid
from os import listdir
from os.path import isfile, join
import time
from PIL import Image as pil_image


# In[3]:


torch.cuda.set_device(1)
torch.cuda.current_device()


# In[4]:


data_dir = Path('data')


# In[5]:


RETRAIN = True
RESIZE_IMAGES = True


# In[6]:


MODEL_NAME='cv_reflect_101_valid'


# In[7]:


NB_NUM='03_11_1'


# In[8]:


DATE = datetime.datetime.today().strftime('%Y%m%d')
UID=str(uuid.uuid4())[:8]
print(f'UID: {UID}, DATE: {DATE}')                                                 


# In[9]:


#DATE = '20191109'
#UID = '123cca5f'


# In[10]:


SUB_NUM='15'
RUN=9


# In[11]:


img_size=256


# In[12]:


train_images=data_dir/f'train/rotated/clipped/reflect/{img_size}'
test_images=data_dir/f'test/rotated/clipped/reflect/{img_size}'


# In[13]:


test_names = get_image_files(test_images)


# In[14]:


assert len(test_names)==7325


# ### add St Lucia pred cement as valid

# In[15]:


df_all=pd.read_csv(data_dir/'df_all_repl_st_lucia_castries_gold_concrete_cement.csv')


# In[16]:


df_all.drop(columns=['target'],inplace=True)


# In[17]:


df_valid=df_all.loc[df_all['verified'] == True]


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


# see https://github.com/kechan/FastaiPlayground/blob/master/Quick%20Tour%20of%20Data%20Augmentation.ipynb

# In[23]:


def gaussian_kernel(size, sigma=2., dim=2, channels=3):
    # The gaussian kernel is the product of the gaussian function of each dimension.
    # kernel_size should be an odd number.
    
    kernel_size = 2*size + 1

    kernel_size = [kernel_size] * dim
    sigma = [sigma] * dim
    kernel = 1
    meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
    
    for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
        mean = (size - 1) / 2
        kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

    # Make sure sum of values in gaussian kernel equals 1.
    kernel = kernel / torch.sum(kernel)

    # Reshape to depthwise convolutional weight
    kernel = kernel.view(1, 1, *kernel.size())
    kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

    return kernel


# In[24]:


def _gaussian_blur(x, size:uniform_int, sigma):
    kernel = gaussian_kernel(size=size, sigma=sigma)
    kernel_size = 2*size + 1

    x = x[None,...]
    padding = int((kernel_size - 1) / 2)
    x = F.pad(x, (padding, padding, padding, padding), mode='reflect')
    x = torch.squeeze(F.conv2d(x, kernel, groups=3))

    return x

gaussian_blur = TfmPixel(_gaussian_blur)


# In[25]:


#im = pil_image.open(train_images/'7a204ec4.tif')


# In[26]:


#blr_im=_gaussian_blur(im, size=(1, 10), sigma=2.)


# To put this in a `DataBunch` while using the [data block API](https://docs.fast.ai/data_block.html), we then need to using `ImageList` (and not `ImageDataBunch`). This will make sure the model created has the proper loss function to deal with the multiple classes.

# In[27]:


#xtra_tfms=[dihedral(p=0.5), rand_crop(p=0.4), rand_zoom(scale=(1.,1.5),p=0.4)] 
xtra_tfms=[rand_crop(p=0.4), gaussian_blur(size=(1, 10), sigma=1., p=0.1)]
tfms = get_transforms(flip_vert=True, max_lighting=0.25, max_zoom=1.2, max_warp=0., xtra_tfms=xtra_tfms)


# In[28]:


#tfms = get_transforms(flip_vert=True, max_lighting=0.25, max_zoom=1.2, max_warp=0.)


# ### setup dataset

# In[29]:


np.random.seed(42)
dep_var='roof_material'
src = (ImageList.from_df(path=train_images, df=df_valid, cols='id', suffix='.tif')
       .split_by_rand_pct(0.1)
      .label_from_df(cols=dep_var)
      .add_test_folder(test_images))


# In[30]:


data = (src.transform(tfms, size=img_size)
        .databunch().normalize(imagenet_stats))


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


# In[36]:


learn = cnn_learner(data, arch, metrics=error_rate, bn_final=True)


# In[37]:


#learn.model = torch.nn.DataParallel(learn.model, device_ids=[0, 1, 2])


# We use the LR Finder to pick a good learning rate.

# ### Train model

# In[38]:


learn.lr_find()


# In[39]:


learn.recorder.plot()


# Then we can fit the head of our network.

# In[40]:


lr = 1e-2


# In[41]:


learn.fit_one_cycle(5, slice(lr))


# In[42]:


learn.fit_one_cycle(5, slice(lr))


# In[44]:


print(time.localtime())


# In[42]:


learn.loss_func


# In[37]:


print(f'stage-1-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}')


# In[37]:


learn.save(f'stage-1-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}')
#saves in parent of models directory
#learn.export()


# In[37]:


learn.load(f'stage-1-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}')


# In[38]:


learn.fit_one_cycle(3, slice(lr))


# In[39]:


learn.save(f'stage-1-1-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}')


# #### Load model

# In[52]:


learn.load(f'stage-1-1-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}')


# In[53]:


interp = ClassificationInterpretation.from_learner(learn)
interp.most_confused(min_val=2)


# In[54]:


interp.plot_confusion_matrix(figsize=(10,10), dpi=60)


# ### Re-train

# In[55]:


learn.unfreeze()


# In[56]:


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




