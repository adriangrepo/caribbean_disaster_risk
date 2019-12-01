#!/usr/bin/env python
# coding: utf-8

# ## Train and Prediction on all data
# 
# Using rotated to hz + OpenCv border
# 
# Basic default transforms



get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')




from fastai.vision import *
import geopandas as gpd
from resizeimage import resizeimage
import datetime
import uuid
from os import listdir
from os.path import isfile, join




torch.cuda.set_device(0)
torch.cuda.current_device()




data_dir = Path('data')




RETRAIN = True
RESIZE_IMAGES = True




MODEL_NAME='cv_reflect_101_valid'




NB_NUM='03_11'




DATE = datetime.datetime.today().strftime('%Y%m%d')
UID=str(uuid.uuid4())[:8]
print(f'UID: {UID}, DATE: {DATE}')                                                 




DATE = '20191109'
UID = '123cca5f'




SUB_NUM='15'




img_size=256




train_images=data_dir/f'train/rotated/clipped/reflect/{img_size}'
test_images=data_dir/f'test/rotated/clipped/reflect/{img_size}'




test_names = get_image_files(test_images)




assert len(test_names)==7325




df_all=pd.read_csv(data_dir/'df_train_all.csv')




len(df_all)




df_valid=df_all.loc[df_all['verified'] == True]




df_test=pd.read_csv(data_dir/'df_test_all.csv')




df_test.tail()




assert len(df_test)==7325




df_valid.loc[df_valid['id'] == '7a204ec4']




len(df_valid)


# To put this in a `DataBunch` while using the [data block API](https://docs.fast.ai/data_block.html), we then need to using `ImageList` (and not `ImageDataBunch`). This will make sure the model created has the proper loss function to deal with the multiple classes.



tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)


# ### setup dataset



np.random.seed(42)
dep_var='roof_material'
src = (ImageList.from_df(path=train_images, df=df_valid, cols='id', suffix='.tif')
       .split_by_rand_pct(0.2)
      .label_from_df(cols=dep_var)
      .add_test_folder(test_images))




data = (src.transform(tfms, size=img_size)
        .databunch().normalize(imagenet_stats))




#to check what params object has
#dir(data)




data.label_list




data.loss_func


# `show_batch` still works, and show us the different labels separated by `;`.



data.show_batch(rows=3, figsize=(12,9))


# ### Model



arch = models.resnet50
arch_name = 'rn50'




learn = cnn_learner(data, arch, metrics=error_rate, bn_final=True)




#learn.model = torch.nn.DataParallel(learn.model, device_ids=[0, 1, 2])


# We use the LR Finder to pick a good learning rate.

# ### Train model



learn.lr_find()




learn.recorder.plot()


# Then we can fit the head of our network.



lr = 5e-3




learn.fit_one_cycle(5, slice(lr))




print(f'stage-1-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}')




learn.save(f'stage-1-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}')
#saves in parent of models directory
#learn.export()




learn.load(f'stage-1-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}')




learn.fit_one_cycle(3, slice(lr))




learn.save(f'stage-1-1-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}')


# #### Load model



learn.load(f'stage-1-1-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}')




interp = ClassificationInterpretation.from_learner(learn)
interp.most_confused(min_val=2)




interp.plot_confusion_matrix(figsize=(10,10), dpi=60)


# ### Re-train



learn.unfreeze()




learn.lr_find()
learn.recorder.plot()




learn.fit_one_cycle(5, slice(5e-6, lr/5))




learn.save(f'stage-2-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}')




learn.load(f'stage-2-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}')




learn.fit_one_cycle(5, slice(5e-7, lr/5))




learn.recorder.plot_losses()




learn.save(f'stage-2-1-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}')




learn.load(f'stage-2-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}')




learn.export()


# ### inference



#test_images=data_dir/f'test/rotated/clipped/{img_size}'
test_dataset=ImageList.from_folder(test_images)




len(test_dataset)




learn = load_learner(path=data_dir/f'train/rotated/clipped/{img_size}', test=test_dataset)




#learn = load_learner(path=data_dir/f'train/rotated/clipped/{img_size}', file=f'stage-2-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}.pkl', test=test_dataset)




learn.data.loss_func




type(learn.data)




type(learn.dl(DatasetType.Test))




len(learn.dl(DatasetType.Test))


# Get number of items in the Valid dataset (in DeviceDataLoader)



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



preds,y= learn.get_preds(ds_type=DatasetType.Test)




labels = np.argmax(preds, 1)




len(preds)




preds[0].tolist()




preds_list=[]
for pred in preds:
    preds_list.append(pred.tolist())




len(labels)




learn.data.classes




data.classes




test_predictions = [learn.data.classes[int(x)] for x in labels]




test_predictions[0]




type(learn.data.test_ds)




learn.data.test_ds.x.items




ids=[]
for item in learn.data.test_ds.x.items:
    base, id = os.path.split(item)
    id = id.split('.tif')[0]
    ids.append(id)




preds_list[0]




cols = learn.data.classes.copy()
cols.insert(0,'id')
df = pd.DataFrame(list(zip(ids, preds_list)), 
               columns =['id', 'pred']) 




cols




df.head()




pred_df = pd.DataFrame(df['pred'].values.tolist())




pred_df.insert(loc=0, column='id', value=ids)




pred_df.columns = cols




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



pred_ids=pred_df['id'].values.tolist()




df_baseline = pd.read_csv(data_dir/f'submissions/mean_baseline.csv')




df_baseline.head()




baseline_ids=df_baseline['id'].values.tolist()




assert set(pred_ids)==set(baseline_ids)


# #### sort by baseline ids



pred_df['id_cat'] = pd.Categorical(
    pred_df['id'], 
    categories=baseline_ids, 
    ordered=True
)




pred_df.head()




pred_df=pred_df.sort_values('id_cat')




pred_df.head()




pred_df.drop(columns=['id_cat'],inplace=True)




pred_df.to_csv(data_dir/f'submissions/stage-2-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}.csv', index=False)




### Submission 2: 0.4687




arch_name = 'rn50'
pred_df=pd.read_csv(data_dir/f'submissions/stage-2-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}.csv')




pred_df.drop(columns=['id'],inplace=True)
classes=pred_df.idxmax(axis=1)
pd.value_counts(classes).plot(kind="bar")






