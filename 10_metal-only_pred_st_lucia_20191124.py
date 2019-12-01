#!/usr/bin/env python
# coding: utf-8

# ## Predict St Lucia Metal Roofs
# 
# Using rotated to hz + OpenCv border
# 

from fastai.vision import *
from callbacks import *
import geopandas as gpd
from resizeimage import resizeimage
import datetime
import uuid
from os import listdir
from os.path import isfile, join
from ipyexperiments import *
from shutil import copyfile

torch.cuda.set_device(0)

data_dir = Path('data')

NB_NUM='10'

DATE = datetime.datetime.today().strftime('%Y%m%d')
UID=str(uuid.uuid4())[:8]
print(f'NB_NUM: {NB_NUM}, UID: {UID}, DATE: {DATE}, GPU:{torch.cuda.current_device()}')
#DATE='20191124'
#UID='6d85dc5a'
img_size=256

#only need to do this once
STARTUP=False

TRAIN_IMAGES=data_dir/f'train/rotated/clipped/{img_size}'
SILVER_TRAIN=TRAIN_IMAGES/'silver_train'
#pseudo test set - what we run inference on
SILVER_UNVERIFIED = TRAIN_IMAGES/'silver_unverified'

def setup_df(train_type):
    '''Only using verified (+gold st_lucia cement) data for training'''
    assert isinstance(train_type, str)
    target_col=None
    df_all=pd.read_csv(data_dir/'df_all_repl_st_lucia_castries_gold_concrete_cement.csv')
    df_all.drop(columns=['target'],inplace=True)

    if train_type=='region':
        df_metal = df_all.copy()
        df_metal['target']=df_metal['roof_material']+'_'+df_metal['region']
        target_col='target'
        df_unverified = df_all.copy()
    elif train_type=='country':
        df_metal = df_all.copy()
        df_metal['target']=df_metal['roof_material']+'_'+df_metal['country']
        target_col = 'target'
        df_unverified = df_all.copy()
    elif train_type=='roof_material':
        df_metal = df_all.copy()
        target_col = 'roof_material'
        df_unverified = df_all.copy()
    #restricted training set options
    elif train_type=='im_hm_region':
        df_metal = df_all.loc[df_all['roof_material'].isin(['irregular_metal', 'healthy_metal'])]
        df_metal['target']=df_metal['roof_material']+'_'+df_metal['region']
        target_col='target'
        df_unverified = df_all.copy()
    elif train_type=='im_hm_country':
        df_metal = df_all.loc[df_all['roof_material'].isin(['irregular_metal', 'healthy_metal'])]
        df_metal['target']=df_metal['roof_material']+'_'+df_metal['country']
        target_col = 'target'
        df_unverified = df_all.copy()
    elif train_type=='im_hm_binary':
        #only training on 'irregular_metal','healthy_metal' for pred of these cats on St Lucia
        df_metal = df_all.loc[df_all['roof_material'].isin(['irregular_metal','healthy_metal'])]
        target_col = 'roof_material'
        df_unverified = df_all.loc[df_all['roof_material'].isin(['irregular_metal', 'healthy_metal'])]

    df_verified = df_metal.loc[df_metal['verified'] == True]

    ### Test is unverified St Lucia
    df_unverified = df_unverified.loc[df_unverified['verified'] == False]
    return df_verified, df_unverified, target_col

def create_tfms():
    # To put this in a `DataBunch` while using the [data block API](https://docs.fast.ai/data_block.html),
    # we then need to using `ImageList` (and not `ImageDataBunch`).
    # This will make sure the model created has the proper loss function to deal with the multiple classes.
    xtra_tfms=[rand_crop(p=0.4)]
    tfms = get_transforms(flip_vert=True, max_lighting=0.25, max_zoom=1.2, max_warp=0., xtra_tfms=xtra_tfms)
    return tfms

def setup_data(bs, df, tfms, target_col, data_type):
    np.random.seed(42)
    dep_var=target_col
    src = (ImageList.from_df(path=SILVER_TRAIN/f'{data_type}', df=df, cols='id', suffix='.tif')
           .split_by_rand_pct(0.1)
          .label_from_df(cols=dep_var)
          .add_test_folder(SILVER_UNVERIFIED/f'{data_type}'))

    data = (src.transform(tfms, size=img_size)
            .databunch(bs=bs).normalize(imagenet_stats))
    return data

def train_model(train_type, data, run, arch = models.resnet50, arch_name = 'rn50', lr = 1e-2, epochs=5, loss_fn='cef', epsilon=0.1):
    # Model
    learn = cnn_learner(data, arch, metrics=error_rate, bn_final=True).to_fp16()
    if loss_fn=='cef':
        learn.loss_func = CrossEntropyFlat()
    elif loss_fn=='lsce':
        learn.loss_func = LabelSmoothingCrossEntropy(eps=epsilon)
    print(f'classes: {learn.data.classes}')
    save_name=f'stage-1-{arch_name}-{NB_NUM}_{run}-{train_type}-{loss_fn}-{epochs}-{DATE}-{UID}'
    callbacks = [SaveModelCallback(learn, every='improvement', monitor='valid_loss', name=save_name, uid=UID)]
    learn.fit_one_cycle(epochs, slice(lr), callbacks=callbacks)
    #learn.save(f'stage-1-{arch_name}-{NB_NUM}_{run}-{train_type}-{loss_fn}-{epochs}-{DATE}-{UID}')
    if arch_name=='rn50':
        if epochs-2>0:
            save_name = f'stage-1-1-{arch_name}-{NB_NUM}_{run}-{train_type}-{loss_fn}-{epochs}-{DATE}-{UID}'
            learn.fit_one_cycle(epochs-2, slice(lr), callbacks=callbacks)
            #learn.save(f'stage-1-1-{arch_name}-{NB_NUM}_{run}-{train_type}-{loss_fn}-{epochs}-{DATE}-{UID}')
    else:
        save_name = f'stage-1-1-{arch_name}-{NB_NUM}_{run}-{train_type}-{loss_fn}-{epochs}-{DATE}-{UID}'
        learn.fit_one_cycle(epochs, slice(lr), callbacks=callbacks)
    return learn


def unfreeze_train_model(train_type, run, arch_name, learn, lr, lr_uf=5e-6, epochs=5, loss_fn='cef'):
    # ### Re-train
    learn.unfreeze()
    save_name=f'stage-2-{arch_name}-{NB_NUM}_{run}-{train_type}-{loss_fn}-{epochs}-{DATE}-{UID}'
    callbacks = [SaveModelCallback(learn, every='improvement', monitor='valid_loss', name=save_name, uid=UID)]
    learn.fit_one_cycle(epochs, slice(lr_uf, lr/5), callbacks=callbacks)
    #save_name = f'stage-2-2-{arch_name}-{NB_NUM}_{run}-{train_type}-{loss_fn}-{epochs}-{DATE}-{UID}'
    #learn.fit_one_cycle(epochs, slice(lr_uf, lr / 5), callbacks=callbacks)
    #learn.save(f'stage-2-{arch_name}-{NB_NUM}_{run}-{train_type}-{loss_fn}-{epochs}-{DATE}-{UID}')
    return learn


def save_model(learn, train_type, run, arch_name, epochs, loss_fn):
    learn = learn.to_fp32()
    pkl_file = f'stage-2-{arch_name}-{NB_NUM}_{run}-{train_type}-{loss_fn}-{epochs}-{DATE}-{UID}.pkl'
    learn.export(file=pkl_file)
    return pkl_file


def format_im_hm_metals(learn, pred_df, train_type):
    # print(pred_df.head(n=2))
    l_classes = list(learn.data.classes)
    if train_type == 'im_hm_region':
        # verified datasets:
        all_healthy_metals = ['healthy_metal_mixco_1_and_ebenezer',
                              'healthy_metal_mixco_3',
                              'healthy_metal_borde_rural',
                              'healthy_metal_dennery',
                              'healthy_metal_borde_soacha']
        all_irregular_metals = ['irregular_metal_mixco_1_and_ebenezer',
                                'irregular_metal_borde_rural',
                                'irregular_metal_borde_soacha',
                                'irregular_metal_dennery',
                                'irregular_metal_mixco_3']
        # assert set(list(learn.data.classes)) == set(healthy_metals+irregular_metals)
        healthy_metals = [s for s in l_classes if 'healthy_metal' in s]
        irregular_metals = [s for s in l_classes if 'irregular_metal' in s]
        if not (set(all_healthy_metals) == set(healthy_metals)):
            print(f'validated healthy_metals {all_healthy_metals}!=learn {healthy_metals}')
        if not (set(all_irregular_metals) == set(irregular_metals)):
            print(f'validated irregular_metals {all_irregular_metals}!=learn {irregular_metals}')

        pred_df["healthy_metal"] = pred_df[healthy_metals].max(axis=1)
        pred_df["irregular_metal"] = pred_df[irregular_metals].max(axis=1)

        # #### drop region
        pred_df.drop(columns=healthy_metals, inplace=True)
        pred_df.drop(columns=irregular_metals, inplace=True)

    elif train_type == 'im_hm_country':
        # verified only
        all_healthy_metals = ['healthy_metal_guatemala',
                              'healthy_metal_st_lucia',
                              'healthy_metal_colombia']
        all_irregular_metals = ['irregular_metal_colombia',
                                'irregular_metal_st_lucia',
                                'irregular_metal_guatemala']
        healthy_metals = [s for s in l_classes if 'healthy_metal' in s]
        irregular_metals = [s for s in l_classes if 'irregular_metal' in s]
        if not (set(all_healthy_metals) == set(healthy_metals)):
            print(f'validated healthy_metals {all_healthy_metals}!=learn {healthy_metals}')
        if not (set(all_irregular_metals) == set(irregular_metals)):
            print(f'validated irregular_metals {all_irregular_metals}!=learn {irregular_metals}')
        pred_df["healthy_metal"] = pred_df[healthy_metals].max(axis=1)
        pred_df["irregular_metal"] = pred_df[irregular_metals].max(axis=1)

        # #### drop country
        pred_df.drop(columns=healthy_metals, inplace=True)
        pred_df.drop(columns=irregular_metals, inplace=True)

    else:
        print(f'binary classes: {l_classes}')
    return pred_df

def inference_multi(df, train_type, pkl_file, arch_name, run, epochs, data_type, loss_fn):
    test_dataset = ImageList.from_df(df, path=SILVER_UNVERIFIED/f'{data_type}', cols='id', suffix='.tif')
    print(test_dataset.__len__())
    learn = load_learner(path= SILVER_TRAIN/f'{data_type}', file=pkl_file, test=test_dataset)
    preds,y= learn.get_preds(ds_type=DatasetType.Test)
    labels = np.argmax(preds, 1)

    preds_list=[]
    for pred in preds:
        preds_list.append(pred.tolist())

    test_predictions = [learn.data.classes[int(x)] for x in labels]

    ids=[]
    for item in learn.data.test_ds.x.items:
        base, id = os.path.split(item)
        id = id.split('.tif')[0]
        ids.append(id)

    if train_type.startswith('im_hm'):
        cols = list(learn.data.classes.copy())
    else:
        cols = learn.data.classes.copy()
    cols.insert(0,'id')
    df = pd.DataFrame(list(zip(ids, preds_list)),
                   columns =['id', 'pred'])

    pred_df = pd.DataFrame(df['pred'].values.tolist())
    pred_df.insert(loc=0, column='id', value=ids)
    pred_df.columns = cols

    if train_type.startswith('im_hm'):
        pred_df=format_im_hm_metals(learn, pred_df, train_type)
    # #### Format correctly
    pred_ids=pred_df['id'].values.tolist()
    df_baseline = pd.read_csv(data_dir/f'submissions/mean_baseline.csv')
    baseline_ids=df_baseline['id'].values.tolist()
    pred_df['id_cat'] = pd.Categorical(
        pred_df['id'],
        categories=baseline_ids,
        ordered=True
    )
    pred_df=pred_df.sort_values('id_cat')
    pred_df.drop(columns=['id_cat'],inplace=True)
    pred_df=pred_df.drop_duplicates(subset=['id'])
    print(f'{arch_name}-{NB_NUM}_{run}-{train_type}-{loss_fn}-{epochs}-{DATE}-{UID}.csv')
    proc_dir=data_dir/'processing'
    proc_dir.mkdir(exist_ok=True)
    pred_file=f'processing/{arch_name}-{NB_NUM}_{run}-{train_type}-{data_type}-{epochs}-{loss_fn}-{DATE}-{UID}.csv'
    pred_df.to_csv(data_dir/f'{pred_file}', index=False)
    print(f'saved: {pred_file}')

def setup_arch(arch_name, train_type):
    if arch_name == 'rn50':
        arch = models.resnet50
        if train_type=='im_hm_binary' or train_type=='roof_material':
            lr = 5e-3
            lr_uf = 2e-6
        elif train_type=='im_hm_region' or train_type=='region':
            lr = 1e-2
            lr_uf = 2e-6
        elif train_type=='im_hm_country' or train_type=='country':
            lr = 1e-2
            lr_uf = 2e-6
        bs = 128
    elif arch_name == 'rn152':
        arch = models.resnet152
        if train_type=='im_hm_binary' or train_type=='roof_material':
            lr = 5e-3
            lr_uf = 1e-5
        if train_type=='im_hm_region' or train_type=='region':
            lr = 5e-3
            lr_uf = 1e-5
        if train_type=='im_hm_country' or train_type=='country':
            lr = 5e-3
            lr_uf = 1e-5
        bs = 32
    elif arch_name == 'dn121':
        arch = models.densenet121
        if train_type=='im_hm_binary' or train_type=='roof_material':
            lr = 5e-3
            lr_uf = 1e-5
        elif train_type=='im_hm_region' or train_type=='region':
            lr = 5e-3
            lr_uf = 1e-5
        elif train_type=='im_hm_country' or train_type=='country':
            lr = 5e-3
            lr_uf = 1e-5
        bs = 64
    return arch, lr, lr_uf, bs

def create_pseudo_tt_folders(df_verified, df_unverified, data_type):
    SILVER_TRAIN.mkdir(exist_ok=True)
    d=SILVER_TRAIN/f'{data_type}'
    d.mkdir(exist_ok=True)
    SILVER_UNVERIFIED.mkdir(exist_ok=True)
    d=SILVER_UNVERIFIED/f'{data_type}'
    d.mkdir(exist_ok=True)
    verified_ids=df_verified['id'].values.tolist()
    unverified_ids = df_unverified['id'].values.tolist()
    for v in verified_ids:
        copyfile(TRAIN_IMAGES/f'{v}.tif', SILVER_TRAIN/f'{data_type}/{v}.tif')
    for uv in unverified_ids:
        copyfile(TRAIN_IMAGES/f'{uv}.tif', SILVER_UNVERIFIED/f'{data_type}/{uv}.tif')


def workflow():
    '''binary or metal+country/region combo pred for St Lucia metal type'''
    epochs = 5
    runs = 3
    loss_fn='cef'
    epsilon=0.1
    #for arch_name in ['rn50', 'rn152', 'dn121']:
    #im_hm
    #train_list = ['im_hm_binary', 'im_hm_region', 'im_hm_country']
    data_type = 'all_unverified'
    train_list = ['roof_material', 'region', 'country']
    if STARTUP:
        if data_type=='im_hm':
            df_verified, df_unverified, target_col = setup_df('im_hm_binary')
        elif data_type=='all_unverified':
            df_verified, df_unverified, target_col = setup_df('roof_material')
        create_pseudo_tt_folders(df_verified, df_unverified, data_type)
    for run in range(runs):
        for train_type in train_list:
            for arch_name in ['dn121', 'rn152', 'rn50']:
                for loss_fn in ['cef', 'lsce']:
                    arch, lr, lr_uf, bs = setup_arch(arch_name, train_type)
                    print('===============================')
                    print(f'NB_NUM: {NB_NUM}, model: {arch_name}, run: {run}, lr: {lr}, lr_uf: {lr_uf}, bs: {bs}, train_type: {train_type}, loss_fn: {loss_fn}')
    
                    df_verified, df_unverified, target_col=setup_df(train_type)
                    tfms=create_tfms()
                    data=setup_data(bs, df_verified, tfms, target_col, data_type)
                    learn=train_model(train_type, data, run, arch = arch, arch_name = arch_name, lr = lr, epochs=epochs, loss_fn=loss_fn, epsilon=epsilon)
                    learn=unfreeze_train_model(train_type, run, arch_name, learn, lr, lr_uf=lr_uf, epochs=epochs,loss_fn=loss_fn)
                    pkl_file=save_model(learn, train_type, run, arch_name, epochs, loss_fn)
                    inference_multi(df_unverified, train_type, pkl_file, arch_name, run, epochs=epochs, data_type=data_type, loss_fn=loss_fn)
                    learn.destroy()
                    gc.collect()

if __name__ == "__main__":
    workflow()

