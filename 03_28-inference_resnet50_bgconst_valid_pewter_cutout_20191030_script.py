#!/usr/bin/env python
# coding: utf-8

# ## Train and Prediction on all data
# 
# Using rotated to hz + OpenCv border
# 
# Basic default transforms



from fastai.vision import *
import geopandas as gpd
from resizeimage import resizeimage
import datetime
import uuid
from os import listdir
from os.path import isfile, join
from fastai_extensions import *

pd.set_option('display.max_columns', None)  
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)

torch.cuda.set_device(0)
torch.cuda.current_device()

data_dir = Path('data')
data_04 = Path('data_04')



img_size=256
bs=128


train_images=data_04/f'train/rotated/clipped/constant/{img_size}'
test_images=data_04/f'test/rotated/clipped/constant/{img_size}'

train_names = get_image_files(train_images)
test_names = get_image_files(test_images)

def roof_aug_split():
    df_gold = pd.read_csv(data_dir / 'df_all_repl_st_lucia_castries_gold_concrete_cement.csv')
    df_gold.drop(columns=['target'], inplace=True)
    df_gold = df_gold[['id', 'roof_material', 'verified', 'country', 'region']]

    df_pewter_70pct = pd.read_csv(data_dir / 'st_lucia_castries_gros_islet_70pct_rn50-rn152-dn121_preds.csv')
    df_pewter_70pct.tail()
    frames = [df_gold, df_pewter_70pct]
    df_gold_pewter = pd.concat(frames)

    #### test data
    df_test = pd.read_csv(data_dir / 'df_test_all.csv')
    gp_ids = df_gold_pewter.id.values.tolist()

    trn_file_names = []
    for f in train_names:
        trn_file_names.append(f.name.split('.tif')[0])

    bg_const_names = []
    for f in trn_file_names:
        if '_256_bgconstant' in f:
            bg_const_names.append(f.split('_256_bgconstant')[0])

    raw_names = []
    for f in trn_file_names:
        if '_raw' in f:
            raw_names.append(f.split('_raw')[0])

    zoom_names = []
    for f in trn_file_names:
        if '_zoom' in f:
            zoom_names.append(f.split('_zoom')[0])

    reflect_names = []
    for f in trn_file_names:
        if '_256_reflect' in f:
            reflect_names.append(f.split('_256_reflect')[0])

    wrap_names = []
    for f in trn_file_names:
        if '_256_wrap' in f:
            wrap_names.append(f.split('_256_wrap')[0])

    df_gold_pewter_bg_const = df_gold_pewter.loc[df_gold_pewter['id'].isin(bg_const_names)]
    df_gold_pewter_bg_const['id'] = df_gold_pewter_bg_const['id'] + '_256_bgconstant'
    df_gold_pewter_bg_const = df_gold_pewter_bg_const.drop_duplicates(subset=['id'])

    # Ensure is only ids with _256_bgconstant

    #### raw
    df_gold_pewter_raw = df_gold_pewter.loc[df_gold_pewter['id'].isin(raw_names)]
    df_gold_pewter_raw['id'] = df_gold_pewter_raw['id'] + '_raw'
    df_gold_pewter_raw = df_gold_pewter_raw.drop_duplicates(subset=['id'])

    #### zoom
    df_gold_pewter_zoom = df_gold_pewter.loc[df_gold_pewter['id'].isin(zoom_names)]
    df_gold_pewter_zoom['id'] = df_gold_pewter_zoom['id'] + '_zoom'
    df_gold_pewter_zoom = df_gold_pewter_zoom.drop_duplicates(subset=['id'])

    #### wrap
    df_gold_pewter_wrap = df_gold_pewter.loc[df_gold_pewter['id'].isin(wrap_names)]
    df_gold_pewter_wrap['id'] = df_gold_pewter_wrap['id'] + '_256_wrap'
    len(df_gold_pewter_wrap)
    df_gold_pewter_wrap = df_gold_pewter_wrap.drop_duplicates(subset=['id'])

    df_gold_pewter_reflect = df_gold_pewter.loc[df_gold_pewter['id'].isin(reflect_names)]
    df_gold_pewter_reflect['id'] = df_gold_pewter_reflect['id'] + '_256_reflect'
    df_gold_pewter_reflect = df_gold_pewter_reflect.drop_duplicates(subset=['id'])
    return df_gold_pewter_bg_const, df_gold_pewter_reflect, df_gold_pewter_wrap, df_gold_pewter_zoom, df_gold_pewter_raw

def get_train_val_test(roof_combinations):
    df_gold_pewter = pd.concat(roof_combinations)

    # ### split df manally
    df_val = df_gold_pewter.sample(frac=0.07)
    df_val_ids = df_val.id.values.tolist()

    core_ds = []
    for id in df_val_ids:
        core_ds.append(id.split('_')[0])

    core_ds = list(set(core_ds))
    df_val = df_gold_pewter[df_gold_pewter['id'].str.contains('|'.join(core_ds))]
    df_train = df_gold_pewter[~df_gold_pewter['id'].str.contains('|'.join(core_ds))]
    df_val['is_valid'] = True
    df_train['is_valid'] = False
    frames = [df_val, df_train]
    df_train = pd.concat(frames)
    return df_train, df_val


def get_train_val_split(num_splits, postfix=None):
    '''get all split permutations in one go so can pred all val data'''
    df_all = pd.read_csv(data_dir / 'df_train_all.csv')
    if postfix:
        df_all['id']=df_all['id']+postfix
    #shuffle
    df_all = df_all.sample(frac=1).reset_index(drop=True)
    df_splits=np.array_split(df_all, num_splits)
    temp=[]
    for s in df_splits:
        temp.append(len(s))
    assert sum(temp)==len(df_all)
    split_list=[]
    for s in range(num_splits):
        df_val = df_splits[s]
        df_val['is_valid']=True
        trn_dfs = [x for i, x in enumerate(df_splits) if i != s]
        df_i = pd.concat(trn_dfs)
        df_train = pd.concat([df_i, df_val])
        df_train['is_valid'] = False
        assert len(df_train)==len(df_all)
        split_list.append(df_train)
    return split_list

def setup_dataset(df_train):
    cutout_1 = cutout(n_holes=(1, 4), length=(10, 20), p=.5)
    cutout_2 = cutout(n_holes=(1, 4), length=(10, 20), p=.5)
    cutout_3 = cutout(n_holes=(1, 4), length=(20, 20), p=.6)
    cutout_4 = cutout(n_holes=(1, 4), length=(20, 20), p=.6)
    xtra_tfms = [cutout_1, cutout_2, cutout_3, cutout_4, rand_crop(p=0.4), rand_zoom(scale=(1., 1.5), p=0.4)]
    tfms = get_transforms(flip_vert=True, max_lighting=0.25, max_warp=0., xtra_tfms=xtra_tfms)

    # ### setup dataset
    np.random.seed(42)
    dep_var = 'roof_material'
    src = (ImageList.from_df(path=train_images, df=df_train, cols='id', suffix='.tif')
           # .split_by_rand_pct(0.1)
           .split_from_df(col='is_valid')
           .label_from_df(cols=dep_var)
           .add_test_folder(test_images))

    data = (src.transform(tfms, size=img_size)
            .databunch(bs=bs).normalize(imagenet_stats))
    return data


def index_valid(idxl, learn):
    '''return loss name by order of top loss'''
    lossidx_img={}
    for c, i in enumerate(idxl):
        f=learn.data.valid_ds.items[i]
        lossidx_img[c]=f
    return lossidx_img

def get_img_id(all_images):
    ids=[]
    for img_name in all_images:
        end=img_name.split('/')[-1]
        ids.append(end.split('.tif')[0])
    return ids

def confusion_dump(learn, interp, arch_name, combo_name, run_num):
    all_losses=interp.top_losses(len(interp.losses), largest=True)
    all_idx=all_losses[1].tolist()
    all_loss=all_losses[0].tolist()
    all_images=index_valid(all_idx, learn)
    all_images=list(all_images.values())

    ids=get_img_id(all_images)
    assert len(ids)==len(all_loss)

    # list of strings
    columns=['id','loss']
    df = pd.DataFrame(list(zip(ids, all_loss)),
                   columns =columns)
    mname=f'{arch_name}-{NB_NUM}-{MODEL_NAME}-{tfm_name}-{DATE}-{UID}-{combo_name}-{run_num}'
    print(f'saving {mname}.csv')
    df['model']=mname
    df.to_csv(data_dir/f'processing/model_confusion_qc/{mname}.csv')

def confusion_loss(learn, preds, pred_classes, tl_val, tl_idx, losses, arch_name, combo_name, run_num, model_file, postfix='_preds'):
    print('>>confusion_loss()')

    all_images=index_valid(tl_idx, learn)
    all_images=list(all_images.values())
    ids=get_img_id(all_images)
    assert len(ids)==len(losses)

    columns=['id','pred','loss']
    df = pd.DataFrame(list(zip(ids,pred_classes,losses)), columns =columns)
    mname=f'{model_file}{postfix}'
    df['model']=model_file
    df.to_csv(data_dir/f'processing/model_confusion_qc/{mname}.csv')

def get_top_losses(interp, k=20, largest=True):
    tl_val,tl_idx = interp.top_losses(k, largest)
    classes = interp.data.classes
    preds=[]
    pred_classes=[]
    losses=[]
    for i,idx in enumerate(tl_idx):
        im,cl = interp.data.dl(interp.ds_type).dataset[idx]
        cl = int(cl)
        loss=f'{interp.losses[idx]:.2f}'
        pred=f'{interp.preds[idx][cl]:.2f}'
        pred_class=str(classes[interp.pred_class[idx]])
        preds.append(pred)
        pred_classes.append(pred_class)
        losses.append(loss)
    return preds, pred_classes, tl_val, tl_idx, losses

def create_model(data, arch):
    learn = cnn_learner(data, arch, metrics=error_rate, bn_final=True).to_fp16()
    return learn

def create_preds(run, model_file, arch_name, learn, combo_name, postfix):
    mname=f'{model_file}'
    learn.load(f'{mname}')
    interp = ClassificationInterpretation.from_learner(learn)
    preds, pred_classes, tl_val, tl_idx, losses=get_top_losses(interp, k=len(interp.losses))
    confusion_loss(learn, preds, pred_classes, tl_val, tl_idx, losses, arch_name, combo_name, run, model_file, postfix)

def load_preds(model_file, combo_name, plot=False):
    dfs=[]
    mname=f'{model_file}'
    df=pd.read_csv(data_dir/f'processing/model_confusion_qc/{mname}.csv')
    df.drop(columns=['Unnamed: 0'],inplace=True)
    if plot:
        df.plot(title=f'{model_file} {len(df)}')
    dfs.append(df)

    df_loss = pd.concat(dfs)
    if plot:
        df_loss.plot.hist(alpha=0.5, bins=100)

    # #### group duplicated ids and force to one pred
    dup_df=df_loss[df_loss.duplicated(['id'])]
    non_dup_df=df_loss[~df_loss.duplicated(['id'])]
    dup_df.sort_values(by=['id', 'loss'])
    #TODO check this - was temp df
    df = dup_df[['id','model','pred','loss']]
    return df, non_dup_df

def get_mode_per_column(dataframe, group_cols, col):
    return (dataframe.fillna(-1)  # NaN placeholder to keep group 
            .groupby(group_cols + [col])
            .size()
            .to_frame('count')
            .reset_index()
            .sort_values('count', ascending=False)
            .drop_duplicates(subset=group_cols)
            .drop(columns=['count'])
            .sort_values(group_cols)
            .replace(-1, np.NaN))  # restore NaNs

def regroup_df(df, non_dup_df):
    #ODO check
    #group_cols = ['id', 'model_name']
    group_cols = ['id', 'model']
    non_grp_cols = list(set(df).difference(group_cols))
    output_df = get_mode_per_column(df, group_cols, non_grp_cols[0]).set_index(group_cols)
    for col in non_grp_cols[1:]:
        output_df[col] = get_mode_per_column(df, group_cols, col)[col].values

    output_df=output_df.reset_index()
    #TODO check
    #output_df.rename(columns={'model_name': 'model'}, inplace=True)
    output_df=output_df[['id', 'pred', 'loss', 'model']]
    non_dup_df.head()
    mean_df= pd.concat([output_df, non_dup_df])
    ids=mean_df.id.unique()

    unids=[]
    for id in ids:
        b=len(id.split('_')[0])
        unids.append(id[b+1:])
    return mean_df, unids

def create_tfm_df(df_preds, df_all):
    dup_zoom_id = df_preds[df_preds.duplicated(['id'], keep=False)]
    # #### containd duplicte indices
    df_preds = df_preds.set_index(["id"])
    df_all = df_all.set_index(["id"])

    df_preds['id'] = df_preds.index
    df_all['id'] = df_all.index
    return df_preds, df_all

def create_df_for_tf(mean_df, df_all_tf, tf_name):
    preds = mean_df[mean_df['id'].str.contains(tf_name)]
    preds_, df_all_ = create_tfm_df(preds, df_all_tf)
    return preds_, df_all_


def get_correct_incorrect_df(df_in, df_preds):
    #df_in eg df_gold_pewter_zoom
    #df_preds eg zoom_preds
    df = df_in.copy()

    df['pred']='none'
    df['model']='none'
    df['loss']=-999.25
    df['result']=False

    for row in df_preds.itertuples():
        id=getattr(row, "id")
        pred=getattr(row, "pred")
        loss=getattr(row, "loss")
        model=getattr(row, "model")
        df.at[id, 'loss'] = loss
        df.at[id, 'pred'] = str(pred)
        df.at[id, 'model'] = str(model)

    df['result']=df['pred'].equals(df['roof_material'])
    #df_incorrect=df.loc[(df['pred'] != 'none') & df['result']==False]
    #df_correct = df.loc[(df['pred'] != 'none') & df['result'] == True]
    #df_incorrect=df_incorrect.sort_values(by='loss', ascending=0)
    return df

def save_df(mname, df):
    df.to_csv(data_dir/f'processing/model_confusion_qc/{mname}.csv')

def get_dataset():
    df_gold_pewter_bg_const, df_gold_pewter_reflect, df_gold_pewter_wrap, df_gold_pewter_zoom, df_gold_pewter_raw = roof_aug_split()
    roof_combinations = [df_gold_pewter_bg_const, df_gold_pewter_reflect, df_gold_pewter_wrap, df_gold_pewter_zoom]
    df_train, df_val = get_train_val_test(roof_combinations)
    data = setup_dataset(df_train)
    return data, df_gold_pewter_bg_const, df_gold_pewter_reflect, df_gold_pewter_wrap, df_gold_pewter_zoom, df_gold_pewter_raw

def model_inference(data, runs, model_file, arch, arch_name, combo_name, postfix):
    learn=create_model(data, arch)
    create_preds(runs, model_file, arch_name, learn, combo_name, postfix)


def runner():
    # stage-2-rn50-03_28-bg_const-20191201-0c79e3be-raw_zoom
    background = 'constant'
    arch = models.resnet50
    arch_name='rn50'
    runs=5
    model_name = 'bg_const'
    tfm_name = 'cutout'
    nb_num = '03_28'
    date = '20191201'
    uid = '0c79e3be'
    combo_name='raw_zoom'
    special_code='_rz'
    final_stage='s2'
    pred_postfix='_preds'
    '''
    #TODO
    #stage-2-rn50-03_28-bg_const-20191201-0c79e3be-const_ref_wrap_zoom
    #TODO    
    #stage-2-rn50-03_30-bg_const-20191201-fb9c67aa-raw_zoom
    background = 'constant'
    arch = models.resnet50
    arch_name='rn50'
    runs=5
    MODEL_NAME = 'bg_const'
    tfm_name = 'mixup'
    NB_NUM = '03_30'
    DATE = '20191201'
    UID = 'fb9c67aa'
    combo_name = 'raw_zoom'
    '''
    data, df_gold_pewter_bg_const, df_gold_pewter_reflect, df_gold_pewter_wrap, df_gold_pewter_zoom, df_gold_pewter_raw = get_dataset()

    for run in range(runs):
        run_code='r'+str(run)
        model_file=f'{nb_num}-{uid}{special_code}-{final_stage}-{run_code}-{date}'
        print(f'running {model_file}')
        model_inference(data, runs, model_file, arch, arch_name, combo_name, pred_postfix)

        df, non_dup_df=load_preds(model_file+f'{pred_postfix}',combo_name,plot=False)
        mean_df, unids=regroup_df(df, non_dup_df)
        if combo_name=='raw_zoom':
            frames=[df_gold_pewter_raw, df_gold_pewter_zoom]
        elif combo_name == 'const_ref_wrap_zoom':
            frames = [df_gold_pewter_bg_const, df_gold_pewter_reflect, df_gold_pewter_wrap, df_gold_pewter_zoom]

        df = pd.concat(frames)
        preds, df_all= create_df_for_tf(mean_df, df, combo_name)
        result_df=get_correct_incorrect_df(df_all, preds)
        save_df(model_file+'_result', result_df)


if __name__ == "__main__":
    runner()


