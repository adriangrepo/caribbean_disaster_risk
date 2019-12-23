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




torch.cuda.set_device(0)
torch.cuda.current_device()

data_dir = Path('data')
data_04 = Path('data_04')

MODEL_NAME='bg_const'
NB_NUM='03_28'

DATE = datetime.datetime.today().strftime('%Y%m%d')
UID=str(uuid.uuid4())[:8]


DATE = '20191201'
UID = '0c79e3be'
print(f'UID: {UID}, DATE: {DATE}')

SUB_NUM='30'

img_size=256
bs=128
background='constant'

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

def create_learner(data):
    # ### Model

    arch = models.resnet50
    arch_name = 'rn50'

    learn = cnn_learner(data, arch, metrics=error_rate, bn_final=True).to_fp16()
    return learn, arch_name

def train_model(learn, run_num, arch_name, combo_name, lr = 1e-2):

    learn.fit_one_cycle(5, slice(lr))
    learn.save(f'stage-1-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}-{combo_name}-{run_num}')

    interp = ClassificationInterpretation.from_learner(learn)
    print(interp.most_confused(min_val=2))

    learn.unfreeze()
    learn.fit_one_cycle(3, slice(1e-6, lr / 5))
    learn.save(f'stage-2-{arch_name}-{NB_NUM}-{MODEL_NAME}-{DATE}-{UID}-{combo_name}-{run_num}')

def run_sequence(run_num, roof_combinations, combo_name):
    #each run we get a different data split
    df_train, df_val=get_train_val_test(roof_combinations)
    data=setup_dataset(df_train)
    learn, arch_name=create_learner(data)
    train_model(learn, run_num, arch_name, combo_name, lr=1e-2)
    learn.destroy()
    gc.collect()


def runner():
    # ### using valid + gold preds instead of all
    df_gold_pewter_bg_const, df_gold_pewter_reflect, df_gold_pewter_wrap, df_gold_pewter_zoom, df_gold_pewter_raw = roof_aug_split()
    const_ref_wrap_zoom_combo = [df_gold_pewter_bg_const, df_gold_pewter_reflect, df_gold_pewter_wrap, df_gold_pewter_zoom]
    combo_name='const_ref_wrap_zoom'
    for i in range(5):
        print(f'combo_name: {combo_name} run: {i}')
        run_sequence(i, const_ref_wrap_zoom_combo, combo_name)

    raw_zoom_combo = [df_gold_pewter_zoom, df_gold_pewter_raw]
    combo_name='raw_zoom'
    for i in range(5):
        print(f'combo_name: {combo_name} run: {i}')
        run_sequence(i, raw_zoom_combo, combo_name)

if __name__ == "__main__":
    runner()



