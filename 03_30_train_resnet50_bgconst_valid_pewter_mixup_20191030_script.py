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
from PIL import Image as pil_image

torch.cuda.set_device(2)
torch.cuda.current_device()

data_dir = Path('data')
data_04 = Path('data_04')

MODEL_NAME='bg_const'

NB_NUM='03_30'
tfm_name='mixup'

DATE = datetime.datetime.today().strftime('%Y%m%d')
UID=str(uuid.uuid4())[:8]
print(f'UID: {UID}, DATE: {DATE}')                                                 

#DATE = '20191201'
#UID = 'fb9c67aa'

SUB_NUM='30'

img_size=256
bs=128
background='constant'

train_images=data_04/f'train/rotated/clipped/constant/{img_size}'
test_images=data_04/f'test/rotated/clipped/constant/{img_size}'

train_names = get_image_files(train_images)
test_names = get_image_files(test_images)

# ### using valid + gold preds instead of all
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

    # ### split df manually
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

def get_train_val_test_splits(num_splits, roof_combinations):
    '''get all split permutations in one go so can pred all val data'''
    df_all =  pd.concat(roof_combinations)

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

def get_train_val_all(frac):
    df_all = pd.read_csv(data_dir / 'df_train_all.csv')
    df_train = df_all.sample(frac=frac)
    train_ids=df_train.id.values.tolist()
    all_ids=df_all.id.values.tolist()
    val_ids = list(set(all_ids) - set(train_ids))
    df_val = df_all.loc[df_all['id'].isin(val_ids)]
    df_val['is_valid']==True
    df_train['is_valid']==False
    df_tall= pd.concat([df_train,df_val])
    assert len(df_tall)==len(df_all)
    return df_tall, df_val

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

def get_transforms_cutout():
    cutout_1 = cutout(n_holes=(1, 4), length=(10, 20), p=.5)
    cutout_2 = cutout(n_holes=(1, 4), length=(10, 20), p=.5)
    cutout_3 = cutout(n_holes=(1, 4), length=(20, 20), p=.6)
    cutout_4 = cutout(n_holes=(1, 4), length=(20, 20), p=.6)
    xtra_tfms = [cutout_1, cutout_2, cutout_3, cutout_4, rand_crop(p=0.4), rand_zoom(scale=(1., 1.5), p=0.4)]
    tfms = get_transforms(flip_vert=True, max_lighting=0.25, max_warp=0., xtra_tfms=xtra_tfms)
    return tfms


def get_transforms_crop():
    xtra_tfms = [rand_crop(p=0.4), rand_zoom(scale=(1., 1.5), p=0.4)]
    tfms = get_transforms(flip_vert=True, max_lighting=0.25, max_warp=0., xtra_tfms=xtra_tfms)
    return tfms

def check_images_exist(df_train):
    ids=df_train.id.values.tolist()
    onlyfiles = [f for f in listdir(train_images) if isfile(join(train_images, f))]
    onlyfiles = [f for f in onlyfiles if f.endswith('.tif')]
    #assert (all(x.strip() in test_list for x.strip() in sub_list))
    missing = list(set(ids) - set(onlyfiles))
    print(missing)
    assert set(ids).issubset(set(onlyfiles))

# ### setup dataset
def setup_dataset(df_train, tfms):

    # ### setup dataset
    np.random.seed(42)
    dep_var = 'roof_material'
    print(df_train.head())
    print(len(df_train))
    check_images_exist(df_train)
    src = (ImageList.from_df(path=train_images, df=df_train, cols='id', suffix='.tif')
           # .split_by_rand_pct(0.1)
           .split_from_df(col='is_valid')
           .label_from_df(cols=dep_var)
           .add_test_folder(test_images))

    data = (src.transform(tfms=tfms, size=img_size)
            .databunch(bs=bs).normalize(imagenet_stats))
    return data

def create_learner(data):
    # ### Model
    arch = models.resnet50
    arch_name = 'rn50'

    learn = cnn_learner(data, arch, metrics=error_rate, bn_final=True).to_fp16()
    learn = learn.mixup()
    return learn, arch_name

def train_model(learn, nb_num, uid, special_code, run_num, date, lr = 1e-2, epochs=5):

    learn.fit_one_cycle(epochs, slice(lr))
    learn.save(f'{nb_num}-{uid}{special_code}-p1-r{run_num}-{date}')

    learn.unfreeze()
    learn.fit_one_cycle(epochs, slice(1e-6, lr / 5))

    learn.save(f'{nb_num}-{uid}{special_code}-s2-r{run_num}-{date}')

    return learn

def show_n_images(im_list):
    for iname in im_list:
        #plt.imshow(im, interpolation = 'bicubic')
        im = pil_image.open(iname)
        plt.imshow(im)
        plt.title(iname)
        plt.show()

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

def confusion_dump(learn, nb_num, uid, special_code, date, run_num):
    interp = ClassificationInterpretation.from_learner(learn)
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
    mname=f'{nb_num}-{uid}{special_code}-s2-r{run_num}-{date}'
    print(f'saving {mname}.csv')
    df['model']=mname
    df.to_csv(data_dir/f'processing/model_confusion_qc/{mname}.csv')

def run_sequence(run_num, roof_combinations, nb_num, uid, special_code, date, epochs, aug_type=None):
    #each run we get a different data split
    df_train, df_val=get_train_val_test(roof_combinations)
    if aug_type == 'cutout':
        data=setup_dataset(df_train, tfms=get_transforms_cutout())
    else:
        data=setup_dataset(df_train, tfms=get_transforms_crop())
    learn, arch_name=create_learner(data)
    if aug_type == 'mixup':
        learn=learn.mixup()
    elif aug_type == 'ricap':
        learn = learn.ricap()

    learn=train_model(learn, nb_num, uid, special_code, run_num, date, lr=1e-2, epochs=epochs)
    #confusion_dump(learn, interp, arch_name, combo_name, run_num)

def run_splits(df_train, run_num, nb_num, uid, special_code, date, epochs, aug_type=None):
    #each run we get a different data split
    if aug_type == 'cutout':
        data=setup_dataset(df_train, tfms=get_transforms_cutout())
    else:
        data=setup_dataset(df_train, tfms=get_transforms_crop())
    learn, arch_name=create_learner(data)
    if aug_type == 'mixup':
        learn=learn.mixup()
    elif aug_type == 'ricap':
        learn = learn.ricap()
    learn=train_model(learn, nb_num, uid, special_code, run_num, date,  lr=1e-2, epochs=epochs)
    #confusion_dump(learn, interp, arch_name, combo_name, run_num)

def run_all(run_num, df_train, nb_num, uid, special_code, date, epochs, aug_type=None):
    #each run we get a different data split
    if aug_type == 'cutout':
        data=setup_dataset(df_train, tfms=get_transforms_cutout())
    else:
        data=setup_dataset(df_train, tfms=get_transforms_crop())
    learn, arch_name=create_learner(data)
    if aug_type == 'mixup':
        learn=learn.mixup()
    elif aug_type == 'ricap':
        learn = learn.ricap()
    learn=train_model(learn, nb_num, uid, special_code, run_num, date,  lr=1e-2, epochs=epochs)
    #confusion_dump(learn, interp, arch_name, combo_name, run_num)

def runner():
    #all training data
    combo_name='all'
    frac=0.8
    num_splits=5
    epochs=5
    nb_num = '03_30'
    date = DATE
    uid = UID
    print(DATE)
    print(UID)
    final_stage = 's2'


    # ### using valid + gold preds instead of all
    df_gold_pewter_bg_const, df_gold_pewter_reflect, df_gold_pewter_wrap, df_gold_pewter_zoom, df_gold_pewter_raw = roof_aug_split()
    reflect_zoom_combo=[df_gold_pewter_reflect, df_gold_pewter_zoom]
    split_list = get_train_val_test_splits(num_splits, reflect_zoom_combo)
    combo_name='_refz'
    epochs = 5
    for i in range(num_splits):
        aug_type = 'mixup'
        special_code = '_refz_mi'
        print(f'combo_name: {combo_name} run: {i}')
        run_splits(split_list[i], i, nb_num, uid, special_code, date, epochs, aug_type)
    for i in range(num_splits):
        aug_type = 'cutout'
        special_code = '_refz_cu'
        print(f'combo_name: {combo_name} run: {i}')
        run_splits(split_list[i], i, nb_num, uid, special_code, date, epochs, aug_type)
    for i in range(num_splits):
        aug_type = 'ricap'
        special_code = '_refz_ri'
        print(f'combo_name: {combo_name} run: {i}')
        run_splits(split_list[i], i, nb_num, uid, special_code, date, epochs, aug_type)


    raw_zoom_combo = [df_gold_pewter_zoom, df_gold_pewter_raw]
    split_list = get_train_val_test_splits(num_splits, raw_zoom_combo)
    combo_name='_rawz'
    epochs = 5
    for i in range(num_splits):
        aug_type = 'mixup'
        special_code = '_rawz_mi'
        print(f'combo_name: {combo_name} run: {i}')
        run_splits(split_list[i], i, nb_num, uid, special_code, date, epochs, aug_type)
    for i in range(num_splits):
        aug_type = 'cutout'
        special_code = '_rawz_cu'
        print(f'combo_name: {combo_name} run: {i}')
        run_splits(split_list[i], i, nb_num, uid, special_code, date, epochs, aug_type)
    for i in range(num_splits):
        aug_type = 'ricap'
        special_code = '_rawz_ri'
        print(f'combo_name: {combo_name} run: {i}')
        run_splits(split_list[i], i, nb_num, uid, special_code, date, epochs, aug_type)

    split_list=get_train_val_split(num_splits, postfix='_raw')
    for i in range(num_splits):
        aug_type = 'mixup'
        combo_name = 'all_'+aug_type
        special_code = '_a_mi'
        print(f'combo_name: {combo_name} run: {i}')
        run_all(i, split_list[i], nb_num, uid, special_code, date, epochs,aug_type)
    for i in range(num_splits):
        aug_type = 'cutout'
        combo_name = 'all_' + aug_type
        special_code = '_a_cu'
        print(f'combo_name: {combo_name} run: {i}')
        run_all(i, split_list[i], nb_num, uid, special_code, date, epochs,aug_type)
    for i in range(num_splits):
        aug_type = 'ricap'
        combo_name = 'all_' + aug_type
        special_code = '_a_ri'
        print(f'combo_name: {combo_name} run: {i}')
        run_all(i, split_list[i], nb_num, uid, special_code, date, epochs,aug_type)


if __name__ == "__main__":
    runner()