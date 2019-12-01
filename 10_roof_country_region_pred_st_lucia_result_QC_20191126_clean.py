#!/usr/bin/env python
# coding: utf-8

# ## Predict St Lucia Metal Roofs
# 
# Using rotated to hz + OpenCv border
# 






from fastai.vision import *
import geopandas as gpd
from resizeimage import resizeimage
import datetime
import uuid
from os import listdir
from os.path import isfile, join
from ipyexperiments import *
import pandas as pd




torch.cuda.set_device(1)
torch.cuda.current_device()




data_dir = Path('data')




MODEL_NAME='result_QC'




NB_NUM='10_0'
RUN='0'




DATE = datetime.datetime.today().strftime('%Y%m%d')
UID=str(uuid.uuid4())[:8]
print(f'UID: {UID}, DATE: {DATE}')                                                 




#DATE = '20191124'
#UID = '1328c588'




SUB_NUM='2'




img_size=256
bs=128




train_images=data_dir/f'train/rotated/clipped/{img_size}'
test_images=data_dir/f'test/rotated/clipped/{img_size}'




batch_runs_df=pd.read_csv(data_dir/f'processing/8daa24a2_clean.txt', delimiter=':', names=["epoch", "valid_loss", "model"])




batch_runs_df.head()




models=batch_runs_df.model.unique()




model_dfs={}
useable_df={}




for model in models:
    df = batch_runs_df.loc[batch_runs_df['model']==model]
    model_dfs[model]=df
    if df.valid_loss.min()<0.6:
        useable_df[model]=df
        print(df.valid_loss.min())




useable_df.keys()




len(useable_df)




for key, value in useable_df.items():
    plt.figure()
    x = value['epoch']
    y1 = value['valid_loss']
    plt.plot(x,y1)
    #plt.ylim(0, 1)
    plt.title(key)


# No Label Smoothing models were any good



en_df=pd.read_csv(data_dir/f'processing/3c19f55d_run.txt', delimiter=':', names=["epoch", "valid_loss", "model"])




en_df.head(n=20)




en_df=en_df.drop(en_df.index[12])




en_df.head(n=20)




useable_df['best-efficient_net_b4-06_16-efficientnet-20191128-3c19f55d'] = en_df




list(useable_df.keys())




plt.figure()
x = en_df['epoch']
y1 = en_df['valid_loss']
plt.plot(x,y1)
#plt.ylim(0, 1)
plt.title(key)









def gen_df(file_):
    df = pd.read_csv(data_dir/f'processing/{file_}')
    if file_=='efficientnet_b4-06_16-efficientnet-20191128-3c19f55d.csv':
        print('efficientnet_b4')
        df['model']='efficientnet_b4'
        df['run']=0
        df['ttype']='cef'
        df['uid']='3c19f55d'
    else:
        df['model']=file_.split('-')[0]
        pfix=file_.split('10_')[1]
        run=pfix.split('-')[0]
        df['run']=run
        ttype=pfix.split(f'{run}-')[1].split('-')[0]
        df['ttype']=ttype
        df['uid']=file_
    return df


# #### concat predictions



def concat_preds_1():
    c_list_ = []
    rm_list=[]
    region_list=[]
    data_folder = data_dir/'processing/1_without_cbs/'
    for file_ in os.listdir(data_folder):
        if file_.endswith('.csv'):
            if 'country' in file_:
                c_list_.append(gen_df(file_))
            elif 'roof_material' in file_:
                rm_list.append(gen_df(file_))
            elif 'efficientnet' in file_:
                rm_list.append(gen_df(file_))
            elif 'region' in file_:
                region_list.append(gen_df(file_))
    df_sl_c = pd.concat(c_list_)
    df_sl_rm = pd.concat(rm_list)
    df_sl_region = pd.concat(region_list)




useable_list=[]
for i in list(useable_df.keys()):
    fi=i.strip()
    #print(fi)
    if 'best-efficient' in fi:
        print(fi)
        fi = 'efficientnet_b4-06_16-efficientnet-20191128-3c19f55d'
    useable_list.append(fi)




def concat_preds_2():
    rm_list=[]
    c_list=[]
    r_list=[]
    data_folder = data_dir/'processing/'
    l=useable_list
    ls=[]
    f_list=[]
    for i in l:
        ls.append(i.strip())
    for file_ in os.listdir(data_folder):
        if file_.endswith('.csv'):
            f = file_.replace("-all_unverified-", "-")
            f=f.strip()
            if f.split('.csv')[0] in ls:
                if 'roof_material' in f:
                    rm_list.append(gen_df(f))
                elif 'country' in f:
                    c_list.append(gen_df(f))
                elif 'region' in f:
                    r_list.append(gen_df(f))
                elif 'efficientnet' in f:
                    print('efficientnet')
                    rm_list.append(gen_df(f))
                else:
                    print(f.split('.csv')[0])
                f_list.append(f.split('.csv')[0])
    df_roofmat = pd.concat(rm_list)
    df_cntry = pd.concat(c_list)
    df_region = pd.concat(c_list)
    return df_roofmat, df_cntry, df_region, f_list




#useable_list




#f_list




df_roofmat, df_cntry, df_region, f_list=concat_preds_2()




dif_list = list(set(useable_list) - set(f_list))




#assert len(dif_list)==0




df_roofmat.head()




df_roofmat.uid.unique()




df_region.head()




df_region.model.unique()




df_cntry.head()




df_cntry.model.unique()




def get_c_r_by_model_type(df, model, ttype):
    df = df.loc[(df['model'] == model) & (df['ttype']== ttype)]
    df.drop(columns=['id','model','run','ttype','uid'],inplace=True)
    classes=df.idxmax(axis=1)
    df['roof_material']=classes
    return df




def print_lengths(ls):
    for l in ls:
        print(len(l))




def country_to_roof(df_in):
    ttype='region'
    model_dfs={}
    for model in ['rn50','rn152', 'dn121']:
        df = df_in.copy()
        cols=list(df)
        hm=[]
        im=[]
        cc=[]
        ot=[]
        inc=[]
        healthy_metals = [s for s in cols if 'healthy_metal' in s]
        irregular_metals = [s for s in cols if 'irregular_metal' in s]
        concrete_cements = [s for s in cols if 'concrete_cement' in s]
        others = [s for s in cols if 'other' in s]
        incompletes = [s for s in cols if 'incomplete' in s]

        df["healthy_metal"] = df[healthy_metals].max(axis=1)
        df["irregular_metal"] = df[irregular_metals].max(axis=1)
        df["concrete_cement"] = df[concrete_cements].max(axis=1)
        df["other"] = df[others].max(axis=1)
        df["incomplete"] = df[incompletes].max(axis=1)

        # #### drop region/country
        df.drop(columns=healthy_metals, inplace=True)
        df.drop(columns=irregular_metals, inplace=True)
        df.drop(columns=concrete_cements, inplace=True)
        df.drop(columns=others, inplace=True)
        df.drop(columns=incompletes, inplace=True)
        print(df.head())
        df = get_c_r_by_model_type(df, model, ttype)
        print(df.head())
        model_dfs[model]=df
    return model_dfs




df_region.head()




model_dfs=country_to_roof(df_region)




df_region_rn50=model_dfs['rn50']
df_region_rn152=model_dfs['rn152']
df_region_dn121=model_dfs['dn121']




df_region_rn50.head()




def plot_df_types(df1, df2, df3):
    ax = df1.roof_material.value_counts().plot(kind='bar', color='blue', width=.75, legend=True, alpha=0.2)
    ax1 = df2.roof_material.value_counts().plot(kind='bar', color='green', width=.5, legend=True, alpha=0.2)
    df3.roof_material.value_counts().plot(kind='bar', color='maroon', width=.25, alpha=0.5, legend=True)




plot_df_types(df_region_rn50, df_region_rn152, df_region_dn121)




##### Tuesday night up to here
















df_binary_rn50.head()




print_lengths([df_binary_rn50, df_region_rn50, df_country_rn50])




df_binary_rn152 = get_all_model_type('rn152', 'binary')
df_region_rn152 = get_all_model_type('rn152', 'region')
df_country_rn152 = get_all_model_type('rn152', 'country')




plot_df_types(df_binary_rn152, df_region_rn152, df_country_rn152)




print_lengths([df_binary_rn152, df_region_rn152, df_country_rn152])




df_binary_dn121 = get_all_model_type('dn121', 'binary')
df_region_dn121 = get_all_model_type('dn121', 'region')
df_country_dn121 = get_all_model_type('dn121', 'country')




plot_df_types(df_binary_dn121, df_region_dn121, df_country_dn121)




print_lengths([df_binary_dn121, df_region_dn121, df_country_dn121])




#### Healthy metal




df_hm=df_sl_metal[['id','healthy_metal']]




len(df_hm)




result = df_hm.groupby(['id'], as_index=False).agg(
                     ['mean', 'std'])




len(result)




result.head()


# Sort by tuple



result.sort_values(by=('healthy_metal','std'),ascending=False).head()


# Or drop header



result.columns = result.columns.droplevel(0)




result.head()




result=result.sort_values(by=('std'),ascending=False).reset_index()




result.head()




result.plot.hist(bins=100, alpha=0.3)


# Try as first pass using 0.8 mean as cutoff



hm_10pct_stddev=result.loc[result['std'] <= 0.1]
hm_20pct_stddev=result.loc[result['std'] <= 0.2]




hm_80pct_plus=result.loc[result['mean'] >= 0.8]
hm_70pct_plus=result.loc[result['mean'] >= 0.7]




len(hm_80pct_plus)/len(result)


# #### Irregular metal



df_im=df_sl_metal[['id','irregular_metal']]




result_im = df_im.groupby(['id'], as_index=False).agg(
                     ['mean', 'std'])




result_im.columns = result_im.columns.droplevel(0)




result_im.plot.hist(bins=100, alpha=0.3)




im_10pct_stddev=result_im.loc[result_im['std'] <= 0.1]
im_20pct_stddev=result_im.loc[result_im['std'] <= 0.2]




im_80pct_plus=result_im.loc[result_im['mean'] >= 0.8]
im_70pct_plus=result_im.loc[result_im['mean'] >= 0.7]




len(im_80pct_plus)/len(result_im)


# #### Save these as 'silver' validated



im_10pct_stddev['roof_type']= 'irregular_metal'
im_20pct_stddev['roof_type']= 'irregular_metal'

hm_10pct_stddev['roof_type']= 'healthy_metal'
hm_20pct_stddev['roof_type']= 'healthy_metal'




im_80pct_plus['roof_type']= 'irregular_metal'
hm_80pct_plus['roof_type']= 'healthy_metal'




im_70pct_plus['roof_type']= 'irregular_metal'
hm_70pct_plus['roof_type']= 'healthy_metal'




im_80pct_plus.head()




im_80pct_plus.drop(columns=['mean','std'],inplace=True)
hm_80pct_plus.drop(columns=['mean','std'],inplace=True)
im_70pct_plus.drop(columns=['mean','std'],inplace=True)
hm_70pct_plus.drop(columns=['mean','std'],inplace=True)

im_10pct_stddev.drop(columns=['mean','std'],inplace=True)
im_20pct_stddev.drop(columns=['mean','std'],inplace=True)
hm_10pct_stddev.drop(columns=['mean','std'],inplace=True)
hm_20pct_stddev.drop(columns=['mean','std'],inplace=True)




silver_val_80pct_rooftypes=im_80pct_plus.append(hm_80pct_plus).reset_index()
silver_val_70pct_rooftypes=im_70pct_plus.append(hm_70pct_plus).reset_index()

silver_val_10pct_std_rooftypes=im_10pct_stddev.append(hm_10pct_stddev).reset_index()
silver_val_20pct_std_rooftypes=im_20pct_stddev.append(hm_20pct_stddev).reset_index()




silver_val_80pct_rooftypes = silver_val_80pct_rooftypes.rename(columns={'index': 'id', 'id': 'drop'})
silver_val_70pct_rooftypes = silver_val_70pct_rooftypes.rename(columns={'index': 'id', 'id': 'drop'})

silver_val_10pct_std_rooftypes=silver_val_10pct_std_rooftypes.rename(columns={'index': 'id', 'id': 'drop'})
silver_val_20pct_std_rooftypes=silver_val_20pct_std_rooftypes.rename(columns={'index': 'id', 'id': 'drop'})




silver_val_80pct_rooftypes.drop(columns=['drop'],inplace=True)
silver_val_70pct_rooftypes.drop(columns=['drop'],inplace=True)

silver_val_10pct_std_rooftypes.drop(columns=['drop'],inplace=True)
silver_val_20pct_std_rooftypes.drop(columns=['drop'],inplace=True)




len(silver_val_80pct_rooftypes)




len(silver_val_70pct_rooftypes)


# these should not be > dataset length, dont use for now



len(silver_val_10pct_std_rooftypes)




len(silver_val_20pct_std_rooftypes)




len(df_sl_metal)




silver_val_80pct_rooftypes.to_csv(data_dir/'st_lucia_80pct_silver_healthy_irregular_metal.csv', index=False)
silver_val_70pct_rooftypes.to_csv(data_dir/'st_lucia_70pct_silver_healthy_irregular_metal.csv', index=False)

#silver_val_10pct_std_rooftypes.to_csv(data_dir/'st_lucia_silver_10pct_std_dev_healthy_irregular_metal.csv', index=False)
#silver_val_20pct_std_rooftypes.to_csv(data_dir/'st_lucia_silver_20pct_std_dev_healthy_irregular_metal.csv', index=False)




### all pred helthy




all_h = df_hm.groupby(['id'], as_index=False).agg(
                     ['count', 'sum'])




all_h.head()




df_hm.head()




df_sl_metal.head()




df=df_sl_metal[['healthy_metal','irregular_metal']]
classes=df.idxmax(axis=1)
df['roof_material']=classes




df['id']=df_sl_metal['id']




df.drop(columns=['healthy_metal','irregular_metal'],inplace=True)




df.head()





df=df.replace('healthy_metal', 1)
df=df.replace('irregular_metal', 0)




all_m = df.groupby(['id'], as_index=False).agg(
                     ['count', 'sum'])




all_m.head()




all_m.columns = all_m.columns.droplevel(0)




all_models_irregualr_metal=all_m.loc[all_m['sum'] == 0]




all_models_irregualr_metal.head()




all_models_healthy_metal=all_m.loc[all_m['sum'] == 30]




len(all_models_irregualr_metal)




len(all_models_healthy_metal)




all_models_irregualr_metal.head()




all_models_irregualr_metal.to_csv(data_dir/'st_lucia_all_res_dense_models_irregular_metal.csv')
all_models_healthy_metal.to_csv(data_dir/'st_lucia_all_res_dense_models_healthy_metal.csv')

