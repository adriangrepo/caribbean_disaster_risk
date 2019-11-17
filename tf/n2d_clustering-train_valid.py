#!/usr/bin/env python
# coding: utf-8

# ### image clustering
# 
# use venv N2D
# 
# https://arxiv.org/pdf/1908.05968.pdf
# 
# https://n2d.readthedocs.io/en/latest/quickstart.html#building-the-model
# 
# see also:
# 
# https://github.com/sudiptodip15/ClusterGAN
# 
# https://github.com/zhampel/clusterGAN

# In[1]:


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
 
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="0";  


# In[2]:


import os
import urllib3

import numpy as np
import pandas as pd
from keras.datasets import fashion_mnist
from keras.datasets import mnist
from pathlib import Path
from PIL import Image
from skimage import color
from skimage import io
import n2d


# In[3]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[4]:


x = np.concatenate((x_train, x_test))


# In[5]:


x.shape


# In[6]:


y = np.concatenate((y_train, y_test))


# In[7]:


x = x.reshape((x.shape[0], -1))
x = np.divide(x, 255.)


# In[8]:


x.shape


# ### fashion

# In[9]:


def load_fashion():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x = np.concatenate((x_train, x_test))
    print(x.shape)
    y = np.concatenate((y_train, y_test))
    print(y.shape)
    x = x.reshape((x.shape[0], -1))
    x = np.divide(x, 255.)
    y_names = {0: "T-shirt", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
               5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle Boot"}
    return x, y, y_names


# In[10]:


x, y, y_names=load_fashion()


# In[11]:


y[0]


# ### caribbean 
# 
# load as greyscale

# In[12]:


img_size=256


# In[13]:


data_dir = Path('../data')


# In[14]:


train_images=data_dir/f'train/rotated/clipped/{img_size}'
test_images=data_dir/f'test/rotated/clipped/{img_size}'


# In[15]:


def get_caribbean_images():
    images=[]
    names=[]
    for img in train_images.iterdir():
        if str(img).endswith('.tif'):
            name = str(img).split('/')[-1]
            name = name.split('.tif')[0]
            names.append(name)
            im = Image.open(img).convert('L')
            in_data = np.asarray(im, dtype=np.uint8)
            images.append(in_data)
    np_img=np.array(images)
    return np_img, names
    


# In[16]:


np_img, names=get_caribbean_images()


# In[17]:


np_img.shape


# In[18]:


np_img = np_img.reshape((np_img.shape[0], -1))


# ### clustering

# In[19]:


df_all=pd.read_csv(data_dir/'df_train_all.csv')


# In[20]:


df_valid=df_all.loc[df_all['verified'] == True]


# In[21]:


len(df_all)


# In[22]:


len(df_valid)


# In[23]:


df_valid=df_valid.drop_duplicates(subset=['id'])


# In[24]:


def get_image_cats(names):
    categories=[]
    for im_id in names:
        cat = df_valid.loc[df_valid['id'] == im_id, 'roof_material']
        categories.append(str(cat).split()[1])
    return categories


# In[25]:


categories=get_image_cats(names)


# In[26]:


n_cl_caribbean=list(set(categories))


# In[27]:


y = df_valid['roof_material'].values.tolist()


# <pre>
# f_x, f_y, f_names = data.load_fashion()
# 
# n_cl_fashion = 10
# 
# fashioncl = n2d.n2d(f_x, nclust = n_cl_fashion)
# fashioncl.preTrainEncoder(weights = "fashion-1000-ae_weights.h5")
# 
# manifold_cluster = n2d.UmapGMM(n_cl_fashion)
# fashioncl.predict(manifold_cluster)
# 
# fashioncl.visualize(f_y, f_names, dataset = "fashion", nclust = n_cl_fashion)
# print(fashioncl.assess(f_y))
# </pre>

# In[28]:


data_dir = Path('weights')
data_dir.mkdir(exist_ok=True)


# In[29]:


carribbean_cl = n2d.n2d(np_img, nclust = len(n_cl_caribbean))
carribbean_cl.preTrainEncoder()


# In[30]:


manifold_cluster = n2d.UmapGMM(len(n_cl_caribbean))
carribbean_cl.predict(manifold_cluster)


# In[33]:


print(carribbean_cl.assess(y))


# In[32]:


carribbean_cl.visualize(y, names=None, dataset = "caribbean", nclust = len(n_cl_caribbean))


# In[ ]:




