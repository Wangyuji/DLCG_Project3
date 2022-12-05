#!/usr/bin/env python
# coding: utf-8

# In[3]:


from PIL import Image
import os
import json


# In[12]:


def genDataList(t_sets, dataType, output):
    print("Loading datasets....")
    t_imgs = list()
    #train_imgs = list()
    #test_imgs = list()
    
    for i in t_sets:
        for j in os.listdir(i):
            path = os.path.join(i, j)
           # img = Image.open(path, mode='r')    
            t_imgs.append(path)
            
    if dataType == 'train':
        print("size of train set %d \n" % len(t_imgs))
  
        with open(os.path.join(output, 'train_images.json'), 'w') as j:  
            json.dump(t_imgs, j)
    
    if dataType == 'test':
        print("size of test set %d \n" % len(t_imgs))
  
        with open(os.path.join(output, 'test_images.json'), 'w') as j:  
            json.dump(t_imgs, j)

    
    print("Load complete")
    print("all the things save in output folder")


# In[13]:


#genDataList(train_sets=['./BSDS100', './Urban100'], test_sets=['./Set5', './Set14'], output = './output')


# In[ ]:




