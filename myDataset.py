#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
from torch.utils.data import Dataset
import os
import json
from PIL import Image
from torchvision import transforms


transform = transforms.Compose([transforms.RandomCrop(96), transforms.ToTensor()]) 
    
    
class dataSet(Dataset):
    
    def __init__(self, dataPath, dataType,transforms = transform):
        """
        :dataPath = json file
        :dataType = train or test
        
        """
        
        self.dataPath = dataPath
        self.dataType = dataType
        self.transforms = transform
        
        if self.dataType == 'train':
            with open(os.path.join(dataPath, 'train_images.json'), 'r') as f:
                self.imgs = json.load(f)
        else:
            with open(os.path.join(dataPath, 'test_images.json'), 'r') as f:
                self.imgs = json.load(f)

    #get img's lenth  
    def __len__(self):
        return len(self.imgs)
    
    #get item
    def __getitem__(self,index):
       
        img = Image.open(self.imgs[index], mode='r')
        img = img.convert('RGB')
        result = self.transforms(img)  #deal with original image
        cropResult = torch.nn.MaxPool2d(4)(result)
 
        return result, cropResult


# In[ ]:




