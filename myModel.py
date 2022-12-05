#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import torch.nn as nn

class ResBlock(nn.Module):
    """Residual Bloack part"""
    def __init__(self,inputC,outputC):
        super(ResBlock,self).__init__()
        self.conv1 = nn.Conv2d(inputC,outputC,kernel_size=1,bias=False)
        self.bn1 = nn.BatchNorm2d(outputC)
        self.conv2 = nn.Conv2d(outputC,outputC,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(outputC)
        self.conv3 = nn.Conv2d(outputC,outputC,kernel_size=1,bias=False)
        self.relu = nn.PReLU()
        
    def forward(self,x):
        resudial = x 
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(x)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(x)
        
        out += resudial
        out = self.relu(out)
        return out

class SRResNet(nn.Module):
    def __init__(self, num_blocks = 16):
        
        super(SRResNet,self).__init__()
        
        #in channel=3,out_channels=64
        self.conv1 = nn.Conv2d(3,64,kernel_size=9,padding=4,padding_mode='reflect',stride=1)
        self.relu = nn.PReLU()
        #self.res_block = nn.Sequential(*[ResidualBlock(kernel_size=3, out_channels=64) for i in range(num_blocks)])
        resBlockLayer = 16
        self.res_block = self._makeLayer_(64,64,resBlockLayer)
        
        self.conv2 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.PReLU()
        
        #subpixel convolution
        self.subConv1 = nn.Conv2d(64,256,kernel_size=3,stride=1,padding=2,padding_mode='reflect')
        self.shuffle1 = nn.PixelShuffle(2)
        self. reluSub1 = nn.PReLU()
        
        self.subConv2 = nn.Conv2d(64,256,kernel_size=3,stride=1,padding=1,padding_mode='reflect')
        self.shuffle2 = nn.PixelShuffle(2)
        self. reluSub2 = nn.PReLU()
        
        
        
        #in channel=64,out_channels=3
        self.conv3 = nn.Conv2d(64,3,kernel_size=9,stride=1)
        
    def _makeLayer_(self,inputC,outputC,blocks):
        layers = []
        layers.append(ResBlock(inputC,outputC))
        
        for i in range(1,blocks):
            layers.append(ResBlock(outputC,outputC))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.relu(output)
        residual = output
        
        output - self.res_block(output)
        output = self.conv2(output)
        output = self.bn2(output)
        output += residual
        
        output = self.subConv1(output)   
        output = self.shuffle1(output)
        output = self.reluSub1(output)
        
        output = self.subConv2(output)   
        output = self.shuffle2(output)
        output = self.reluSub2(output)
        output = self.conv3(output)
        
        return output


# In[4]:





# In[ ]:




