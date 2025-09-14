import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from torchvision import models
import time
import torch
import copy

class BasicModule(torch.nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.module_name = str(type(self))

    def load(self, path, use_gpu=False):
        if not use_gpu:
            self.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
        else:
            self.load_state_dict(torch.load(path))

    def save(self, name=None):
        if name is None:
            prefix = self.module_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), 'checkpoint/' + name)
        return name

    def forward(self, *input):
        pass

class ImgModule(BasicModule):
    def __init__(self, y_dim, bit, norm=True, mid_num1=1024*8, mid_num2=1024*8, hiden_layer=3):
        super(ImgModule, self).__init__()
        self.module_name = "image_model"
        mid_num1 = mid_num1 if hiden_layer > 1 else bit
        modules = [nn.Linear(y_dim, mid_num1)]
        if hiden_layer >= 2:
            modules += [nn.ReLU(inplace=True), nn.Dropout(0.2)]
            pre_num = mid_num1
            for i in range(hiden_layer - 2):
                if i == 0:
                    modules += [nn.Linear(mid_num1, mid_num2), nn.ReLU(inplace=True), nn.Dropout(0.2)]
                else:
                    modules += [nn.Linear(mid_num2, mid_num2), nn.ReLU(inplace=True), nn.Dropout(0.2)]
                pre_num = mid_num2
                
            modules += [nn.Linear(pre_num, bit)]
        self.fc1 = nn.Sequential(*modules)
        self.fc2 = copy.deepcopy(self.fc1)
        #self.apply(weights_init)
        self.norm = norm
        
    def forward(self, x, train = False):
        out1 = self.fc1(x).tanh()
        out2 = self.fc2(x).tanh()
        if self.norm: 
            out1 = out1 / torch.norm(out1, dim=1, keepdim=True)
            out2 = out2 / torch.norm(out2, dim=1, keepdim=True)
        if train:
            return out1, out2
        else:  
            return (out1 + out2)/2

class TxtModule(BasicModule):
    def __init__(self, y_dim, bit, norm=True, mid_num1=1024*8, mid_num2=1024*8, hiden_layer=2):
        super(TxtModule, self).__init__()
        self.module_name = "text_model"
        mid_num1 = mid_num1 if hiden_layer > 1 else bit
        modules = [nn.Linear(y_dim, mid_num1)]
        if hiden_layer >= 2:
            modules += [nn.ReLU(inplace=True), nn.Dropout(0.2)]
            pre_num = mid_num1
            for i in range(hiden_layer - 2):
                if i == 0:
                    modules += [nn.Linear(mid_num1, mid_num2), nn.ReLU(inplace=True), nn.Dropout(0.2)]
                else:
                    modules += [nn.Linear(mid_num2, mid_num2), nn.ReLU(inplace=True), nn.Dropout(0.2)]
                pre_num = mid_num2
            modules += [nn.Linear(pre_num, bit)]  
        
        self.fc1 = nn.Sequential(*modules)
        self.fc2 = copy.deepcopy(self.fc1)
        self.norm = norm

    def forward(self, x, train = False):
        out1 = self.fc1(x).tanh()
        out2 = self.fc2(x).tanh()
        if self.norm: 
            out1 = out1 / torch.norm(out1, dim=1, keepdim=True)
            out2 = out2 / torch.norm(out2, dim=1, keepdim=True)
        if train:
            return out1, out2
        else:
            return (out1 + out2)/2
    
    