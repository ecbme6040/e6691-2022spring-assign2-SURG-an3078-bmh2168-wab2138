import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os
from PIL import Image


class ImagesDataset(Dataset):
    
    def __init__(self):
        # Initialize data, download, etc.
        # read with numpy or pandas
        
        #read all images paths,label dic
        with open('./pickle/frames2label.p', 'rb') as fp:
            frames2label = pickle.load(fp)
            
            
        self.ROOT='./e6691-bucket-images/'
        self.frames2label=frames2label
        self.n_samples = len(self.frames2label)
        self.index2data=[]
        self.convert_tensor = transforms.Compose([transforms.PILToTensor()])
        
        
        for frame in self.frames2label:
            self.index2data.append(frame)
       

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        path=os.path.join(self.ROOT,self.index2data[index])
        return torch.Tensor.float(self.convert_tensor(Image.open(path))), torch.nn.functional.one_hot(self.frames2label[self.index2data[index]], num_classes=17)
   
    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples