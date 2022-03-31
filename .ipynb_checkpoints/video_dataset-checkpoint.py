import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os
from PIL import Image


class ImagesDataset(Dataset):
    
    def __init__(self,frames2label):
        # Initialize data, download, etc.
        # read with numpy or pandas
        
        #read all images paths,label dic
        self.root='./e6691-bucket-images/'
        self.frames2label=frames2label
        self.n_samples = len(self.frames2label)# a remplir
        self.index2data=[]
        for frame in self.frames2label:
            self.index2data.append(frame)
        '''
        xy = np.loadtxt('./data/wine/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        
        self.n_samples = xy.shape[0]
        # here the first column is the class label, the rest are the features
        self.x_data = torch.from_numpy(xy[:, 1:]) # size [n_samples, n_features]
        self.y_data = torch.from_numpy(xy[:, [0]]) # size [n_samples, 1]
        '''

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return read_image(self.index2data[index]), self.frames2label[self.index2data[index]]
    def read_image(self,file):
        path=os.path.join(self.root,file)
        img = Image.open(path)
        convert_tensor = transforms.Compose([transforms.PILToTensor()])
        return torch.Tensor.float(convert_tensor(img))
    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples


# create dataset
with open('./pickle/frames2label.p', 'rb') as fp:
    frames2label = pickle.load(fp)
dataset = ImagesDataset(frames2label)

# +
#Test code

# +
import matplotlib.pyplot as plt
# get first sample and unpack
first_data = dataset[0]
features, labels = first_data
print( labels)
plt.imshow( features.permute(1, 2, 0) /255 )
# Load whole dataset with DataLoader
# shuffle: shuffle data, good for training
# num_workers: faster loading with multiple subprocesses
# !!! IF YOU GET AN ERROR DURING LOADING, SET num_workers TO 0 !!!
train_loader = DataLoader(dataset=dataset,
                          batch_size=4,
                          shuffle=True,
                          num_workers=2)

# convert to an iterator and look at one random sample
dataiter = iter(train_loader)
data = dataiter.next()
features, labels = data
print(labels)
print(features.shape)
plt.imshow(features[0].permute(1, 2, 0) /255 )
