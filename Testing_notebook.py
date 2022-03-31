import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from ImagesDataset import ImagesDataset

# create dataset
dataset = ImagesDataset()

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
