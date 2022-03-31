# +
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.utils.data import DataLoader
cudnn.benchmark = True
plt.ion()   # interactive mode
from utils.ImagesDataset import ImagesDataset
from tqdm import tqdm

# +
# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip()
    ]),
    'val': transforms.Compose([]),
}

dataset = ImagesDataset()
dataset.__len__()

# -

l=dataset.__len__()
val_split=0.3
train_set, val_set = torch.utils.data.random_split(dataset, [l-int(val_split*l), int(val_split*l)],generator=torch.Generator().manual_seed(42))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


bs=128
dataset_sizes = {'train':l-int(val_split*l),'val': int(val_split*l)}
train_loader = DataLoader(dataset=train_set,
                          batch_size=bs,
                          shuffle=True,
                          num_workers=4)
val_loader = DataLoader(dataset=val_set,
                          batch_size=bs,
                          shuffle=True,
                          num_workers=4)
dataloaders={'train':train_loader,'val':val_loader}


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
   
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            with tqdm(dataloaders[phase], unit="batch") as tepoch:
                batches=0
                for inputs, labels in tepoch:
                    tepoch.set_description(f"Epoch {epoch}")
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()
                    
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        #print('out',outputs)
                        #print('labels',labels)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    batches+=1                      
                    tepoch.set_postfix(loss=running_loss/(batches*bs), accuracy=100. * running_corrects.item()/(batches*bs))

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# +
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, 17)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=2, gamma=0.1)
# -

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=4)

torch.save(model_ft.state_dict(), './models/resnet18.pt')

# +
# #!gsutil cp -r ./models gs://e6691-bucket-models
# -

from torchsummary import summary
summary(model_ft, (3, 224, 224))



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
print('yumm!')
# -










