'''
Code adapted from: https://towardsdatascience.com/multilayer-perceptron-for-image-classification-5c1f25738935
dataset: https://www.kaggle.com/competitions/dogs-vs-cats/data
'''

import numpy as np 
import pandas as pd 
import os
import copy
import time
import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
import copy
import time
import albumentations as A
import torch_optimizer as optim
from res_mlp_pytorch import ResMLP
from PIL import Image
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader


class FoodDataset(Dataset):
    def __init__(self, data_type=None, transforms=None):
        self.path = 'C:/test/demo/data/' + data_type + '/'
        self.images_name = os.listdir(self.path)
        self.transforms = transforms
        
    def __len__(self):
        return len(self.images_name)
    
    def __getitem__(self, idx):

        data = self.images_name[idx]
        
        label = data.split('.')[0]
        if label=='cat':
            label = 0
        elif label=='dog':
            label=1
       
        label = torch.tensor(label)
        
        image = cv2.imread(self.path + data)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transforms:
            aug = self.transforms(image=image)
            image = aug['image']
        
        return (image, label)


if __name__=='__main__':
    train_data = FoodDataset('train',
                            A.Compose([
                                A.RandomResizedCrop(256, 256),
                                A.HorizontalFlip(),
                                A.Normalize(),
                                ToTensorV2()
                            ]))
                            
    val_data = FoodDataset('val',
                          A.Compose([
                            A.Resize(384, 384),
                            A.CenterCrop(256, 256),
                            A.Normalize(),
                            ToTensorV2(),
                          ]))
                          
    test_data = FoodDataset('test',
                            A.Compose([
                            A.Resize(384, 384),
                            A.CenterCrop(256, 256),
                            A.Normalize(),
                            ToTensorV2(),
                          ]))

    dataloaders = {
        'train': DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4),
        'val': DataLoader(val_data, batch_size=32, shuffle=True, num_workers=4),
        'test': DataLoader(test_data, batch_size=32, shuffle=True, num_workers=4)
    }

    dataset_sizes = {
        'train': len(train_data),
        'val': len(val_data),
        'test': len(test_data)
    }

    # Train The Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Lamb(model.parameters(), lr=0.005, weight_decay=0.2)
    
    epochs=20
    since = 0.0
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 0.0
    best_acc = 0
    
    for ep in range(epochs):
        print(f"Epoch {ep}/{epochs-1}")
        print("-"*10)
        
        for phase in ['train']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_corrects = 0
                
            for images, labels in dataloaders[phase]:
                images = images.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    
                running_loss += loss.item() * images.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print(f"{phase} Loss:{epoch_loss:.4f} Acc:{epoch_acc:.4f}")
            
            if phase == 'val':
                if ep == 0:
                    best_loss = epoch_loss
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                else:
                    if epoch_loss < best_loss:
                        best_loss = epoch_loss
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(model.state_dict())
            
        print()
        
    time_elapsed = time.time() - since
    
    print(f'Training complete in {time_elapsed // 60}m {time_elapsed % 60}s')
    print(f'Best val loss: {best_loss:.4f}')
    print(f'Best acc: {best_acc}')
    
    model.load_state_dict(best_model_wts)
