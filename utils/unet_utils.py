import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import skimage
import skimage.io
import glob





class aerial_images:
    """
    Dataset class for MNIST
    """

    def __init__(self, root):
        """
        root -- path to data
        """
        
        paths = glob.glob(root + "/*")

        img_dat = glob.glob(paths[1] + "/*")
        lab_dat = glob.glob(paths[0] + "/*")

        self.data = np.array([img_dat, lab_dat]).T
        print(np.shape(self.data))
    
    def __len__(self):
        """
        Returns the lenght of the dataset (number of images)
        """
        # TODO: return the length (number of images) of the dataset
        return len(self.data)
        

    def __getitem__(self, index):
        """
        Loads and returns one image as floating point numpy array
        
        index -- image index in [0, self.__len__() - 1]
        """
        image = torch.tensor(skimage.io.imread(self.data[index][0])).permute(2,0,1).unsqueeze(0).float()
        label = torch.tensor(skimage.io.imread(self.data[index][1])).float()
    
        return (image, label)



class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
    

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, sizes):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2)

        for size in sizes:
            self.downs.append(DoubleConv(in_channels, size))
            in_channels = size
        
        for size in reversed(sizes):
            self.ups.append(nn.ConvTranspose2d(2*size, size, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(2*size, size))

        self.bottleneck = DoubleConv(sizes[-1], 2 * sizes[-1])
        self.final_conv = nn.Conv2d(in_channels = sizes[0], out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        skip_cons = []
        
        for down in self.downs:
            x = down(x)
            skip_cons.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_cons.reverse()
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            x = torch.cat((skip_cons[int(i//2)], x), dim = 1)
            x = self.ups[i+1](x)

        x = self.final_conv(x)
        x = nn.Sigmoid()(x)
        return x
    


def train(model, dataloader, optimizer, criterion, device):
    model.train()
    for batch_idx, (data, target) in enumerate(dataloader):
        #get data and feet through model
        data, target = data.to(device), target.to(device)
        
        while target.dim() != 4:
            target = torch.unsqueeze(target, dim = 0)
        
        optimizer.zero_grad()
        output = model(data)

        #one-hot encoding for pixel-wise BCE
        #target_oh = nn.functional.one_hot(target.to(torch.int64), num_classes = 24).transpose(1,4).squeeze(-1).float()
        #output_oh = nn.functional.one_hot(output.to(torch.int64), num_classes = 24).transpose(1,4).squeeze(-1).float()

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item() * data.size()