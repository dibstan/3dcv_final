import torch
import torch.nn as nn
from utils.Buffer import ImageBuffer
import tqdm 
import os
import numpy as np

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
    
def train(model, dataloader_training, dataLoader_validation , optimizer, criterion, device, buffer_size, buffer_update_freq,buffer_pick_size,n_epochs,patch_size,tag):
    """
    Train a unet model.

    parameters:
        model:                  Instance of the model class which has to be trained
        dataloader_training:    Dataloader used to extract training data from the file system
        dataLoader_validation:  Dataloader used to extract validation data from the file system
        optimizer               Optimizer used to update the model parameters.
        criterion:              Returns scalar value which is minimized with respect to the model parameters.
        device:                 Device on which the training runs.
        buffer_size:            Number of raw images in the buffer.
        buffer_update_freq:     Numbe rof trainning steps after which a new image is loaded to the buffer
        buffer_pick_size:       Number of images picked from the buffer in each training iteration
        multiplier:             Number of random variations of each immage picked from the buffer
        patch_size:             Size of the patches used to train the model
        n_epochs:               Number of epochs
        tag:                    Unique identifier to mark the training run

    returns:
        status:         True if the training finishes successfully

    IMPORTANT:

    Loading images from the storage is expensive (0.5s per image on laptop). Therefore, we proceed as follows:
    We have a buffer containing at most buffer_size images. After buffer_update_freq training iterations, a new image is taken from the file system
    and added to the buffer. If the buffer is already full, the oldest image is removed. In each training iteration, we randomly select buffer_pick_size
    images from the buffer. By applying randomly selected data augmentations we create multiplier patches of size patch_size x patch_size out of 
    each of the chosen raw images. This avoids loading a new batch in each training iteration but still ensures, that we have at least some variation in the 
    patches we use to train the model.
    """

    #Create folders to store the results of the training run
    os.makedirs(f"results/{tag}/state_dicts")
    os.makedirs(f"results/{tag}/code")
    os.makedirs(f"results/{tag}/data")
    os.makedirs(f"results/{tag}/images")

    #Create files to store the training loss and the validation loss
    with open(f"results/{tag}/data/training_loss.txt","w") as file:
        file.write("training-loss\n")
    file.close()

    with open(f"results/{tag}/data/validation_loss.txt","w") as file:
        file.write("validation-loss\n")
    file.close()

    #Store the loss
    storage_training_loss = torch.zeros(10).to(device)
    counter_training_loss = 0

    #For reproducability: Copy all the code files that have been used in the training
    """
    TO DO
    """

    #Set the model to training mode
    model.train()

    #initializ ethe buffer class
    buffer_images = ImageBuffer(buffer_size=buffer_size,buffer_pick_size=buffer_pick_size,C = dataloader_training.dataset.n_chanels,H = dataloader_training.dataset.height,W = dataloader_training.dataset.width)
    buffer_labels = ImageBuffer(buffer_size=buffer_size,buffer_pick_size=buffer_pick_size,C = 1,H = dataloader_training.dataset.height,W = dataloader_training.dataset.width)

    for epoch in tqdm.tqdm(range(1,n_epochs+1)):

        #Loop over all images in the training set
        for batch_idx, (X,Y) in enumerate(dataloader_training):

            #Update the buffers
            buffer_images.update(new_image = X[0])
            buffer_labels.update(new_image = Y[0][0])

            for j in range(buffer_update_freq):

                #sample from the buffers
                indices = np.random.permutation(min(buffer_images.size,buffer_images.buffer_pick_size))

                raw_images_tensor = buffer_images.sample(indices = indices)
                raw_labels_tensor = buffer_labels.sample(indices = indices)

                import matplotlib.pyplot as plt

                im = raw_images_tensor[0].transpose(0,2).transpose(0,1).numpy()
                print(im.shape)
                print(im)
                plt.imshow(im)
                plt.show()

                break

                """
                TO DO:
                
                1) Use raw images from the buffer to create a minibatch and apply augmentation
                   The image batch requires shape [batchsize,3,patch_size,patch_size], the label batch requires shape [batchsize,patch_size,patch_size]
                2) Sent the miibatch to the device
                3) replace the dummy implementation
                """

                batch_images = raw_images_tensor[:,:,:patch_size,:patch_size].to(device)            #Only Dummy implementation
                batch_labels = raw_labels_tensor[:,:patch_size,:patch_size].long().to(device)       #Only Dummy implementation

                #Get prediction from the model
                prdictions = model(batch_images)

                #Compute the loss
                loss = criterion(input = prdictions,target = batch_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                storage_training_loss[counter_training_loss] = loss.detach()
                if storage_training_loss.shape[0]-1 == counter_training_loss:
                    with open(f"results/{tag}/data/training_loss.txt","a+") as file:
                        np.savetxt(file,storage_training_loss.to("cpu").numpy())
                    file.close()
                    counter_training_loss = 0
                
                else: 
                    counter_training_loss += 1
            break

        model.eval()

        #Save the state dict
        torch.save(model.state_dict(), f"results/{tag}/state_dicts/state-dict_epoch-{epoch}.pt")

        #Get the validation loss of the model
        total_val_loss = validate(model = model, dataloader = dataLoader_validation, criterion = criterion, device = device,patch_size = patch_size)

        #Save the total validation loss
        with open(f"results/{tag}/data/validation_loss.txt","a+") as file:
            np.savetxt(file,np.array([total_val_loss]))
        file.close()
        
        model.train()

    status = True
    return status


def validate(model, dataloader, criterion, device,patch_size):
    """
    Compute the total loss of the model on the validation set

    parameters:
        model:       Model which is evaluated
        dataLoader:  Dataloader used to extract validation data from the file system
        criterion:   Criterion used to measure the discrepancy between the ground truth and the prediction
        device:      Device on which the evaluation runs
        patch_size:  Size of the image patches which are passed through the model


    returns:   
        total_loss:  Total loss of the model on the validation set
    
    """
    total_loss = 0
    with torch.no_grad():
        for i,(X,Y) in enumerate(dataloader):

            """
            TO DO:

            Select proper patch from the images.
            """

            batch_images = X[:,:,:patch_size,:patch_size].float().to(device)    #Only Dummy implementation
            batch_labels = Y[:,0,:patch_size,:patch_size].long().to(device)       #Only Dummy implementation

            predictions = model(batch_images)

            print(batch_labels.shape,predictions.shape)

            loss = criterion(input = predictions,target = batch_labels)

            total_loss += loss.item() * X.shape[0]

    return total_loss