import torch
import torch.nn as nn
from utils.Buffer import ImageBuffer
import numpy as np
import utils.image_utils as iu
import tqdm 
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import skimage
from utils.image_utils import list_scaled_images

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
    
def train(model, dataloader_training, dataLoader_validation , optimizer, criterion, device, buffer_size, buffer_update_freq,
          buffer_pick_size, n_epochs, patch_size, batch_size, tag, rotation, mirroring, scaling_factor, use_original,threshold):
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
        rotation:               Use rotation for image augmentation
        mirroring:              Use mirroring for image augmentation
        scaling_factor:         How much the image should be downscaled in each scaling
        use_original:           Use the original image for training or start with the first downscaled version
        threshold:              number of different classes per patch which is required to keep the patch in the training set

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

    with open(f"results/{tag}/data/classes_per_patch.txt","w") as file:
        file.write("classes per patch\n")
    file.close()

    with open(f"results/{tag}/data/validation_loss.txt","w") as file:
        file.write("validation-loss\n")
    file.close()

    with open(f"results/{tag}/data/F1_score.txt","w") as file:
        file.write("F1-score\n")
    file.close()

    with open(f"results/{tag}/data/accuracy.txt","w") as file:
        file.write("accuracy\n")
    file.close()

    #Store the loss
    storage_training_loss = torch.zeros(10).to(device)
    counter_training_loss = 0

    #Store how many different classes are contained in each patch
    storage_classes_per_pixel = torch.zeros(25)

    #For reproducability: Copy all the code files that have been used in the training
    """
    TO DO
    """

    #Set the model to training mode
    model.train()

    #initialize the buffer class
    buffer_images = ImageBuffer(buffer_size=buffer_size,buffer_pick_size=buffer_pick_size,C = dataloader_training.dataset.n_chanels,H = dataloader_training.dataset.height,W = dataloader_training.dataset.width)
    buffer_labels = ImageBuffer(buffer_size=buffer_size,buffer_pick_size=buffer_pick_size,C = 1,H = dataloader_training.dataset.height,W = dataloader_training.dataset.width)



    #for epoch in tqdm.tqdm(range(1,n_epochs+1)):
    for epoch in range(1,n_epochs+1):
        print('Epoch: {} \n'.format(epoch))

        #Loop over all images in the training set
        for batch_idx, (X,Y) in enumerate(tqdm.tqdm(dataloader_training)):
            #print(torch.cuda.memory_allocated(0))
            #Update the buffers
            buffer_images.update(new_image = X[0])
            buffer_labels.update(new_image = Y[0][0])

            for j in range(buffer_update_freq):

                #sample from the buffers
                indices = np.random.permutation(min(buffer_images.size,buffer_size))[:buffer_pick_size]

                raw_images_tensor = buffer_images.sample(indices = indices)
                raw_labels_tensor = buffer_labels.sample(indices = indices)

                batch_images = torch.Tensor()
                batch_labels = torch.Tensor()

                #Create patches from each of the raw images samplesd from the buffer
                for i, image in enumerate(raw_images_tensor):
                    batch_image = iu.prepare_image_torch(image.permute(1,2,0), patch_size, rotation = rotation, mirroring = mirroring, n=scaling_factor, use_original=use_original)
                    batch_label = iu.prepare_image_torch(raw_labels_tensor[i], patch_size, rotation = rotation, mirroring = mirroring, n=scaling_factor, use_original=use_original)

                    batch_labels = torch.cat([batch_labels, batch_label])
                    batch_images = torch.cat([batch_images, batch_image])

                n_befor_quility_insp = batch_image.shape[0]

                a = batch_labels.reshape(batch_labels.shape[0],-1)

                counts = torch.zeros(a.shape[0])

                for ri in range(a.shape[0]):
                    counts[ri] = len(torch.unique(a[ri]))
                
                mask = (counts >= threshold)

                #If no patch meets the criterion, use the image with the highest rate of different class labels
                if mask.sum() == 0:
                    i_max = torch.argmax(counts)

                    mask[i_max] = True

                batch_images = batch_images[mask]
                batch_labels = batch_labels[mask]

                #Save the recorded counts
                uique_counts,counts_counts = torch.unique(counts,return_counts = True)

                storage_classes_per_pixel[uique_counts.numpy()] += counts_counts

                continue


            
                if (epoch == 1 and batch_idx == 0):
                    print("#########################################################################################")
                    print(f"\tNumber of images created from each image:\t{batch_images.shape[0]}")
                    print(f"\tShape of image batch:\t{batch_images.shape}")
                    print("#########################################################################################")
                    
                #if batch_idx == 1:
                #    fig, ax = plt.subplots(2)
                #    ax[0].imshow(batch_images[0].permute(1,2,0).cpu().detach().numpy())
                #    ax[1].imshow(batch_labels[0].cpu().detach().numpy())
                
                ind = torch.randperm(batch_images.size()[0])
                batch_images = batch_images[ind]
                batch_labels = batch_labels[ind]

                #Iterate over all patches in teh current selection in portions of batch_size images
                for i in range(0, batch_images.size()[0]-1, batch_size):
                    
                    image = batch_images[i:i+batch_size].to(device)
                    label = batch_labels[i:i+batch_size].long().to(device)

                    #Get prediction from the model
                    prdictions = model(image)

                    loss = criterion(inputs = prdictions,targets = label)

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

            #Validation
            if (batch_idx % 50 == 0 and batch_idx!=0):
                model.eval()

                #Save the state dict
                #torch.save(model.state_dict(), f"results/{tag}/state_dicts/state-dict_epoch-{epoch}.pt")

                #Get the validation loss of the model
                total_val_loss, f1_score, acc_score = validate(model = model, dataloader = dataLoader_validation, criterion = criterion, device = device,patch_size = patch_size, batch_size = batch_size,
                                          tag = tag,epoch = epoch, scaling_factor= scaling_factor, use_original = use_original, batch_idx = batch_idx)

                #Save the performance measures
                with open(f"results/{tag}/data/validation_loss.txt","a+") as file:
                    np.savetxt(file,np.array([total_val_loss]))
                file.close()

                with open(f"results/{tag}/data/F1_score.txt","a+") as file:
                    np.savetxt(file,np.array([f1_score]))
                file.close()

                with open(f"results/{tag}/data/accuracy.txt","a+") as file:
                    np.savetxt(file,np.array([acc_score]))
                file.close()                

                model.train()

    #Store the statistics about the number of samples per class
    with open(f"results/{tag}/data/classes_per_patch.txt","a+") as file:
        np.savetxt(file,storage_classes_per_pixel.to("cpu").numpy())
    file.close()

    #Plot the distribution of the number of classes per patch
    fig = plt.figure(figsize = (30,8))
    ax = plt.axes()
    ax.plot(np.arange(len(storage_classes_per_pixel)),storage_classes_per_pixel.numpy() / storage_classes_per_pixel.sum())
    ax.set_xlabel("classes per patch")
    ax.set_xlabel("rel. count")
    plt.savefig(f"results/{tag}/images/class_counts.jpg")
    plt.close()

    status = True
    return status


def visualizer(ground_truth,prediction,image,image_folder,fs = 30):
    """
    parameters:
        ground_truth:       Tensoor of shape (H,W) containing the true labels
        prediction:         Tensor of shape (C,H,W) containing the predicted logits
        image:              Raw image
        image_folder:       path to folder where teh images a<re stored
        fs:                 Fontsize for labeling
    """
    
    #Get the probabilities of the individual classes
    class_probs_per_pixel = torch.nn.functional.softmax(prediction, dim=0)

    #Get the Maximum a posterioiri labels
    pred_hard_decision = torch.argmax(class_probs_per_pixel,dim = 0)

    #Plot the hard decision
    fig,axs = plt.subplots(1,3,figsize = (30,8))
    axs[0].imshow(image.permute(1,2,0).detach().numpy())
    axs[0].axis("off")
    axs[0].set_title("Image",fontsize = fs)

    axs[1].imshow(ground_truth.detach().numpy())
    axs[1].axis("off")
    axs[1].set_title("Ground truth class labels",fontsize = fs)

    axs[2].imshow(pred_hard_decision.detach().numpy())
    axs[2].axis("off")
    axs[2].set_title("MAP labels",fontsize = fs)

    plt.savefig(image_folder + "hard-labels.jpg")
    plt.close()

    #Plot the confusion matrix for the hard decisions
    class_names = ["unlabeled","paved-area","dirt","grass","gravel","water","rocks","pool","vegetation","roof","wall","window","door","fence","fence-pole","person","dog","car","bicycle","tree","bald-tree","ar-marker","obstacle","conflicting"]
    cm = metrics.confusion_matrix(y_true = ground_truth.flatten(), y_pred =pred_hard_decision.flatten(),labels = np.arange(len(class_names)))
    
    fig = plt.figure(figsize = (15,15))
    im = plt.imshow(cm)
    plt.xticks(ticks = np.arange(len(class_names)),labels = class_names,rotation=90,fontsize = fs)
    plt.yticks(ticks = np.arange(len(class_names)),labels = class_names,fontsize = fs)
    plt.xlabel("Predicted class",fontsize = fs)
    plt.ylabel("True class",fontsize = fs)
    plt.tight_layout()

    plt.savefig(image_folder + "confusion-matrix.jpg")
    plt.close()

def validate(model, dataloader, criterion, device, patch_size, batch_size, tag, epoch, scaling_factor, use_original, batch_idx):
    """
    Compute the total loss of the model on the validation set

    parameters:
        model:          Model which is evaluated
        dataLoader:     Dataloader used to extract validation data from the file system
        criterion:      Criterion used to measure the discrepancy between the ground truth and the prediction
        device:         Device on which the evaluation runs
        patch_size:     Size of the image patches which are passed through the model
        scaling_factor: How much the image should be downscaled in each scaling
        use_original:   Use the original image for training or start with the first downscaled version
        batch_idx:      Index of the current batch


    returns:   
        total_loss:  Total loss of the model on the validation set
    
    """
    total_loss = 0
    f1_score = 0
    acc_score = 0
    counter = 0

    with torch.no_grad():
        for i,(X,Y) in enumerate(dataloader):

            """
            TO DO:

            Select proper patch from the images.
            """
            batch_images = iu.prepare_image_torch(X[0].permute(1,2,0), patch_size, rotation = False, mirroring = False, n=scaling_factor, use_original=use_original).float().to(device)
            batch_labels = iu.prepare_image_torch(Y[0][0], patch_size, rotation = False, mirroring = False, n=scaling_factor, use_original=use_original).long().to(device)
            #batch_images = X[:,:,:patch_size,:patch_size].float().to(device)    #Only Dummy implementation
            #batch_labels = Y[:,0,:patch_size,:patch_size].long().to(device)       #Only Dummy implementation

            for j in range(0, batch_images.size()[0]-1, batch_size):

                image = batch_images[j:j+batch_size].to(device)
                label = batch_labels[j:j+batch_size].long().to(device)
                prediction = model(image)

                loss = criterion(inputs = prediction,targets = label)
                total_loss += loss.item() * X.shape[0]

                #Quantities to calculate F1 and accuracy
                y_true = label.cpu().numpy().flatten()
                y_pred = torch.argmax(prediction, dim=1).cpu().numpy().flatten()

                #Calculate this batches F1 score
                f1 = metrics.f1_score(y_true = y_true, y_pred = y_pred, average = 'weighted')
                f1_score += f1

                #Calculate this batches accuracy
                acc = metrics.accuracy_score(y_true = y_true, y_pred = y_pred,)
                acc_score += acc

                counter += 1

            #visualization
            if i%10 == 0:
                path = f"results/{tag}/images/Visualization_epoch-{epoch}_batch-{i+1}_idx{batch_idx}/"
                os.makedirs(path)
                visualizer(ground_truth = label[0].detach().cpu(),prediction = prediction[0].detach().cpu(),image = image[0].detach().cpu(),image_folder = path,fs = 30)
    f1_score = f1_score/(counter)
    acc_score = acc_score/(counter)    

    return total_loss, f1_score, acc_score