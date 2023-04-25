#Load dependencies and packages
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
from utils.image_utils import list_scaled_images
import json

##############################################################################################################################################
#Unet Model
##############################################################################################################################################

def initialize_weights(m):
  """
  Initialize the weights of the UNet.

  parameters:
    m:  Model instance

  returns:
    None
  """
  if isinstance(m, nn.Conv2d):
      nn.init.xavier_normal_(m.weight.data)
  elif isinstance(m, nn.ConvTranspose2d):
      nn.init.xavier_normal_(m.weight.data)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        parameters:
            in_channels:    Number of channels for the input image
            out_channels:   Number of channels for the output image
        """
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
        """
        parameters:
            in_channels:    Number of channels for the input image
            out_channels:   Number of channels for the output image
            sizes:          
        """
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

##############################################################################################################################################
#Training of a model
##############################################################################################################################################

def train(model, dataloader_training, dataLoader_validation , optimizer, criterion, device, buffer_size, buffer_update_freq,
          buffer_pick_size, n_epochs, patch_size, batch_size, tag, rotation, mirroring, scaling_factor, use_original,threshold,config):
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
        batch_size:             Batch size
        n_epochs:               Number of epochs
        tag:                    Unique identifier to mark the training run
        rotation:               Use rotation for image augmentation
        mirroring:              Use mirroring for image augmentation
        scaling_factor:         How much the image should be downscaled in each scaling
        use_original:           Use the original image for training or start with the first downscaled version
        threshold:              number of different classes per patch which is required to keep the patch in the training set
        config:                 Dict containig all the hyperparameters used to train the model

    returns:
        status:                 True if the training finishes successfully
    """

    #Create folders to store the results of the training run
    os.makedirs(f"results/{tag}/state_dicts")
    os.makedirs(f"results/{tag}/data")
    os.makedirs(f"results/{tag}/images")

    #Save the configurations of the training
    with open(f"results/{tag}/data/config.json","w") as file:
        json.dump(config,file)
    file.close()

    #Create files to store the training loss and the validation loss

    #Training objective
    with open(f"results/{tag}/data/training_loss.txt","w") as file:
        file.write("training-loss\n")
    file.close()

    #Number of different classes in the image patches
    with open(f"results/{tag}/data/classes_per_patch.txt","w") as file:
        file.write("classes per patch\n")
    file.close()

    #Objective value on the validation set
    with open(f"results/{tag}/data/validation_loss.txt","w") as file:
        file.write("validation-loss\nTotal Train iter\tepoch\titer in epoch\tvalidation loss\n")
    file.close()

    #F1 Score on the validation set
    with open(f"results/{tag}/data/F1_score.txt","w") as file:
        file.write("F1-score\nTotal Train iter\tepoch\titer in epoch\tvalidation F1-score\n")
    file.close()

    #Accuracy on the validation set
    with open(f"results/{tag}/data/accuracy.txt","w") as file:
        file.write("accuracy\nTotal Train iter\tepoch\titer in epoch\tvalidation accuracy\n")
    file.close()

    #Storage for the objective values
    storage_training_loss = torch.zeros([config["DropLossFreq"],2]).to(device)
    counter_training_loss = 0

    #Store how many different classes are contained in each patch
    storage_classes_per_pixel = torch.zeros(config["DropLossFreq"])

    #Set the model to training mode
    model.train()

    #initialize the buffer class
    buffer_images = ImageBuffer(buffer_size=buffer_size,buffer_pick_size=buffer_pick_size,C = dataloader_training.dataset.n_chanels,H = dataloader_training.dataset.height,W = dataloader_training.dataset.width)
    buffer_labels = ImageBuffer(buffer_size=buffer_size,buffer_pick_size=buffer_pick_size,C = 1,H = dataloader_training.dataset.height,W = dataloader_training.dataset.width)

    #Count the total number of training iterations
    nTrainIterTotal = 0

    for epoch in range(1,n_epochs+1):
        print('Epoch: {} \n'.format(epoch))

        #Count the actual number of training iterations within an epoch
        withinBatchCounter = 0

        #Loop over all images in the training set
        for batch_idx, (X,Y) in enumerate(tqdm.tqdm(dataloader_training)):

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

                #Create patches from each of the raw images sampled from the buffer
                for i, image in enumerate(raw_images_tensor):
                    batch_image = iu.prepare_image_torch(image.permute(1,2,0), patch_size, rotation = rotation, mirroring = mirroring, n=scaling_factor, use_original=use_original)
                    batch_label = iu.prepare_image_torch(raw_labels_tensor[i], patch_size, rotation = rotation, mirroring = mirroring, n=scaling_factor, use_original=use_original)

                    batch_labels = torch.cat([batch_labels, batch_label])
                    batch_images = torch.cat([batch_images, batch_image])

                #Eliminate patches with a too small number of different classes
                a = batch_labels.reshape(batch_labels.shape[0],-1)
                counts = torch.zeros(a.shape[0])

                for ri in range(a.shape[0]):
                    counts[ri] = len(torch.unique(a[ri]))
                
                mask = (counts >= threshold)

                #If no patch meets the criterion, use the image with the highest rate of different class labels
                if mask.sum() == 0:
                    continue
      
                batch_images = batch_images[mask]
                batch_labels = batch_labels[mask]

                #Save the recorded counts
                uique_counts,counts_counts = torch.unique(counts,return_counts = True)
                storage_classes_per_pixel[uique_counts.numpy()] += counts_counts
                
                #Suhuffle the batch
                ind = torch.randperm(batch_images.size()[0])
                batch_images = batch_images[ind]
                batch_labels = batch_labels[ind]

                #Iterate over all patches in teh current selection in portions of batch_size images
                for i in range(0, batch_images.size()[0], batch_size):
                    
                    image = batch_images[i:i+batch_size].to(device)
                    label = batch_labels[i:i+batch_size].long().to(device)

                    #Get prediction from the model
                    prdictions = model(image)

                    #Compute the objective function
                    loss = criterion(inputs = prdictions,targets = label)

                    #Optimize the loss with respect to the model parameters
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    #Another update step was performed
                    withinBatchCounter += 1
                    nTrainIterTotal += 1

                    #Store the loss
                    storage_training_loss[counter_training_loss][0] = nTrainIterTotal
                    storage_training_loss[counter_training_loss][1] = loss.detach()

                    #Drop the training loss if the storage is filled
                    if storage_training_loss.shape[0]-1 == counter_training_loss:
                        with open(f"results/{tag}/data/training_loss.txt","a+") as file:
                            np.savetxt(file,storage_training_loss.to("cpu").numpy())
                        file.close()
                        counter_training_loss = 0
                    else: 
                        counter_training_loss += 1

            #Validation
            if (batch_idx % config["BatchFreqValidate"] == 0 and batch_idx!=0):

                #Set the model to evaluation mode
                model.eval()
                evaluator(config = config,epoch = epoch,model = model,tag = tag,dataloader = dataLoader_validation,withinBatchCounter = withinBatchCounter,criterion = criterion,device = device,nTrainIterTotal = nTrainIterTotal)
                model.train()

        #Perform a full evaluation at the end of each epoch and store the current state of the model
        model.eval()
        torch.save(model.state_dict(),f"results/{tag}/state_dicts/state_dict_epoch-{epoch}.pt")
        evaluator(config = config,epoch = epoch,model = model,tag = tag,dataloader = dataLoader_validation,withinBatchCounter = -1,criterion = criterion,device = device,nTrainIterTotal = nTrainIterTotal)
        model.train()

        #Reset the counter for the training iterations before starting with the new epoch
        withinBatchCounter = 0

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

##############################################################################################################################################
#Evaluation of the model during the training
##############################################################################################################################################

def evaluator(config,epoch,model,tag,dataloader,withinBatchCounter,criterion,device,nTrainIterTotal):
        """
        Parameters:
            config:                 Dictionary containing the hyperparameters of the training
            epoch:                  Current training epoch
            model:                  Model which is evaluated
            tag:                    Tag to identify the training run
            dataloader:             Dataloader containing the validation samples
            withinBatchCounter:     Number of update steps in the current epoch
            criterion:              Function to compute the loss which is optimized during the training
            device:                 Device on which the training runs 
            nTrainIterTotal:        Total number of training iterations
        
        returns: 
            None
        """

        #Evaluate the performance of the model on the validation set
        total_val_loss, f1_score, acc_score = validate(
            model = model, 
            dataloader = dataloader, 
            criterion = criterion, 
            device = device,
            patch_size = config["patchSize"], 
            batch_size = config["batchSize"],
            tag = tag,
            epoch = epoch, 
            scaling_factor= config["scalingFactor"], 
            use_original = config["useOriginal"], 
            withinBatchCounter = withinBatchCounter)

        #Save the performance measures
        with open((f"results/{tag}/data/validation_loss.txt"),"a+") as file:
            np.savetxt(file,np.array([[nTrainIterTotal,epoch,withinBatchCounter,total_val_loss]]))
        file.close()

        with open(f"results/{tag}/data/F1_score.txt","a+") as file:
            np.savetxt(file,np.array([[nTrainIterTotal,epoch,withinBatchCounter,f1_score]]))
        file.close()

        with open(f"results/{tag}/data/accuracy.txt","a+") as file:
            np.savetxt(file,np.array([[nTrainIterTotal,epoch,withinBatchCounter,acc_score]]))
        file.close()

def visualizer(ground_truth,prediction,image,image_folder,fs = 30):
    """
    parameters:
        ground_truth:       Tensor of shape (H,W) containing the true labels
        prediction:         Tensor of shape (C,H,W) containing the predicted logits
        image:              Raw image
        image_folder:       Path to folder where the images are stored
        fs:                 Fontsize for labeling

    returns:
        None
    """
    
    #Get the probabilities of the individual classes
    class_probs_per_pixel = torch.nn.functional.softmax(prediction, dim=0)

    #Get the Maximum a posterioiri labels
    pred_hard_decision = torch.argmax(class_probs_per_pixel,dim = 0)

    #Plot the image
    fig,axs = plt.subplots(1,3,figsize = (30,8))
    axs[0].imshow(image.permute(1,2,0).detach().numpy(),vmin = 1,vmax = 25)
    axs[0].axis("off")
    axs[0].set_title("Image",fontsize = fs)

    #Plot the Ground truth labeling
    axs[1].imshow(ground_truth.detach().numpy(),vmin = 1,vmax = 25)
    axs[1].axis("off")
    axs[1].set_title("Ground truth class labels",fontsize = fs)

    #Plot the MAP prediction of the model
    axs[2].imshow(pred_hard_decision.detach().numpy(),vmin = 1,vmax = 25)
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

def validate(model, dataloader, criterion, device, patch_size, batch_size, tag, epoch, scaling_factor, use_original, withinBatchCounter):
    """
    Compute the total loss of the model on the validation set

    parameters:
        model:                  Model which is evaluated
        dataLoader:             Dataloader used to extract validation data from the file system
        criterion:              Criterion used to measure the discrepancy between the ground truth and the prediction
        device:                 Device on which the evaluation runs
        patch_size:             Size of the image patches which are passed through the model
        batch_size:             Batch size 
        tag:                    Tag to identify the training run
        epoch:                  Current training epoch
        scaling_factor:         How much the image should be downscaled in each scaling
        use_original:           Use the original image for training or start with the first downscaled version
        withinBatchCounter:     Number of iterations in the current epoch

    returns:   
        total_loss:             Total loss of the model on the validation set
        f1_score:               F1 score on the validation set    
        acc_score               Accuracy on the validation set
    
    """

    #Initialize the performance measures
    total_loss = 0
    f1_score = 0
    acc_score = 0
    counter = 0

    with torch.no_grad():
        for i,(X,Y) in enumerate(dataloader):

            #Select patches from the images
            batch_images = iu.prepare_image_torch(X[0].permute(1,2,0), patch_size, rotation = False, mirroring = False, n=scaling_factor, use_original=use_original).float().to(device)
            batch_labels = iu.prepare_image_torch(Y[0][0], patch_size, rotation = False, mirroring = False, n=scaling_factor, use_original=use_original).long().to(device)
            
            #Evaluate the images in the current batch in small slices
            for j in range(0, batch_images.size()[0], batch_size):
                
                #Select the sub-batch
                image = batch_images[j:j+batch_size].to(device)
                label = batch_labels[j:j+batch_size].long().to(device)

                #Get the prediction
                prediction = model(image)

                #Get the MAP prediction from the model output
                Y_pred = torch.argmax(prediction, dim=1).cpu().numpy()

                #Compute objective value
                loss = criterion(inputs = prediction,targets = label)
                total_loss += loss.item()

                #Quantities to calculate F1 and accuracy
                y_true = label.cpu().numpy().flatten()
                y_pred = Y_pred.flatten()

                #Calculate this batches F1 score
                f1 = metrics.f1_score(y_true = y_true, y_pred = y_pred, average = 'weighted')
                f1_score += f1

                #Calculate this batches accuracy
                acc = metrics.accuracy_score(y_true = y_true, y_pred = y_pred,)
                acc_score += acc

                #Increase the counter which is used for normalization
                counter += 1    

            #Visualize the results
            if i%10 == 0:
                path = f"results/{tag}/images/Visualization_epoch-{epoch}_iter-{withinBatchCounter}/"
                os.makedirs(path)
                visualizer(ground_truth = label[0].detach().cpu(),prediction = prediction[0].detach().cpu(),image = image[0].detach().cpu(),image_folder = path,fs = 30)

    #Normalize the properties
    total_loss = total_loss / counter
    f1_score = f1_score/counter
    acc_score = acc_score/counter  

    return total_loss, f1_score, acc_score