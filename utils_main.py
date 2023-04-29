import utils.ObjectiveFunctions as of
import utils.unet_utils as ut
import torch
import matplotlib.pyplot as plt
import numpy as np
from utils.image_utils import prepare_image_torch
from sklearn import metrics

######################################################################################################################
#Dictionaries to select objective function, optimizer andthe model family
######################################################################################################################

#Possible loss functions
loss_dict = {
    "dice": of.DiceLoss,
    "combo":of.CriterionCombo,
    "focal": of.FocalLoss,
}

#Possible model architectures
model_dict = {
    "uNet":ut.UNet
}

#Possible optimizers
optimizer_dict = {
    "Adam":torch.optim.Adam,
    "SGD":torch.optim.SGD
}
######################################################################################################################
#Model for naive classifer based on the prioir distribution
######################################################################################################################
class pseudoClf():
    def __init__(self,prior,patchSize):
        
        self.prior = prior
        #Get the clf of the prior distribution
        self.cds = np.cumsum(prior)
        self.patchSize = patchSize

    def __call__(self,X):

        #sample class labels from a multinomial distribution
        labels = np.random.multinomial(1,self.prior,[self.patchSize,self.patchSize])
        labels = torch.tensor(labels).reshape([1,self.patchSize,self.patchSize,self.cds.shape[0]])
        labels = labels.permute(0,3,1,2)

        return labels

######################################################################################################################
#functions for visualization
######################################################################################################################

fs_labels = 35
fs_titles = 45
fontdict_labels = {"fontsize":fs_labels}
fontdict_titles = {"fontsize":fs_titles}

def plotter_testSet(indices_to_eval,model,DS_test,config,device):
    fig,axs = plt.subplots(len(indices_to_eval),4,figsize = (24,9 * len(indices_to_eval)))

    for i in range(len(indices_to_eval)):
        im,groundteruth = DS_test.__getitem__(indices_to_eval[i])

        #Get patches from the images that match the model requirements
        im_patch = prepare_image_torch(im.permute(1,2,0), config["patchSize"], rotation = False, mirroring = False, n=config["scalingFactor"], use_original=config["useOriginal"]).float()
        label_patch = prepare_image_torch(groundteruth[0], config["patchSize"], rotation = False, mirroring = False, n=config["scalingFactor"], use_original=config["useOriginal"]).long().to(device)[0]

        #Get the model prediction
        prediction = model(im_patch)[0]

        #Get the MAP labeling
        MAP_labling = torch.argmax(prediction,dim = 0)

        #Plot the image
        axs[i][0].set_title("True image\n",fontsize = 20)
        axs[i][0].axis("off")
        axs[i][0].imshow(np.transpose(im_patch[0]))

        #Plot the ground truth labeling
        axs[i][1].set_title("True labeling\n",fontsize = 20)
        axs[i][1].axis("off")
        im1 = axs[i][1].imshow(label_patch.T.cpu(),vmin = 1,vmax = 24)
        plt.colorbar(mappable=im1,ax = axs[i][1],fraction=0.046)
        
        #Get the prediction using the trained model
        axs[i][2].set_title("MAP prediction\n",fontsize = 20)
        axs[i][2].axis("off")
        im2 = axs[i][2].imshow(MAP_labling.detach().numpy().T,vmin = 1,vmax = 24)
        plt.colorbar(mappable=im2,ax = axs[i][2],fraction=0.046)

        #Plot the confusion matrix
        class_names = ["unlabeled","paved-area","dirt","grass","gravel","water","rocks","pool","vegetation","roof","wall","window","door","fence","fence-pole","person","dog","car","bicycle","tree","bald-tree","ar-marker","obstacle","conflicting"]
        cm = metrics.confusion_matrix(y_true = label_patch.flatten().cpu(), y_pred =MAP_labling.flatten(),labels = np.arange(len(class_names)))

        axs[i][3].set_title("Confusion matrix",fontsize = 20)
        axs[i][3].set_xticks(ticks = np.arange(len(class_names)))
        axs[i][3].set_xticklabels(labels = class_names,rotation=90,fontsize = 15)
        axs[i][3].set_yticks(ticks = np.arange(len(class_names)))
        axs[i][3].set_yticklabels(labels = class_names,rotation=0,fontsize = 15)
        axs[i][3].set_xlabel("Predicted class",fontsize = 20)
        axs[i][3].set_ylabel("True class",fontsize = 20)
        im3 = axs[i][3].imshow(cm)
        plt.colorbar(mappable=im3,ax = axs[i][3],fraction=0.046)

    plt.tight_layout()

def plotter_loss(tag):

    fig,axs = plt.subplots(3,1,figsize = (25,25))

    #Objective value
    axs[0].set_title("\nTraining objective\n",fontdict=fontdict_titles)
    axs[0].set_xlabel("training iteration",fontdict=fontdict_labels)
    axs[0].set_ylabel("objective",fontdict=fontdict_labels)
    axs[0].tick_params(axis='both', which='major', labelsize=fs_labels)

    #Load the training loss
    trainigLoss = np.loadtxt(f"results/{tag}/data/training_loss.txt",skiprows=1)

    #Load the validation loss
    validationLoss = np.loadtxt(f"results/{tag}/data/validation_loss.txt",skiprows=2)

    axs[0].plot(trainigLoss[:,0],trainigLoss[:,1],color = "k",marker = "o",label = "training loss")
    axs[0].plot(validationLoss[:,0],validationLoss[:,3],color = "b",marker = "o",label = "validation loss")
    axs[0].legend(fontsize = fs_labels)

    #F1 score
    axs[1].set_title("\nF1 score on the validation set\n",fontdict=fontdict_titles)
    axs[1].set_xlabel("training iteration",fontdict=fontdict_labels)
    axs[1].set_ylabel("F1 score",fontdict=fontdict_labels)
    axs[1].tick_params(axis='both', which='major', labelsize=fs_labels)

    f1validation = np.loadtxt(f"results/{tag}/data/F1_score.txt",skiprows=2)
    axs[1].plot(f1validation[:,0],f1validation[:,3],color = "k",marker = "o")

    #accuracy
    axs[2].set_title("\nAccuracy on the validation set\n",fontdict=fontdict_titles)
    axs[2].set_xlabel("training iteration",fontdict=fontdict_labels)
    axs[2].set_ylabel("accuracy",fontdict=fontdict_labels)
    axs[2].tick_params(axis='both', which='major', labelsize=fs_labels)

    accvalidation = np.loadtxt(f"results/{tag}/data/accuracy.txt",skiprows=2)
    axs[2].plot(accvalidation[:,0],accvalidation[:,3],color = "k",marker = "o")
    plt.tight_layout()

    return accvalidation

def eval_prior(DS_training):
    class_counts = np.zeros(24)

    for i in range(len(DS_training)):
        X,Y = DS_training.__getitem__(i)
        
        #Get the number of unique classes in the ground truth
        u,c = np.unique(Y,return_counts = True)

        class_counts[u] += c

    prior = class_counts / class_counts.sum()

    a = np.concatenate((np.arange(1,len(prior)+1).reshape(-1,1),prior.reshape(-1,1)),-1)
    with open("./empiricalPriorTrainingSet.txt","w") as file:
        np.savetxt(file,a)
    file.close()

    class_names = ["unlabeled","paved-area","dirt","grass","gravel","water","rocks","pool","vegetation","roof","wall","window","door","fence","fence-pole","person","dog","car","bicycle","tree","bald-tree","ar-marker","obstacle","conflicting"]

    fig,axs = plt.subplots(2,1,figsize = (25,25))

    axs[0].set_title(f"Prior distribution of the pixel values",fontdict=fontdict_titles)
    axs[0].set_xticks(ticks = np.arange(len(class_names)))
    axs[0].set_xticklabels(labels = class_names,rotation=90,fontsize = 15)
    axs[0].set_xlabel("Class name",fontdict=fontdict_labels)
    axs[0].set_ylabel("p(Class)",fontdict=fontdict_labels)
    axs[0].tick_params(axis='both', which='major', labelsize=fs_labels)
    axs[0].bar(x = np.arange(0,24),height=prior,width=1,color = "k")

    axs[1].set_title(f"Cumulative prior distribution of the pixel values",fontdict=fontdict_titles)
    axs[1].set_xticks(ticks = np.arange(len(class_names)))
    axs[1].set_xticklabels(labels = class_names,rotation=90,fontsize = 15)
    axs[1].set_xlabel("Class name",fontdict=fontdict_labels)
    axs[1].set_ylabel("Cumulative(p(Class))",fontdict=fontdict_labels)
    axs[1].tick_params(axis='both', which='major', labelsize=fs_labels)
    axs[1].bar(x = np.arange(0,24),height=np.cumsum(prior),width=1,color = "k")

    plt.tight_layout()

    return prior

def eval_performance(model,DS,config,device):

    F1_score = 0
    accuracy = 0

    for i in range(len(DS)):

        #Load the image
        X,Y = DS.__getitem__(i)

        #Select a patch from the image
        X_patch = prepare_image_torch(X.permute(1,2,0), config["patchSize"], rotation = False, mirroring = False, n=config["scalingFactor"], use_original=config["useOriginal"]).float()
        Y_patch = prepare_image_torch(Y[0], config["patchSize"], rotation = False, mirroring = False, n=config["scalingFactor"], use_original=config["useOriginal"]).long().to(device)[0]

        #Get the prediction form the model
        pred = model(X_patch)

        #Get the labeling 
        Y_pred = torch.argmax(pred, dim=1).cpu().numpy()
        
        #Compute the accuracy score
        acc = metrics.accuracy_score(y_true = Y_patch.flatten().cpu(), y_pred = Y_pred.flatten(),normalize = True)
        accuracy += acc

        #Compute the accuracy score
        F1 = metrics.f1_score(y_true = Y_patch.flatten().cpu(), y_pred = Y_pred.flatten(),average='weighted')
        F1_score += F1

    return accuracy / len(DS),F1_score / len(DS)

def plotter_classCouts(tag,config):
    fig = plt.figure(figsize = (25,25 / 3))
    ax = plt.axes()

    classesPerPatch = np.loadtxt(f"results/{tag}/data/classes_per_patch.txt",skiprows=1)
    classesPerPatch /= classesPerPatch.sum()

    r = classesPerPatch[config["classCountThreshold"]:].sum() / classesPerPatch.sum()

    ax.set_title(f"\nClasses per patch during the training\nacceptance rate: r = {round(r,4)}\n",fontdict=fontdict_titles)
    ax.set_xlabel("classes per patch",fontdict=fontdict_labels)
    ax.set_ylabel("relative count",fontdict=fontdict_labels)
    ax.tick_params(axis='both', which='major', labelsize=fs_labels)
    ax.bar(x = np.arange(0,config["classCountThreshold"]),height=classesPerPatch[:config["classCountThreshold"]],width=1,color = "orange",label = "below threshold")
    ax.bar(x = np.arange(config["classCountThreshold"],25),height=classesPerPatch[config["classCountThreshold"]:],width=1,color = "g",label = "above threshold")
    #ax.legend(fontsize = fs_labels)