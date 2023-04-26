import utils.ObjectiveFunctions as of
import utils.unet_utils as ut
import torch
import matplotlib.pyplot as plt
import numpy as np

######################################################################################################################
#Dictionaries to select objective function, optimizer andthe model family
######################################################################################################################

#Possible loss functions
loss_dict = {
    "dice": of.DiceLoss,
    "combo":of.CriterionCombo,
    "focal": of.FocalLoss
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
#functions for visualization
######################################################################################################################

def plotter_loss(tag):

    fs_labels = 20
    fs_titles = 30
    fontdict_labels = {"fontsize":fs_labels}
    fs_labels = 20
    fontdict_titles = {"fontsize":fs_titles}

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

    f1validation = np.loadtxt(f"results/{tag}/data/accuracy.txt",skiprows=2)
    axs[2].plot(f1validation[:,0],f1validation[:,3],color = "k",marker = "o")
    plt.tight_layout()

    return validationLoss