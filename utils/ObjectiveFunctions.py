import torch.nn as nn
import segmentation_models_pytorch as smp
import torch
import numpy as np

def CriterionCombo(inputs,targets):

    #Get One Hot encoding of the true labels
    y_true = torch.nn.functional.one_hot(targets,24).permute(0,3,1,2).float()

    assert y_true.shape == inputs.shape
    
    BCELoss = smp.losses.SoftBCEWithLogitsLoss()
    TverskyLoss = smp.losses.TverskyLoss(mode = 'multilabel', log_loss = False)

    L1 = BCELoss(y_pred = inputs,y_true = y_true)
    L2 = TverskyLoss(y_pred = inputs,y_true = y_true)

    loss = 0.5 * L1 + 0.5 * L2

    return loss

def DiceLoss(inputs,targets):
    
    #Use Tversky loss with alpha = beta = 0.5 for dice loss
    #This has the advantage of working better with possible class imbalance

    #Get One Hot encoding of the true labels
    y_true = torch.nn.functional.one_hot(targets,24).permute(0,3,1,2).float()

    assert y_true.shape == inputs.shape
    TverskyLoss = smp.losses.TverskyLoss(mode = 'multilabel', log_loss = False)

    loss = TverskyLoss(y_pred = inputs,y_true = y_true)

    return loss

def FocalLoss(inputs,targets):

    #Use Focal loss

    #Get One Hot encoding of the true labels
    y_true = torch.nn.functional.one_hot(targets,24).permute(0,3,1,2).float()

    assert y_true.shape == inputs.shape
    FocalLoss = smp.losses.FocalLoss(mode = 'multilabel')

    loss = FocalLoss(y_pred = inputs.contiguous(),y_true = y_true.contiguous())

    return loss