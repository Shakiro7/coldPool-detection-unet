#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 11:06:49 2021

@author: jannik
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, out_channels=2, weight=None, size_average=True):
        super(DiceLoss, self).__init__()
        self.out_channels = out_channels

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)
        inputs = torch.softmax(inputs, dim=1)
        targets = F.one_hot(targets, num_classes = self.out_channels).permute(0,3,1,2).contiguous()

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()                         
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  

        return 1 - dice


class DiceCeLoss(nn.Module):
    def __init__(self, out_channels=2, alpha=0.5, weight=None, size_average=True):
        super(DiceCeLoss, self).__init__()
        self.out_channels = out_channels
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()

    def forward(self, inputs, targets, smooth=1):

        ce = self.ce(inputs, targets)
        
        inputs = torch.softmax(inputs, dim=1)
        targets = F.one_hot(targets, num_classes = self.out_channels).permute(0,3,1,2).contiguous()

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()                         
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)                 

        return self.alpha * (1 - dice) + (1-self.alpha) * ce




def accuracy(out, target):
    yPred = torch.argmax(out.data, 1)
    correct = (yPred == target).sum().item()
    if out.ndim == 4:
        acc = 100*correct / (yPred.shape[-1]*yPred.shape[-2]*yPred.shape[0])
    else:
        acc = 100*correct / (yPred.shape[-1]*yPred.shape[-2]*yPred.shape[0]*yPred.shape[1])
    return acc


def miou(out, target, num_classes):
    from torchmetrics import JaccardIndex
    jaccard = JaccardIndex(num_classes=num_classes,absent_score=1.0)
    iou_score = jaccard(out.cpu(), target.cpu())
    return iou_score
