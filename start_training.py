#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 15:18:09 2021

@author: jannik
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from preprocessing import DataSampler, ComposeDouble, FunctionWrapperDouble, normalize_01
from unet import UNet
from trainer import Trainer
import pathlib
from sklearn.model_selection import train_test_split
from metrics import DiceLoss, DiceCeLoss
from utils import get_filenames_of_path
from postprocessing import Postprocessor





# MODEL
# =============================================================================
# Channels
in_c = 6 # Options: 2 (2D), 6 (p3D3t), 10 (p3D5t)
out_c = 2

# Network depth
n_blocks = 6
startFilt = 128
dim = 2
activation='leaky'
normalization='batch'

# Hyperparameters
learningRate = 0.00001
learningRateScheduler = True
batchSize = 8
epochs = 40

# Save model state?
save_model = True

# Target validation accuracy (stop criterion)
target_accuracy = 1.0


# DATASET
# =============================================================================
dataset_name = 'train_p3D3t'

# Root directory containing "TrainingSet" folder (inputs & targets), "Output/TrainingDataframes" folder, "SavedModels" folder 
root = pathlib.Path.home() / 'root' / 'of'/ 'project' / 'folder'

# Patch overlap on each side (0 if input and target have equal H and W)
overlap = 64

# Random state
random_seed = 30

# Training/validation split
train_size = 0.75  




# ANALYSIS
# =============================================================================
# Confusion matrix
save_confusion = False

# Loss and accuracy
save_metrics = True











# Input and target files
inputs = get_filenames_of_path(root / ('TrainingSet/Input/'+dataset_name))
targets = get_filenames_of_path(root / 'TrainingSet/Target')

# Training transformations and augmentations
transforms = ComposeDouble([
    #FunctionWrapperDouble(create_dense_target, input=False, target=True),
    #FunctionWrapperDouble(np.moveaxis, input=True, target=False, source=-1, destination=0),
    FunctionWrapperDouble(normalize_01)
])

# Split the dataset
inputs_train, inputs_valid, targets_train, targets_valid = train_test_split(
    inputs,
    targets,
    random_state=random_seed,
    train_size=train_size,
    shuffle=True)
 
# Datasets
dataset_train = DataSampler(inputs=inputs_train,
                            targets=targets_train,
                            transform=transforms)
dataset_valid = DataSampler(inputs=inputs_valid,
                            targets=targets_valid,
                            transform=transforms)

# Dataloader
dataloader_training = DataLoader(dataset=dataset_train,
                                  batch_size=batchSize,
                                  shuffle=True)
dataloader_validation = DataLoader(dataset=dataset_valid,
                                    batch_size=batchSize,
                                    shuffle=True)

# Device
print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'

# Model
model = UNet(in_channels=in_c,
              out_channels=out_c,
              n_blocks=n_blocks,
              start_filters=startFilt,
              activation=activation,
              normalization=normalization,
              conv_mode='same',
              dim=dim,
              overlap=overlap)
model.to(device)

# Criterion
#criterion = torch.nn.CrossEntropyLoss()
#criterion = DiceLoss(out_channels=out_c)
criterion = DiceCeLoss(out_channels=out_c,alpha=0.5)
loss = 'diceCe'

# Optimizer
#optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

# Learning rate scheduler
if learningRateScheduler:
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
else:
    lr_scheduler = None

# Analysis
analysis = Postprocessor(dataset_name=dataset_name,
                         root=root,
                         save_confusion=save_confusion,
                         save_metrics=save_metrics)

# Trainer
trainer = Trainer(model=model,
                  device=device,
                  criterion=criterion,
                  optimizer=optimizer,
                  analysis=analysis,
                  training_DataLoader=dataloader_training,
                  validation_DataLoader=dataloader_validation,
                  lr_scheduler = lr_scheduler,
                  epochs=epochs,
                  epoch=0,
                  target_accuracy=target_accuracy)

# Train and validate the model
trainer.run_trainer()

# Save model for inference
if save_model:

    torch.save(model.state_dict(), root / ("SavedModels/"+dataset_name+"_"+str(n_blocks)+"blocks_"+str(startFilt)+
               "filter_"+str(batchSize)+"batch_"+str(learningRate)+"lrScheduled"+str(learningRateScheduler)+"_"+str(epochs)+"epochs_"+activation+normalization+loss+".pt"))







