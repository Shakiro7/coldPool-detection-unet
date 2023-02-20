#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 14:34:31 2022

@author: jannik
"""

import numpy as np
import pathlib


class Postprocessor:
    def __init__(self,
                 dataset_name: str,
                 root: pathlib.PosixPath,
                 save_confusion: bool = False,
                 save_metrics: bool = False                 
                 ):
        
        self.dataset_name = dataset_name
        self.root = root
        self.save_confusion = save_confusion
        self.save_metrics = save_metrics
        
 
        
# Function to plot the training metrics 
# Source: https://towardsdatascience.com/creating-and-training-a-u-net-model-with-pytorch-for-2d-3d-semantic-segmentation-dataset-fb1f7f80fe55
def plot_training(training_accuracy,
                  training_losses,
                  training_iou,
                  validation_accuracy,
                  validation_losses,
                  validation_iou,
                  figsize=(15, 5)
                  ):
    """
    Returns a plot with metrics for training and validation.
    """
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    list_len = len(training_losses)
    x_range = list(range(1, list_len + 1))  # number of x values

    fig = plt.figure(figsize=figsize)
    grid = gridspec.GridSpec(ncols=3, nrows=1, figure=fig)

    subfig1 = fig.add_subplot(grid[0, 0])
    subfig2 = fig.add_subplot(grid[0, 1])
    subfig3 = fig.add_subplot(grid[0, 2])

    subfigures = fig.get_axes()

    for i, subfig in enumerate(subfigures, start=1):
        subfig.spines['top'].set_visible(False)
        subfig.spines['right'].set_visible(False)

    linestyle = '-'
    color_train = 'darkblue'
    color_val = 'darkorange'
    alpha = 1.0

    # Subfig 1
    subfig1.plot(x_range, training_losses, linestyle, color=color_train, label='Training',
                 alpha=alpha)
    subfig1.plot(x_range, validation_losses, linestyle, color=color_val, label='Validation',
                 alpha=alpha)
    subfig1.title.set_text('Training and Validation Loss')
    subfig1.set_xlabel('Epoch')
    subfig1.legend(loc='upper right')

    # Subfig 2    
    subfig2.plot(x_range, training_accuracy, linestyle, color=color_train, label='Training',
                 alpha=alpha)
    subfig2.plot(x_range, validation_accuracy, linestyle, color=color_val, label='Validation',
                 alpha=alpha)
    subfig2.title.set_text('Training and Validation Accuracy')
    subfig2.set_xlabel('Epoch')
    subfig2.legend(loc='upper left')   
    
    # Subfig 3    
    subfig3.plot(x_range, training_iou, linestyle, color=color_train, label='Training',
                 alpha=alpha)
    subfig3.plot(x_range, validation_iou, linestyle, color=color_val, label='Validation',
                 alpha=alpha)
    subfig3.title.set_text('Training and Validation IOU')
    subfig3.set_xlabel('Epoch')
    subfig3.legend(loc='upper left')     
    
    return fig    
 



def compute_confusionMatrix(y_pred, 
                            y_true,
                            classes):
    
    from sklearn.metrics import confusion_matrix
    import pandas as pd    

    cf_matrix = confusion_matrix(y_true=y_true,
                                 y_pred=y_pred,
                                 labels=[0,1],
                                 normalize=None)
      
    cf_matrix_df = pd.DataFrame(cf_matrix,
                                index=[i for i in classes],
                                columns=[i for i in classes])    
        
    return cf_matrix_df
    

def plot_confusionMatrix(y_pred, 
                         y_true,
                         classes,
                         normalize=None):
    
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import pandas as pd    

    cf_matrix = confusion_matrix(y_true=y_true,
                                 y_pred=y_pred,
                                 normalize=normalize)
    cf_matrix_df = pd.DataFrame(cf_matrix,
                                index=[i for i in classes],
                                columns=[i for i in classes])
    
    s = sns.heatmap(cf_matrix_df, annot=True)
    s.set(xlabel='Predicted class', ylabel='True class')    
        
    return s
    
    
    
if __name__ == "__main__":

    classes1 = ('background', 'chair', 'dog', 'car')
    out1 = np.array([[0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3]]).T
    target1 = np.array([[0,0,0,1,2,1,1,1,2,3,2,2,2,1,1,3,3,3,3,0]]).T
    
    splot = plot_confusionMatrix(y_pred=out1,
                                 y_true=target1,
                                 classes=classes1,
                                 normalize='true')
    
    
    classes2 = [str(i) for i in list(range(2))]
    #classes2 = ('background', 'cp')
    #out2 = np.array([[0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,1,1,1,0,0]]).T
    #target2 = np.array([[0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,0]]).T    
    out2 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    target2 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])     
    
    
    print(len(classes2))
    print(out2.shape)
    print(target2.shape)
    cf_matrix_df = compute_confusionMatrix(y_pred=out2, 
                                            y_true=target2, 
                                            classes=classes2)

    tn = cf_matrix_df[classes2[0]][classes2[0]]
    fp = cf_matrix_df[classes2[1]][classes2[0]]
    fn = cf_matrix_df[classes2[0]][classes2[1]]
    tp = cf_matrix_df[classes2[1]][classes2[1]]
    

   