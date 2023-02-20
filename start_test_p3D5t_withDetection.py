#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 14:41:53 2021

@author: jannik
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
import pandas as pd
import seaborn as sns
import torch
import pathlib
from preprocessing import normalize_01
from unet import UNet
from postprocessing import compute_confusionMatrix
from skimage.measure import label


def get_filenames_of_path(path: pathlib.Path, ext: str = '*'):
    """Returns a list of files in a directory/path. Uses pathlib."""
    filenames = [file for file in sorted(path.glob(ext)) if file.is_file()]
    return filenames


# Export/save dataframes?
save_df = True

# Root directory
root = pathlib.Path.home() / 'labspaces' / 'jannik-cp-detection-project' / 'data' / 'work' / 'unet_data'



# MODEL TO BE LOADED
# =============================================================================
path_model = root / 'SavedModels/train_pseudo3D5t_overlap64_augmented_4perc_lessCpOnly_6blocks_128filter_8batch_1e-05lrScheduledTrue_22epochs_leakybatchdiceCe_train1.pt'
# Channels
in_c = 10
out_c = 2
# Network depth
n_blocks = 6
startFilt = 128
dim = 2
activation='leaky'
normalization='batch'


# DATASET
# =============================================================================
input_name_test = 'test_pseudo3D5t_overlap64_augmented'
# Patch overlap on each side (0 if input and target have equal H and W)
overlap = 64  


# DETECTION
# =============================================================================
minSize = 25


# Input and target files
input_names = get_filenames_of_path(root / ('Test/Input/'+input_name_test))
target_names = get_filenames_of_path(root / 'Test/Target/test_augmented')

# Device
print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model
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


model.load_state_dict(torch.load(path_model))





# ******************************** Functions *******************************


def predict(img,
            model,
            preprocess,
            postprocess,
            device,
            ):
    model.eval()
    img = preprocess(img)  # preprocess image
    x = torch.from_numpy(img).to(device)  # to torch, send to device
    with torch.no_grad():
        out = model(x)  # send through model/network
    y_pred = torch.argmax(out.data, 1)
    y_pred = postprocess(y_pred)
    return y_pred

# Preprocess function
def preprocess(img: np.ndarray):
    img = img.transpose((2,0,1)) # from [H, W, C] to [C, H, W]
    img = normalize_01(img)  # linear scaling to range [0-1]
    img = np.expand_dims(img, axis=0)  # add batch dimension: 2D [B, C, H, W], 3D [B, C, T, H, W]
    img = img.astype(np.float32)  # typecasting to float32
    return img


# Postprocess function
def postprocess(img: torch.tensor):
    img = img.cpu().numpy()  # send to cpu and transform to numpy.ndarray
    img = np.squeeze(img)  # remove batch dim  -> 2D: [C, H, W], 3D: [C, T, H, W]
    return img


# Modified numpy unique function that drops the value 0
def unique_nonzero(array, return_counts=False):    
    if return_counts:
        labels, counts = np.unique(array,return_counts=True)
        if labels[0] == 0:                    
            labels = labels[1:]
            counts = counts[1:]
        return labels, counts
    else:
        labels = np.unique(array,return_counts=False)
        if labels[0] == 0:                    
            labels = labels[1:]
        return labels


def track(newLabeledField,oldLabeledField,trackingFactor=0.5,oldFactor=None):
    """
    Function to compare two labeled fields and track overlapping patches based on a tracking factor.

    Parameters
    ----------
    newLabeledField : array_like
        2D array with labeled reference patches.
    oldLabeledField : array_like
        2D array with labeled patches to compare with the reference patches.
    trackingFactor : float, optional
        Required overlap (proportion of the new patch) between two patches to be tracked. 
        The default is 0.5.
    maxFactor : float, optional
        If specified, patches will only be tracked of they are not larger than maxFactor x reference patch. 
        The default is None.
        
    Returns a labeled field with patches relabeled based on the tracking.
    -------
    blobs_new : array_like
        2D array with relabeled patches based on the tracking.
    """
    
    blobs_new = newLabeledField
    blobs_old = oldLabeledField
    tf = trackingFactor
    tfold = oldFactor
    
    blob_labels_new, blob_counts_new = unique_nonzero(blobs_new,return_counts=True)
    blob_labels_old, blob_counts_old = unique_nonzero(blobs_old,return_counts=True)    
        
    l = 0             
    k = 0
    updated_index = []
    updated_label = []
    
    for blob in blob_labels_new:
        blob_region = blobs_new == blob
        overlap = blob_region * blobs_old
        unique, number = unique_nonzero(overlap, return_counts=True)                  
        itemindex = np.where(unique==blob)
        unique = np.delete(unique, itemindex)
        number = np.delete(number, itemindex)
        if len(unique) > 0:
            if tfold is not None:
                if (np.max(number) > (blob_counts_new[k]*tf)) and (np.max(number) > blob_counts_old[np.where(blob_labels_old==unique[np.argmax(number)])]*tfold):
                    #print("Overlap detetcted: Patch " + str(blob_labels_new[k]) + " is patch " + str(unique[np.argmax(number)]))
                    updated_index.append(k)
                    updated_label.append(unique[np.argmax(number)])
            else:
                if np.max(number) > (blob_counts_new[k]*tf):
                    #print("Overlap detetcted: Patch " + str(blob_labels_new[k]) + " is patch " + str(unique[np.argmax(number)]))
                    updated_index.append(k)
                    updated_label.append(unique[np.argmax(number)])
        k += 1
    
    for index in updated_index:
        blobs_new = np.where(blobs_new==blob_labels_new[index],updated_label[l],blobs_new)                     
        l += 1
        
    return blobs_new




# ******************************** Test *******************************

# Read images and store them in memory
if in_c != 1:
    images = (np.load(img_name) for img_name in input_names)
else:
    images = (np.expand_dims(np.load(img_name), axis=2) for img_name in input_names)
targets = (np.load(tar_name) for tar_name in target_names)

# Derive classes
classes = [str(i) for i in list(range(out_c))]

# Lists to store the CP detection data
cps_list = []
detection_list = []

# Predict on the test data and collect metrics
metricsNoCp_list = [] # list to store stats for all samples without any CP pixel
metricsOnlyCp_list = [] # list to store stats for all samples with only CP pixel
metrics_list = [] # list to store stats for all samples with at least one CP pixel
i = 0
for img,target in zip(images,targets): 
    y_pred = predict(img, model, preprocess, postprocess, device)   
    
    cf_matrix_df = compute_confusionMatrix(y_pred=y_pred.flatten(), 
                                           y_true=target.flatten(), 
                                           classes=classes)
    tn = cf_matrix_df[classes[0]][classes[0]]
    fp = cf_matrix_df[classes[1]][classes[0]]
    fn = cf_matrix_df[classes[0]][classes[1]]
    tp = cf_matrix_df[classes[1]][classes[1]]
    
    tstep = int(str(target_names[i].name)[0:3])
    simulation = str(target_names[i].name)[4:].partition("_200")[0]
    patch = str(target_names[i].name).partition("binaryLabels_")[2].partition(".npy")[0]
    mean_cpCoverage = np.mean(target)
    
    # Label ground truth and predicted CP blobs
    target_labels = label(target,connectivity=1)
    pred_labels = label(y_pred,connectivity=1)   
    
    # Check if target CPs are smaller (have less pixel) than minSize and drop them if yes
    lbls, lbls_count = unique_nonzero(target_labels, return_counts=True)
    l = 0
    for lbl in lbls:
        if lbls_count[l] < minSize:
            target_labels = np.where(target_labels==lbl,0,target_labels)
        l += 1     
    
    # Check if predicted CPs are smaller (have less pixel) than minSize and drop them if yes
    lbls, lbls_count = unique_nonzero(pred_labels, return_counts=True)
    l = 0
    for lbl in lbls:
        if lbls_count[l] < minSize:
            pred_labels = np.where(pred_labels==lbl,0,pred_labels)
        l += 1   
     
    # Change the remaining labels to start after the max of the target labels
    label_list = unique_nonzero(pred_labels)       
    j = len(label_list)
    targetMax = np.max(target_labels)
    for labl in reversed(label_list):
        pred_labels = np.where(pred_labels==labl,targetMax+j,pred_labels)
        j -= 1   

    # Check if the predicted CPs overlap with ground truth CPs for more than 50% of their area covering more than 50% of the ground truth area
    pred_labels = track(newLabeledField=pred_labels,oldLabeledField=target_labels,
                               trackingFactor=0.5,oldFactor=0.5)    
    
    # Get an updated list of the predicted CPs and a list with the ground truth CPs
    target_cps, target_cps_count = unique_nonzero(target_labels, return_counts=True)
    pred_cps, pred_cps_count = unique_nonzero(pred_labels, return_counts=True)
    
    # Loop over the ground truth CPs and check if they were detected
    l = 0
    for cp in target_cps:
        if cp in pred_cps:
            cp_sample = [tstep,patch,simulation,"p3D5t",target_cps_count[l],True]
            cps_list.append(cp_sample)
        else:
            cp_sample = [tstep,patch,simulation,"p3D5t",target_cps_count[l],False]
            cps_list.append(cp_sample)            
        l += 1
    
    # Evaluate the overall detections
    total_cps = len(target_cps)
    detected_cps = len(list(set(target_cps) & set(pred_cps)))
    false_cps = len(pred_cps) - detected_cps
    detection_sample = [tstep,patch,simulation,"p3D5t",total_cps,detected_cps,false_cps]
    detection_list.append(detection_sample)
    
    # Evaluate the metrics
    if np.sum(target) == 0:
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        metrics_sample = [tstep, mean_cpCoverage, accuracy]
        metricsNoCp_list.append(metrics_sample)
    elif np.all(target) == 1:
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        metrics_sample = [tstep, mean_cpCoverage, accuracy]
        metricsOnlyCp_list.append(metrics_sample)
    else:        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        iou = tp / (tp + fp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = (2 * precision * recall) / (precision + recall)
        specificity = tn / (tn + fp)
        metrics_sample = [tstep, mean_cpCoverage, accuracy, iou, precision, recall, f1, specificity]
        metrics_list.append(metrics_sample)
    i += 1

# Convert list to dataframe
metrics_df = pd.DataFrame(metrics_list, columns=["Timestep", "CP_Coverage", "Accuracy", "IOU", "Precision", "Recall", "F1", "Specificity"])
metricsNoCp_df = pd.DataFrame(metricsNoCp_list, columns=["Timestep", "CP_Coverage", "Accuracy"])
metricsOnlyCp_df = pd.DataFrame(metricsOnlyCp_list, columns=["Timestep", "CP_Coverage", "Accuracy"])
cps_df = pd.DataFrame(cps_list, columns=["Timestep", "Patch", "Simulation", "Model", "Area", "Detected"])
detection_df = pd.DataFrame(detection_list, columns=["Timestep", "Patch", "Simulation", "Model", "CPsTotal", "CPsDetected", "CPsFalse"])

# Save/export dataframes
if save_df:
    metrics_df.to_pickle(root / ("Test/Output/Dataframes/metrics_df_"+input_name_test+".pkl"))
    metricsNoCp_df.to_pickle(root / ("Test/Output/Dataframes/metricsNoCp_df_"+input_name_test+".pkl"))
    metricsOnlyCp_df.to_pickle(root / ("Test/Output/Dataframes/metricsOnlyCp_df_"+input_name_test+".pkl"))
    cps_df.to_pickle(root / ("Test/Output/Dataframes/cps_df_"+input_name_test+".pkl"))
    detection_df.to_pickle(root / ("Test/Output/Dataframes/detection_df_"+input_name_test+".pkl"))
    

# Plot main statistics
print("Gobal statistics:")
print("no CP mean accuracy: " + str(metricsNoCp_df["Accuracy"].mean()))
print("only CP mean accuracy: " + str(metricsOnlyCp_df["Accuracy"].mean()))
print("CP mean accuracy: " + str(metrics_df["Accuracy"].mean()))
print("CP mean IOU: " + str(metrics_df["IOU"].mean()))
print("CP mean precision: " + str(metrics_df["Precision"].mean()))
print("CP mean recall: " + str(metrics_df["Recall"].mean()))
print("CP mean F1: " + str(metrics_df["F1"].mean()))
print("CP mean specificity: " + str(metrics_df["Specificity"].mean()))






