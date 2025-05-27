#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 14:41:53 2021

@author: jannik
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import pathlib
from preprocessing import normalize_01
from unet import UNet
from utils import get_filenames_of_path




# Root directory
root = pathlib.Path.home() / 'root' / 'of'/ 'project' / 'folder' # Change this to the root directory of the project

# MODEL TO BE LOADED
# =============================================================================
model_name = "p3D3t" # Options: 2D, p3D3t, p3D5t
path_model = root / 'path-to-saved-neural-network.pt' # Change this to the path of the saved model

# DATA TO BE PREDICTED ON
# =============================================================================
# Input files
# The script expects a subfolder for each model in the "Input" folder, e.g. root/input/2D/<input_files>
input_names = get_filenames_of_path(root / ('input/'+model_name)) 

# Save plot(s) in root directory?
save = False








# Channels
if model_name == "2D":
    in_c = 2
elif model_name == "p3D3t":
    in_c = 6
elif model_name == "p3D5t":
    in_c = 10
else:
    raise ValueError("Unknown model_name. Valid options are: '2D', 'p3D3t', 'p3D5t'")
out_c = 2
# Network architecture
n_blocks = 6
startFilt = 128
dim = 2
activation='leaky'
normalization='batch'
# Input patch overlap on each side (0 if input and target have equal H and W)
overlap = 64  

# Device
print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


model.load_state_dict(torch.load(path_model,map_location=torch.device(device)))



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





# ******************************** Predictions *******************************

# Read images and store them in memory
images = (np.load(img_name) for img_name in input_names)


# Predict on the data
for i,img in zip(range(len(input_names)),images): 
    y_pred = predict(img, model, preprocess, postprocess, device)   
    
    tstep = str(input_names[i].name)[0:3]
    simulation = str(input_names[i].name)[4:].partition("_200")[0]
    patch = str(input_names[i].name).partition("c_")[2].partition(".npy")[0]
    
    # Identify the center time step for the input plots
    center_tstep = int(in_c / 2)
    
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(11,3), gridspec_kw={"wspace":0.07})
    im_rint = axs[0].contourf(np.flipud(img[overlap:-overlap,overlap:-overlap,center_tstep-1]), np.logspace(0,2,5), norm=mpl.colors.LogNorm(),
                              cmap=plt.cm.BuPu, vmin=0.5, vmax=100, extend="both") 
    axs[0].set_title("Rain intensity [mm/h]")
    im_ctt = axs[1].imshow(img[overlap:-overlap,overlap:-overlap,center_tstep], cmap=plt.cm.viridis, vmin=215, vmax=304)  
    axs[1].set_title("Cloud top temperature [K]")
    im_pred = axs[2].imshow(y_pred, cmap=mpl.colors.ListedColormap(["white","black"]))
    axs[2].set_title("Cold pool prediction")
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im_rint, ax=axs[0])
    fig.colorbar(im_ctt, ax=axs[1]).set_ticks([220,260,300])
    fig.colorbar(im_pred, ax=axs[2]).set_ticks([0,1])
    if save:
        plt.savefig(root / (tstep+"_"+simulation+"_patch"+patch+"_"+model_name+"-prediction.png"),
                    bbox_inches="tight", dpi=300)
    plt.show()



