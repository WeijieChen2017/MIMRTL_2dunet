import os
import glob
import cv2
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom

from PIL import Image

def maxmin_norm(data):
    MAX = np.amax(data)
    MIN = np.amin(data)
    data = (data - MIN)/(MAX-MIN)
    return data

def create_index(dataA, n_slice):
    h, w, z = dataA.shape
    index = np.zeros((z,n_slice))
    
    for idx_z in range(z):
        for idx_c in range(n_slice):
            index[idx_z, idx_c] = idx_z-(n_slice-idx_c+1)+n_slice//2+2
    index[index<0]=0
    index[index>z-1]=z-1
    return index

def slice5_AB(dataA, dataB, name_dataset, n_slice=1, name_tag="", resize_f=1):
    # shape supposed to be 512*512*284 by default
    assert dataA.shape == dataB.shape, ("DataA should share the same shape with DataB.")
    path2save = "./pytorch-CycleGAN-and-pix2pix/datasets/"+name_dataset+"/train/"
    h, w, z = dataA.shape
    h = h*resize_f
    w = w*resize_f
    img = np.zeros((n_slice, h, w*2))
    index = create_index(dataA, n_slice)
    print(index)
        
    for idx_z in range(z):
        for idx_c in range(n_slice):
            img[idx_c, :, :w] = zoom(dataA[:, :, int(index[idx_z, idx_c])], zoom=resize_f)
            img[idx_c, :, w:] = zoom(dataB[:, :, int(index[idx_z, idx_c])], zoom=resize_f)
        name2save = path2save+name_tag+"_"+str(idx_z)+".npy"
        np.save(name2save, img)
    print(str(z)+" images have been saved.")

def slice5_A(dataA, name_dataset, n_slice=1, name_tag="", resize_f=1):
    # shape supposed to be 512*512*284 by default
    path2save = "./pytorch-CycleGAN-and-pix2pix/datasets/"+name_dataset+"/test/"
    h, w, z = dataA.shape
    h = h*resize_f
    w = w*resize_f
    img = np.zeros((n_slice, h, w))
    index = create_index(dataA, n_slice)
    print(index)
        
    for idx_z in range(z):
        for idx_c in range(n_slice):
            img[idx_c, :, :] = zoom(dataA[:, :, int(index[idx_z, idx_c])], zoom=resize_f)
        name2save = path2save+name_tag+"_"+str(idx_z)+".npy"
        np.save(name2save, img)
    print(str(z)+" images have been saved.")

name_dataset = "stick"
n_slice = 7

import os

for folder_name in ["train", "test", "trainA", "trainB", "testA", "testB"]:
    path = "./pytorch-CycleGAN-and-pix2pix/datasets/"+name_dataset+"/"+folder_name+"/"
    if not os.path.exists(path):
        os.makedirs(path)

list_ori = glob.glob("./data/"+name_dataset+"/test/*.nii")
list_ori.sort()
print("Test:")
for path_ori in list_ori:
    filename_ori = os.path.basename(path_ori)[:]
    filename_ori = filename_ori[:filename_ori.find(".")]
    print(filename_ori)
    data_ori = maxmin_norm(nib.load(path_ori).get_fdata())
    slice5_A(dataA=data_ori, name_dataset=name_dataset, n_slice=n_slice, name_tag=filename_ori, resize_f = 1)
    print("------------------------------------------------------------------------")

list_ori = glob.glob("./data/"+name_dataset+"/pure/*.nii")
list_ori.sort()
for path_ori in list_ori:
    print("TrainA:")
    filename_ori = os.path.basename(path_ori)[:]
    filename_ori = filename_ori[:filename_ori.find(".")]
    print(filename_ori)
    data_ori = maxmin_norm(nib.load(path_ori).get_fdata())
    
    list_sim = glob.glob("./data/"+name_dataset+"/blur/*"+filename_ori+"*.nii")
    list_sim.sort()
    
    for path_sim in list_sim:
        print("Pairs")
        filename_sim = os.path.basename(path_sim)[:]
        filename_sim = filename_sim[:filename_sim.find(".")]
        print("A:", filename_ori)
        print("B:", filename_sim)
                
        data_sim = maxmin_norm(nib.load(path_sim).get_fdata())
        slice5_AB(dataA=data_ori, dataB=data_sim,
                  name_dataset=name_dataset, n_slice=n_slice,
                  name_tag=filename_sim, resize_f=1)
        
    print("------------------------------------------------------------------------")
        
        
        
#         data_ori = nib.load(path_ori).get_fdata()
#         norm_ori = maxmin_norm(data_ori)*255
#         sliced_save(data=norm_ori,
#                     name_tag=os.path.basename(path_ori)[:-4],
#                     path2save="./pytorch-CycleGAN-and-pix2pix/datasets/"+name_dataset+"/"+folder_name+"/",
#                     n_slice=n_slice)