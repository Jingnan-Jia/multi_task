#!/usr/bin/env python
# coding: utf-8

# In[1]:

import copy
import glob
import numpy as np
import csv
from medutils.medutils import get_all_ct_names, load_itk, save_itk
from monai.transforms import LoadImaged, AddChanneld, Orientationd, Spacingd, ScaleIntensityRanged, SpatialPadd,     RandAffined, RandCropByPosNegLabeld, RandGaussianNoised, RandSpatialCropd, CastToTyped, ToTensord, AsDiscreted, Resize, Spacing
import matplotlib.pyplot as plt
from monai.networks import one_hot
import torch


# In[2]:

mask_fpath = "/data/jjia/multi_task/mt/scripts/data/data_ori_space/lobe/GLUCOLD_patients_01_seg.nii.gz"
img_ori_intensity, ori, sp = load_itk(mask_fpath, require_ori_sp=True)
img_ori = (img_ori_intensity - np.min(img_ori_intensity))/ (np.max(img_ori_intensity) - np.min(img_ori_intensity))
low_sp = Resize(spatial_size=(256, 256, 256), mode='nearest')
hgh_sp_linear = Resize(spatial_size=img_ori.shape, mode="trilinear")
hgh_sp_nearest = Resize(spatial_size=img_ori.shape, mode='nearest')


# In[3]:


img_ori = img_ori[None]


# In[4]:


img_low_res = low_sp(img_ori)


# In[5]:

img_hgh_res_linear = hgh_sp_linear(img_low_res)


# In[6]:
img_hgh_res_nearest = hgh_sp_nearest(img_low_res)


# I am surprised that nearest neighbor mode cost more time than linear model

# In[7]:


slice_idx = 350
plt.figure(figsize=(7, 21))
plt.subplot(3,1,1)
plt.imshow(img_ori[0][slice_idx])
plt.title('original mask')
plt.subplot(3,1,2)
plt.imshow(img_hgh_res_linear[0][slice_idx])
plt.title('Nearest downsampling + Linear upsamplilng mask')
plt.subplot(3,1,3)
plt.imshow(img_hgh_res_nearest[0][slice_idx])
plt.title('Nearest downsampling + Nearest upsamplilng mask')
plt.show()


# From the above image, we can see the linear interpolation can introduce some values outsite the 6 classes. This is not what we want. The nearest upsamplling will introduce some zigzag pattern, which should be removed as well.
# I hope to have upsample method which can keep 6 classes and no ziazag pattern.
# So I can at first seperate the low resolution image to 6 different channels (one hot encoding), then do linear upsampling for each channel, then binarize each channel, then merge the 6 channels to one (one hot decoding).

# In[9]:
# img_ori_intensity = img_ori_intensity[None]
# img_low_res = low_sp(img_ori_intensity)  # require positive integer labels
# # img_low_res = img_low_res[None]
# img_low_res = torch.tensor(img_low_res)
#
# img_low_res_6chn = one_hot(img_low_res, num_classes=6, dim=0)
#
# img_low_res_6chn = img_low_res_6chn.cpu().detach().numpy()
#
# a0 = hgh_sp_linear(img_low_res_6chn[0][None])
# a1 = hgh_sp_linear(img_low_res_6chn[1][None])
# a2 = hgh_sp_linear(img_low_res_6chn[2][None])
# a3 = hgh_sp_linear(img_low_res_6chn[3][None])
# a4 = hgh_sp_linear(img_low_res_6chn[4][None])
# a5 = hgh_sp_linear(img_low_res_6chn[5][None])
#
# img_hgh_res_6chn = torch.tensor(np.array([a0, a1, a2, a3, a4, a5]))
#
# print([np.max(img_low_res_6chn[i]) for i in range(5)])
#
# plt.figure(figsize=(8,8))
# plt.imshow(img_hgh_res_6chn[3][0][300].cpu().detach().numpy())
# plt.show()
#
# img_hgh_res_6chn[img_hgh_res_6chn>=0.5] = 1
# img_hgh_res_6chn[img_hgh_res_6chn<0.5] = 0  # shape (6, 1, 712, 512, 512)
#
# img_hgh_res_6chn = (img_hgh_res_6chn.argmax(dim=1, keepdims=True)).float()
# I have realized that bilinear/trilinear/cubic upsampling can not lead to smooth edge. So I started to try '
tmp = (img_hgh_res_nearest * 5).astype(np.int)
img_hgh_res_nearest_6chn = one_hot(torch.tensor(tmp), num_classes=6, dim=0)

conv = torch.nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(4, 4, 6), padding='same', bias=False)
conv.weight = torch.nn.Parameter(torch.ones((1, 1, 4, 4, 6)) / 96.0)

im_tensor_ = torch.zeros((img_hgh_res_nearest_6chn.shape))
# im_ls = []
for idx, im in enumerate(img_hgh_res_nearest_6chn):
    im=im[None][None]
    im_conv = conv(im)
    im_tensor_[idx] = im_conv[0][0]
    # im_ls.append(im_conv)

im_tensor = im_tensor_.clone().detach()
td = 0.5
# for im in im_ls:
im_tensor[im_tensor>td] = 1
im_tensor[im_tensor<td] = 0

# img_hgh_res_nearest_6chn_conv = torch.tensor(np.numpy(im_ls))

img_hgh_res_nearest_1chn_conv = (im_tensor.argmax(dim=0, keepdims=False)).float()

slice_idx = 300
plt.figure(figsize=(8,8))
plt.subplot(1,1,1)
plt.imshow(img_hgh_res_nearest_1chn_conv[:, slice_idx, :].detach().numpy())
# plt.title('conv mask')
plt.show()
print('done')
