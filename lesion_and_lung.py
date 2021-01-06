# -*- coding: utf-8 -*-
# @Time    : 12/10/20 12:39 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import jjnutils.util as cu


import glob
import logging
import os
import shutil
import sys
from shutil import copy2
from typing import Type
from shutil import copy2
import time
import random

import numpy as np
import torch
import torch.nn as nn
from ignite.contrib.handlers import ProgressBar
from torch.autograd import Variable
import torch.nn.functional as F
from ignite.engine import Events
# from torch.utils.tensorboard import SummaryWriter
from monai.data.utils import create_file_basename
from statistics import mean

import monai
# from monai.handlers import CheckpointSaver, MeanDice, StatsHandler, ValidationHandler
from monai.handlers import StatsHandler, MeanDice, ValidationHandler
from CheckpointSaver import CheckpointSaver
from monai.transforms import (
    AddChanneld,
    AsDiscreted,
    CastToTyped,
    LoadNiftid,
    Orientationd,
    RandAffined,
    RandCropByPosNegLabeld,
    RandSpatialCropd,
    RandFlipd,
    RandGaussianNoised,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    ToTensord,
)
from typing import Dict

if __name__ == '__main__':
    from mypath import Mypath
    from custom_net import Encoder, DecoderRec, DecoderSeg, EnsembleEncRec, EnsembleEncSeg, DecoderSegRec
    from taskargs import CommonTask
    from set_args_mtnet import args
    from unet_att_dsv import unet_CT_single_att_dsv_3D
    from saharnet import Saharnet_decoder, Saharnet_encoder
    from Generic_UNetPlusPlus import Generic_UNetPlusPlus

else:
    from .mypath import Mypath
    from .custom_net import Encoder, DecoderRec, DecoderSeg, EnsembleEncRec, EnsembleEncSeg, DecoderSegRec
    from .taskargs import CommonTask
    from .set_args_mtnet import args
    from .unet_att_dsv import unet_CT_single_att_dsv_3D
    from .saharnet import Saharnet_decoder, Saharnet_encoder
    from .Generic_UNetPlusPlus import Generic_UNetPlusPlus


lesion_dir = "/data/jjia/monai/models/lesion/1606762984_399/infer_pred/COVID-19-20_TestSet/to_submit_multitask_deepsupervision"
lesion_names = cu.get_all_ct_names(lesion_dir)
lung_dir = "/data/jjia/monai/models/body_masks"
    # "/data/jjia/monai/models/lung/1607203020_836/infer_pred/COVID-19-20_TestSet/to_submit_testset_lung/biggest_parts"
lung_names = cu.get_all_ct_names(lung_dir)

for lesion_name, lung_name in zip(lesion_names, lung_names):
    print(f"lesion name: {lesion_name}")
    print(f"lung name: {lung_name}")
    lesion, origin, spacing = cu.load_itk(lesion_name)
    lung, _, _ = cu.load_itk(lung_name)
    new_img = lesion * lung

    write_fpath = "/data/jjia/monai/lesion_filtered_body/" + lesion_name.split("/")[-1]
    cu.save_itk(write_fpath, new_img, origin, spacing)
    print(f"save successfully at: {write_fpath}")


