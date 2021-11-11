# -*- coding: utf-8 -*-
# @Time    : 11/15/20 7:50 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com

import glob
import logging
import os
import shutil

import monai
import numpy as np
import torch
# from monai.handlers import CheckpointSaver, MeanDice, StatsHandler, ValidationHandler
from monai.transforms import (
    AddChanneld,
    CastToTyped,
    LoadImaged,
    Orientationd,
    RandAffined,
    RandCropByPosNegLabeld,
    RandGaussianNoised,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    ToTensord,
)

# from torch.utils.tensorboard import SummaryWriter
from set_args import args


# from get_unetpp import get_unetpp


def get_xforms(mode="train", keys=("image", "label")):
    """returns a composed transform for train/val/infer."""

    xforms = [
        LoadImaged(keys),
        AddChanneld(keys),
        Orientationd(keys, axcodes="LPS"),
        Spacingd(keys, pixdim=(args.space_xy, args.space_xy, args.space_z), mode=("bilinear", "nearest")[: len(keys)]),
        ScaleIntensityRanged(keys[0], a_min=-1000.0, a_max=500.0, b_min=0.0, b_max=1.0, clip=True),
    ]
    if mode == "train":
        xforms.extend(
            [
                SpatialPadd(keys, spatial_size=(args.patch_xy, args.patch_xy, -1), mode="reflect"),
                # ensure at least HTxHT
                RandAffined(
                    keys,
                    prob=0.15,
                    rotate_range=(-0.05, 0.05),
                    scale_range=(-0.1, 0.1),
                    mode=("bilinear", "nearest"),
                    as_tensor_output=False,
                ),
                RandCropByPosNegLabeld(keys, label_key=keys[1],
                                       spatial_size=(args.patch_xy, args.patch_xy, args.patch_z), num_samples=5),
                # todo: num_samples
                # RandGaussianNoised(keys[0], prob=0.15, std=0.01),
                # RandFlipd(keys, spatial_axis=0, prob=0.5),
                # RandFlipd(keys, spatial_axis=1, prob=0.5),
                # RandFlipd(keys, spatial_axis=2, prob=0.5),
            ]
        )
        dtype = (np.float32, np.uint8)
    if mode == "val":
        dtype = (np.float32, np.uint8)
    if mode == "infer":
        dtype = (np.float32,)
    xforms.extend([CastToTyped(keys, dtype=dtype), ToTensord(keys)])
    return monai.transforms.Compose(xforms)


def get_infer_loader():
    data_folder = ('/data/jjia/monai/COVID-19-20_v2/Validation')
    images = sorted(glob.glob(os.path.join(data_folder, "*_ct.nii.gz")))
    logging.info(f"infer: image ({len(images)}) folder: {data_folder}")
    infer_files = [{"image": img} for img in images]

    keys = ("image",)
    infer_transforms = get_xforms("infer", keys)
    infer_ds = monai.data.Dataset(data=infer_files, transform=infer_transforms)
    infer_loader = monai.data.DataLoader(
        infer_ds,
        batch_size=1,  # image-level batch to the sliding window method, not the window-level batch
        num_workers=6,
        pin_memory=torch.cuda.is_available(),
    )

    return infer_loader


def get_inferer():
    """returns a sliding window inference instance."""

    patch_size = (256, 256, 16)
    sw_batch_size, overlap = args.batch_size, 0.5  # todo: change overlap for inferer
    inferer = monai.inferers.SlidingWindowInferer(
        roi_size=patch_size,
        sw_batch_size=sw_batch_size,
        overlap=overlap,
        mode="gaussian",
        padding_mode="replicate",
    )
    return inferer


dest_folder = "results/boosting/all"
if not os.path.isdir(dest_folder):
    os.makedirs(dest_folder)
saver = monai.data.NiftiSaver(output_dir=dest_folder, mode="nearest")  # todo: change mode

pbb_folder_base = "results/ex"
pbb_folders = [pbb_folder_base + str(i) + "/pbb_maps" for i in [10, 11, 12, 13, 14]]

pbb_folders = [
    "results/ex2_0/pbb_maps",
    # "results/ex3/pbb_maps",
    "results/ex3_0/pbb_maps",
    "results/ex4/pbb_maps",
    "results/ex10/pbb_maps",
    "results/ex11/pbb_maps",
    "results/ex12/pbb_maps",
    "results/ex13/pbb_maps",
    "results/ex14/pbb_maps",
    "models/lesion/1606762775_273/infer_pred/pbb_maps",
    "models/lesion/1606762984_399/infer_pred/pbb_maps",
    "models/lesion/1606769290_556/infer_pred/pbb_maps",
]

data_folder = "COVID-19-20_v2/Validation"
images = sorted(glob.glob(os.path.join(data_folder, "*_ct.nii.gz")))
images_names = [image.split("/")[-1] for image in images]
images_name_wo_ex = [image_name.split(".")[0] for image_name in images_names]
pbb_npy_names = [img + ".npy" for img in images_name_wo_ex]

keys = ("image",)
infer_files = [{"image": img} for img in images]
infer_transforms = get_xforms("infer", keys)
infer_ds = monai.data.Dataset(data=infer_files, transform=infer_transforms)
infer_loader = monai.data.DataLoader(
    infer_ds,
    batch_size=1,  # image-level batch to the sliding window method, not the window-level batch
    num_workers=4,
    pin_memory=torch.cuda.is_available(),
)

inferer = get_inferer()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for npy_name, img_name, infer_data in zip(pbb_npy_names, images_names, infer_loader):
    pbb_list = []
    for pbb_folder in pbb_folders:
        pbb_npy_path = pbb_folder + "/" + npy_name
        pbb = np.load(pbb_npy_path)
        pbb_list.append(pbb)
    pbb_ave = sum(pbb_list) / len(pbb_folder)
    pbb_ave = torch.from_numpy(pbb_ave)
    preds = (pbb_ave.argmax(dim=1, keepdims=True)).float()

    saver.save_batch(preds, infer_data["image_meta_dict"])
    print(f"save {img_name} successfully")

# copy the saved segmentations into the required folder structure for submission
submission_dir = os.path.join(dest_folder, "to_submit")
if not os.path.exists(submission_dir):
    os.makedirs(submission_dir)
files = glob.glob(os.path.join(dest_folder, "volume*", "*.nii.gz"))
for f in files:
    new_name = os.path.basename(f)
    new_name = new_name[len("volume-covid19-A-0"):]
    new_name = new_name[: -len("_ct_seg.nii.gz")] + ".nii.gz"
    to_name = os.path.join(submission_dir, new_name)
    shutil.copy(f, to_name)
logging.info(f"predictions copied to {submission_dir}.")
