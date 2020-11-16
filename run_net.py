# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import logging
import os
import shutil
import sys
from shutil import copy2

import numpy as np
import torch
import torch.nn as nn
from ignite.contrib.handlers import ProgressBar
from torch.autograd import Variable
import torch.nn.functional as F
from ignite.engine import Events
# from torch.utils.tensorboard import SummaryWriter
from set_args import args
from monai.data.utils import create_file_basename

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
    RandFlipd,
    RandGaussianNoised,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    ToTensord,
)


train_workers = 10

# Writer will output to ./runs/ directory by default
# writer = SummaryWriter(args.model_folder)

def get_xforms(mode="train", keys=("image", "label")):
    """returns a composed transform for train/val/infer."""

    xforms = [
        LoadNiftid(keys),
        AddChanneld(keys),
        Orientationd(keys, axcodes="LPS"),
        Spacingd(keys, pixdim=(args.space_xy, args.space_xy, args.space_z), mode=("bilinear", "nearest")[: len(keys)]),
        ScaleIntensityRanged(keys[0], a_min=-1000.0, a_max=500.0, b_min=0.0, b_max=1.0, clip=True),
    ]
    if mode == "train":
        xforms.extend(
            [
                SpatialPadd(keys, spatial_size=(args.patch_xy, args.patch_xy, -1), mode="reflect"),  # ensure at least HTxHT
                RandAffined(
                    keys,
                    prob=0.15,
                    rotate_range=(-0.05, 0.05),
                    scale_range=(-0.1, 0.1),
                    mode=("bilinear", "nearest"),
                    as_tensor_output=False,
                ),
                RandCropByPosNegLabeld(keys, label_key=keys[1], spatial_size=(args.patch_xy, args.patch_xy, args.patch_z), num_samples=3),  # todo: num_samples
                RandGaussianNoised(keys[0], prob=0.15, std=0.01),
                RandFlipd(keys, spatial_axis=0, prob=0.5),
                RandFlipd(keys, spatial_axis=1, prob=0.5),
                RandFlipd(keys, spatial_axis=2, prob=0.5),
            ]
        )
        dtype = (np.float32, np.uint8)
    if mode == "val":
        dtype = (np.float32, np.uint8)
    if mode == "infer":
        dtype = (np.float32,)
    xforms.extend([CastToTyped(keys, dtype=dtype), ToTensord(keys)])
    return monai.transforms.Compose(xforms)


def get_net():
    """returns a unet model instance."""

    n_classes = 2
    base = 1
    net = monai.networks.nets.BasicUNet(
        dimensions=3,
        in_channels=1,
        out_channels=n_classes,
        features=(32*base, 32*base, 64*base, 128*base, 256*base, 32*base),  # todo: change features
        dropout=0.1,
    )
    return net


def get_inferer(_mode=None):
    """returns a sliding window inference instance."""

    patch_size = (args.patch_xy, args.patch_xy, args.patch_z)
    sw_batch_size, overlap = 2, 0.5  # todo: change overlap for inferer
    inferer = monai.inferers.SlidingWindowInferer(
        roi_size=patch_size,
        sw_batch_size=sw_batch_size,
        overlap=overlap,
        mode="gaussian",
        padding_mode="replicate",
    )
    return inferer


def one_hot_embedding(labels, num_classes):
    '''Embedding labels to one-hot form.
    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.
    Returns:
      (tensor) encoded labels, sized [N,#classes].
    '''
    y = torch.eye(num_classes)  # [D,D]
    return y[labels]            # [N,D]

def save_args():
    """ Save args files so that we know the specific setting of the model"""
    if not os.path.isdir(args.model_folder):
        os.makedirs(args.model_folder)
    copy2("set_args.py", args.model_folder+"/used_args.py")

class FocalLoss(nn.Module):
    def __init__(self, num_classes=2):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes

    def focal_loss(self, x, y):
        '''Focal loss.
        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].
        Return:
          (tensor) focal loss.
        '''
        alpha = 0.25
        gamma = 2

        t = one_hot_embedding(y.data.cpu(), 1+self.num_classes)  # [N,21]
        t = t[:,1:]  # exclude background
        t = Variable(t).cuda()  # [N,20]

        p = x.sigmoid()
        pt = p*t + (1-p)*(1-t)         # pt = p if t > 0 else 1-p
        w = alpha*t + (1-alpha)*(1-t)  # w = alpha if t > 0 else 1-alpha
        w = w * (1-pt).pow(gamma)
        return F.binary_cross_entropy_with_logits(x, t, w, size_average=False)

class DiceCELoss(nn.Module):
    """Dice and Xentropy loss"""

    def __init__(self):
        super().__init__()
        self.dice = monai.losses.DiceLoss(to_onehot_y=True, softmax=True)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, y_pred, y_true):
        dice = self.dice(y_pred, y_true)
        # CrossEntropyLoss target needs to have shape (B, D, H, W)
        # Target from pipeline has shape (B, 1, D, H, W)
        cross_entropy = self.cross_entropy(y_pred, torch.squeeze(y_true, dim=1).long())
        return dice + cross_entropy

def logfile():
    print("my custom training handler")


def train(data_folder="."):
    """run a training pipeline."""

    images = sorted(glob.glob(os.path.join(data_folder, "*_ct.nii.gz")))
    labels = sorted(glob.glob(os.path.join(data_folder, "*_seg.nii.gz")))

    logging.info(f"training: image/label ({len(images)}) folder: {data_folder}")

    amp = True  # auto. mixed precision
    keys = ("image", "label")
    train_frac, val_frac = 0.8, 0.2
    n_train = int(train_frac * len(images)) + 1
    n_val = min(len(images) - n_train, int(val_frac * len(images)))
    logging.info(f"training: train {n_train} val {n_val}, folder: {data_folder}")

    train_files = [{keys[0]: img, keys[1]: seg} for img, seg in zip(images[:n_train], labels[:n_train])]
    val_files = [{keys[0]: img, keys[1]: seg} for img, seg in zip(images[-n_val:], labels[-n_val:])]

    # create a training data loader
    batch_size = 2
    logging.info(f"batch size {batch_size}")
    train_transforms = get_xforms("train", keys)
    train_ds = monai.data.CacheDataset(data=train_files, transform=train_transforms)
    train_loader = monai.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=train_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # create a validation data loader
    val_transforms = get_xforms("val", keys)
    val_ds = monai.data.CacheDataset(data=val_files, transform=val_transforms)
    val_loader = monai.data.DataLoader(
        val_ds,
        batch_size=1,  # image-level batch to the sliding window method, not the window-level batch
        num_workers=train_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # create BasicUNet, DiceLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = get_net().to(device)
    if args.ld_model:
        ckpt = get_model_path(args.ld_model)
        net.load_state_dict(torch.load(ckpt, map_location=device))
        logging.info("successfully load model: " + ckpt)
    max_epochs, lr, momentum = args.epochs, 1e-4, 0.95
    logging.info(f"epochs {max_epochs}, lr {lr}, momentum {momentum}")
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    # create evaluator (to be used to measure model quality during training
    val_post_transform = monai.transforms.Compose(
        [AsDiscreted(keys=("pred", "label"), argmax=(True, False), to_onehot=True, n_classes=2)]
    )
    val_handlers = [
        ProgressBar(),
        CheckpointSaver(save_dir=args.model_folder, save_dict={"net": net}, save_key_metric=True, key_metric_n_saved=3),
    ]
    evaluator = monai.engines.SupervisedEvaluator(
        device=device,
        val_data_loader=val_loader,
        network=net,
        inferer=get_inferer(),
        post_transform=val_post_transform,
        key_val_metric={
            "val_mean_dice": MeanDice(include_background=False, output_transform=lambda x: (x["pred"], x["label"]))
        },
        val_handlers=val_handlers,
        amp=amp,
    )

    # evaluator as an event handler of the trainer
    train_handlers = [
        ValidationHandler(validator=evaluator, interval=3, epoch_level=True),
        StatsHandler(tag_name="train_loss", output_transform=lambda x: x["loss"]),
    ]
    trainer = monai.engines.SupervisedTrainer(
        device=device,
        max_epochs=max_epochs,
        train_data_loader=train_loader,
        network=net,
        optimizer=opt,
        loss_function=DiceCELoss(),
        inferer=get_inferer(),
        key_train_metric=None,
        train_handlers=train_handlers,
        amp=amp
    )
    # trainer.add_event_handler(Events.ITERATION_COMPLETED, logfile)

    trainer.run()


def infer(data_folder=".", prediction_folder=args.result_folder, write_pbb_maps=False):
    """
    run inference, the output folder will be "./output"
    :param write_pbb_maps: write probabilities maps to the disk for future boosting
    """
    ckpt = get_model_path(args.model_folder)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = get_net().to(device)
    net.load_state_dict(torch.load(ckpt, map_location=device))
    net.eval()

    image_folder = os.path.abspath(data_folder)
    images = sorted(glob.glob(os.path.join(image_folder, "*_ct.nii.gz")))
    logging.info(f"infer: image ({len(images)}) folder: {data_folder}")
    infer_files = [{"image": img} for img in images]

    keys = ("image",)
    infer_transforms = get_xforms("infer", keys)
    infer_ds = monai.data.Dataset(data=infer_files, transform=infer_transforms)
    infer_loader = monai.data.DataLoader(
        infer_ds,
        batch_size=1,  # image-level batch to the sliding window method, not the window-level batch
        num_workers=train_workers,
        pin_memory=torch.cuda.is_available(),
    )

    inferer = get_inferer()
    saver = monai.data.NiftiSaver(output_dir=prediction_folder, mode="nearest")  # todo: change mode
    with torch.no_grad():
        for infer_data in infer_loader:
            logging.info(f"segmenting {infer_data['image_meta_dict']['filename_or_obj']}")
            preds = inferer(infer_data[keys[0]].to(device), net)
            n = 1.0
            for _ in range(4):
                # test time augmentations
                _img = RandGaussianNoised(keys[0], prob=1.0, std=0.01)(infer_data)[keys[0]]
                pred = inferer(_img.to(device), net)
                preds = preds + pred
                n = n + 1.0
                for dims in [[2], [3]]:
                    flip_pred = inferer(torch.flip(_img.to(device), dims=dims), net)
                    pred = torch.flip(flip_pred, dims=dims)
                    preds = preds + pred
                    n = n + 1.0
            preds = preds / n
            if write_pbb_maps:
                # pass
                filename = infer_data["image_meta_dict"]["filename_or_obj"][0]
                pbb_folder = prediction_folder + "/pbb_maps/"
                npy_name = pbb_folder + filename.split("/")[-1].split(".")[0] + ".npy"
                if not os.path.isdir(pbb_folder):
                    os.makedirs(pbb_folder)
                np.save(npy_name, preds.cpu())
            preds = (preds.argmax(dim=1, keepdims=True)).float()
            saver.save_batch(preds, infer_data["image_meta_dict"])

    # copy the saved segmentations into the required folder structure for submission
    submission_dir = os.path.join(prediction_folder, "to_submit")
    if not os.path.exists(submission_dir):
        os.makedirs(submission_dir)
    files = glob.glob(os.path.join(prediction_folder, "volume*", "*.nii.gz"))
    for f in files:
        new_name = os.path.basename(f)
        new_name = new_name[len("volume-covid19-A-0") :]
        new_name = new_name[: -len("_ct_seg.nii.gz")] + ".nii.gz"
        to_name = os.path.join(submission_dir, new_name)
        shutil.copy(f, to_name)
    logging.info(f"predictions copied to {submission_dir}.")


def get_model_path(model_folder):
    ckpts = sorted(glob.glob(os.path.join(model_folder, "*.pt")))
    ckpt = ckpts[-1]
    for x in ckpts:
        logging.info(f"available model file: {x}.")
    logging.info("----")
    logging.info(f"using {ckpt}.")
    return ckpt


if __name__ == "__main__":

    """
    Usage:
        python run_net.py train --data_folder "COVID-19-20_v2/Train" # run the training pipeline
        python run_net.py infer --data_folder "COVID-19-20_v2/Validation" # run the inference pipeline
    """


    monai.config.print_config()
    monai.utils.set_determinism(seed=0)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    if args.mode == "train":
        save_args()
        data_folder = args.data_folder or os.path.join("COVID-19-20_v2", "Train")
        train(data_folder=data_folder)
    elif args.mode == "infer":
        data_folder = args.data_folder or os.path.join("COVID-19-20_v2", "Validation")
        infer(data_folder=data_folder, write_pbb_maps=True)
    else:
        raise ValueError("Unknown mode.")