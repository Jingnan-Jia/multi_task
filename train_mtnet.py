# -*- coding: utf-8 -*-
# @Time    : 11/20/20 11:58 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com

import glob
import logging
import os
import shutil
import sys
from shutil import copy2
from typing import Type
from shutil import copy2
import time

import numpy as np
import torch
import torch.nn as nn
from ignite.contrib.handlers import ProgressBar
from torch.autograd import Variable
import torch.nn.functional as F
from ignite.engine import Events
# from torch.utils.tensorboard import SummaryWriter
from monai.data.utils import create_file_basename
from get_unetpp import get_unetpp

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
from typing import Dict

if __name__ == '__main__':
    from mypath import Mypath
    from custom_net import Encoder, DecoderRec, DecoderSeg, EnsembleEncRec, EnsembleEncSeg, DecoderSegRec
    from taskargs import CommonTask
    from set_args_mtnet import args
    from unet_att_dsv import unet_CT_single_att_dsv_3D
else:
    from .mypath import Mypath
    from .custom_net import Encoder, DecoderRec, DecoderSeg, EnsembleEncRec, EnsembleEncSeg, DecoderSegRec
    from .taskargs import CommonTask
    from .set_args_mtnet import args
    from .unet_att_dsv import unet_CT_single_att_dsv_3D

import jjnutils.util as cu
from typing import (Dict, List, Tuple, Set, Deque, NamedTuple, IO, Pattern, Match, Text,
                    Optional, Sequence, Union, TypeVar, Iterable, Mapping, MutableMapping, Any)
import csv


class DiceCELoss(nn.Module):
    """Dice and Xentropy loss"""

    def __init__(self):
        super().__init__()
        self.dice = monai.losses.DiceLoss(include_background=False, to_onehot_y=True, softmax=True)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, y_pred, y_true):
        dice = self.dice(y_pred, y_true)
        # CrossEntropyLoss target needs to have shape (B, D, H, W)
        # Target from pipeline has shape (B, 1, D, H, W)
        cross_entropy = self.cross_entropy(y_pred, torch.squeeze(y_true, dim=1).long())
        print(f"dice loss: {dice}, CE loss: {cross_entropy}")
        return dice + cross_entropy


def get_mtnet(netname_label_dict: Dict[str, List], netname_ds_dict: Dict[str, int], base: int = 1) -> Dict[
    str, nn.Module]:
    nets = {}
    enc = Encoder(features=(32 * base, 32 * base, 64 * base, 128 * base, 256 * base, 32 * base), dropout=0.1)
    for net_name, label in netname_label_dict.items():
        if net_name == "net_lesion_att_unet":
            net = unet_CT_single_att_dsv_3D(
                in_channels=1,
                n_classes=len(label),
                base=base
            )
        elif net_name == "net_recon":  # reconstruction
            dec = DecoderRec(features=(32 * base, 32 * base, 64 * base, 128 * base, 256 * base, 32 * base),
                             dropout=0.1)
            net = EnsembleEncRec(enc, dec)
        else:  # segmentation output different channels
            if "itgt" in net_name:  # 2 outputs: pred and rec_loss
                print("itgt net")
                if "vae" in net_name or "VAE" in net_name:  # vae as the reconnet
                    print("vae net")
                    net = monai.networks.nets.SegResNetVAE(
                        input_image_size=(args.patch_xy, args.patch_xy, args.patch_z),
                        spatial_dims=3,
                        init_filters=32 * args.base,  # todo: could change to base
                        in_channels=1,
                        out_channels=len(label),
                        dropout_prob=0.1
                    )
                elif "resnet" in net_name:
                    
                    net = monai.networks.nets.SegResNet(
                        init_filters=32 * args.base,  # todo: could change to base
                        out_channels=len(label),
                        dropout_prob=0.1
                    )
                else:
                    dec = DecoderSegRec(out_channels=len(label),
                                        features=(32 * base, 32 * base, 64 * base, 128 * base, 256 * base, 32 * base),
                                        dropout=0.1, )
                    net = EnsembleEncSeg(enc, dec)

                    # pass
                    # net = ItgtSegRec()  # todo: Seg_net and Rec_net are iteragrated
            else:
                dec = DecoderSeg(out_channels=len(label),
                                 features=(32 * base, 32 * base, 64 * base, 128 * base, 256 * base, 32 * base),
                                 dropout=0.1,
                                 ds=netname_ds_dict[net_name])
                net = EnsembleEncSeg(enc, dec)
        # net = get_net()

        nets[net_name] = net

    return nets


# class Diceloss(torch.nn.Module):
#     def init(self):
#         super().init()
#
#     def forward(self, pred, target):
#         if pred.shape != target.shape:  # do one_hot_encoding
#
#         smooth = 1.
#         iflat = pred.contiguous().view(-1)
#         tflat = target.contiguous().view(-1)
#         intersection = (iflat * tflat).sum()
#         A_sum = torch.sum(iflat * iflat)
#         B_sum = torch.sum(tflat * tflat)
#         return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth))

def get_net():
    """returns a unet model instance."""
    n_classes = 2
    base = 1
    # net = monai.networks.nets.SegResNetVAE(
    #     input_image_size=(args.patch_xy, args.patch_xy, args.patch_z),
    #     spatial_dims=3,
    #     init_filters=32,     # todo: could change to base
    #     in_channels=1,
    #     out_channels=n_classes,
    #     dropout_prob=None
    # )

    net = monai.networks.nets.BasicUNet(
        dimensions=3,
        in_channels=1,
        out_channels=n_classes,
        features=(32 * base, 32 * base, 64 * base, 128 * base, 256 * base, 32 * base),  # todo: change features
        dropout=0.1,
    )

    return net


def get_net_names(myargs) -> List[str]:
    # Define the Model, use dash to separate multi net names, do not use ',' to separate it,
    #  because ',' can lead to unknown error during parse arguments
    net_names = myargs.net_names.split('-')
    net_names = [i.lstrip() for i in net_names]  # remove backspace before each net name
    print('net names: ', net_names)

    return net_names


def get_inferer():
    """returns a sliding window inference instance."""

    patch_size = (args.patch_xy, args.patch_xy, args.patch_z)
    sw_batch_size, overlap = args.batch_size, 0.5  # todo: change overlap for inferer
    inferer = monai.inferers.SlidingWindowInferer(
        roi_size=patch_size,
        sw_batch_size=sw_batch_size,
        overlap=overlap,
        mode="gaussian",
        padding_mode="replicate",
    )
    return inferer


def get_netname_label_dict(net_names: List[str]) -> Dict[str, List]:
    netname_label_dict = {}
    for net_name in net_names:
        if "net_recon" in net_name:
            label = [1]
        elif "net_lobe" in net_name:
            label = [0, 1, 2, 3, 4, 5]
        else:
            label = [0, 1]
        netname_label_dict[net_name] = label
    return netname_label_dict


def get_netname_ds_dict(net_names: List[str]) -> Dict[str, int]:
    netname_ds_dict = {}
    for net_name in net_names:
        if net_name == "net_recon":
            ds = args.ds_rc
        elif net_name == "net_lobe":
            ds = args.ds_lb
        elif net_name == "net_lesion":
            ds = args.ds_ls
        elif net_name == "net_vessel":
            ds = args.ds_vs
        elif net_name == "net_airway":
            ds = args.ds_aw
        elif net_name == "net_lung":
            ds = args.ds_lu
        else:
            print(f"net {net_name} can not accept ds, so ds = 0")
            ds = 0
        netname_ds_dict[net_name] = ds
    return netname_ds_dict


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
        self.num_classes = 2

    def forward(self, inputs, targets):
        inputs = one_hot_embedding(inputs.data.cpu(), self.num_classes)
        inputs = Variable(inputs).cuda()  # [N,20]

        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


def get_loss(task: str) -> Type[nn.Module]:
    loss_fun: Type[nn.Module]
    if task == "recon":
        loss_fun = nn.MSELoss
    else:
        loss_fun = DiceCELoss  # or FocalLoss
    return loss_fun


def inifite_generator(dataloader):
    keys = ("image", "label")
    while True:
        for data in dataloader:
            x_pps = data[keys[0]]
            y_pps = data[keys[1]]
            for x, y in zip(x_pps, y_pps):
                x = x[None, ...]
                y = y[None, ...]
                yield x, y


class TaskArgs(CommonTask):
    def __init__(self,
                 task: str,
                 labels: List[int],
                 net_name: str,
                 ld_name: Optional[str],
                 tr_nb: Union[int],
                 ds: int,
                 tsp: str,
                 lr: float,
                 main_net_name: str,
                 all_nets: Dict[str, nn.Module],
                 sub_dir: str
                 ):
        super().__init__()  # set self.device and self.amp
        self.task: str = task
        self.labels: List[int] = labels
        self.net_name: str = net_name
        self.net: nn.Module = all_nets[net_name]
        self.net.to(self.device)
        self.ld_name: str = ld_name
        self.tr_nb: Union[int] = tr_nb
        self.ds: int = ds
        self.tsp_xy = float(tsp.split("_")[0])
        self.tsp_z = float(tsp.split("_")[1])
        self.sub_dir = sub_dir
        self.load_workers = 6

        self.lr: float = lr
        self.main_net_name: str = main_net_name
        self.main_net = all_nets[self.main_net_name]
        self.keys: Tuple[str, str] = ("pred", "label")

        self.mypath = Mypath(task)
        self.ld_name = ld_name  # for fine-tuning and inference

        if ld_name:  # fine-tuning
            self.trained_model_folder = self.mypath.task_model_dir(current_time=self.ld_name)
            ckpt = get_model_path(self.trained_model_folder)
            self.net.load_state_dict(torch.load(ckpt, map_location=self.device))

        self.n_classes = len(self.labels)

        # if args.mode=="train":
        self.tra_loader, self.val_loader = self.dataloader()
        self.tra_gen = inifite_generator(self.tra_loader)
        self.val_gen = inifite_generator(self.val_loader)

        self.loss_fun = get_loss(task)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.criteria = self.loss_fun()
        self.accumulate_loss: float = 0.0
        self.current_loss: float = 100.0
        self.steps_per_epoch = self.n_train * args.pps

        self.evaluator = self.get_evaluator()
        if args.mode == "infer" or "semibyaug" in self.net_name:
            self.infer_loader = self.get_infer_loader()
        copy2("set_args_mtnet.py", self.mypath.args_fpath())  # save super parameters

    def get_xforms(self, mode: str = "train", keys=("image", "label")):
        """returns a composed transform for train/val/infer."""
        xforms = [
            LoadNiftid(keys),  # .nii, .nii.gz [.mhd, .mha LoadImage]
            AddChanneld(keys),
            Orientationd(keys, axcodes="LPS"),
            Spacingd(keys, pixdim=(self.tsp_xy, self.tsp_xy, self.tsp_z), mode=("bilinear", "nearest")[: len(keys)]),
            ScaleIntensityRanged(keys[0], a_min=-1000.0, a_max=500.0, b_min=0.0, b_max=1.0, clip=True),
        ]
        if mode == "train":
            xforms.extend(
                [
                    SpatialPadd(keys, spatial_size=(args.patch_xy, args.patch_xy, args.patch_z), mode="reflect"),
                    # ensure at least HTxHT*z
                    RandAffined(
                        keys,
                        prob=0.3,
                        rotate_range=(-0.05, 0.05),
                        scale_range=(-0.1, 0.1),
                        mode=("bilinear", "nearest"),
                        as_tensor_output=False,
                    ),
                    RandCropByPosNegLabeld(keys, label_key=keys[1],
                                           spatial_size=(args.patch_xy, args.patch_xy, args.patch_z),
                                           num_samples=args.pps),
                    # todo: num_samples
                    RandGaussianNoised(keys[0], prob=0.3, std=0.01),
                    # RandFlipd(keys, spatial_axis=0, prob=0.5),
                    # RandFlipd(keys, spatial_axis=1, prob=0.5),
                    # RandFlipd(keys, spatial_axis=2, prob=0.5),
                ]
            )
            dtype = (np.float32, np.uint8)
        elif mode == "val":
            dtype = (np.float32, np.uint8)
        elif mode == "infer":
            dtype = (np.float32,)
        else:
            raise Exception(f"mode {mode} is not correct, please set mode as 'train', 'val' or 'infer'. ")
        xforms.extend([CastToTyped(keys, dtype=dtype), ToTensord(keys)])
        return monai.transforms.Compose(xforms)

    # def rand_affine(self, x, y):
    #     keys = ("image", "label")
    #     dtype = (np.float32, np.uint8)
    #     xforms = [RandAffined(keys, prob=0.15, rotate_range=(-0.05, 0.05), scale_range=(-0.1, 0.1),
    #                           mode=("bilinear", "nearest"), as_tensor_output=False),
    #               CastToTyped(keys, dtype=dtype),
    #               ToTensord(keys)
    #               ]

    def get_file_names(self):
        keys = ("image", "label")
        # data_dir = self.mypath.data_dir()
        data_dir = "/data/jjia/monai/COVID-19-20_v2/Train"
        print(data_dir)
        ct_names: List[str] = cu.get_all_ct_names(data_dir, name_suffix="_ct")
        if self.task != "recon":
            gdth_names: List[str] = cu.get_all_ct_names(data_dir, name_suffix="_seg")
        else:
            gdth_names = ct_names
        train_frac, val_frac = 0.8, 0.2
        if self.tr_nb != 0:
            total_nb = min(self.tr_nb, len(ct_names))  # if set tr_nb
        else:
            total_nb = len(ct_names)
        self.n_train: int = int(train_frac * total_nb) + 1  # avoid empty train data
        if self.net_name != self.main_net_name:
            self.n_val: int = 2
        else:
            self.n_val: int = min(total_nb - self.n_train, int(val_frac * total_nb))
            # self.n_val: int = 5

        logging.info(f"In task {self.task}, training: train {self.n_train} val {self.n_val}")

        train_files = [{keys[0]: img, keys[1]: seg} for img, seg in
                       zip(ct_names[:self.n_train], gdth_names[:self.n_train])]
        val_files = [{keys[0]: img, keys[1]: seg} for img, seg in zip(ct_names[-self.n_val:], gdth_names[-self.n_val:])]
        return train_files, val_files

    def dataloader(self):
        ct_name_list: List[str]
        gdth_name_list: List[str]
        train_files, val_files = self.get_file_names()

        train_transforms = self.get_xforms("train")
        if args.smartcache or self.n_train > 200:
            if args.smartcache:
                self.cache_num = min(args.smartcache, self.n_train - 1)
            else:
                self.cache_num = 40
            # cache_num must be smaller than dataset length to support replacement.
            self.train_ds = monai.data.SmartCacheDataset(data=train_files,
                                                         transform=train_transforms,
                                                         replace_rate=0.1,
                                                         cache_num=self.cache_num,
                                                         num_init_workers=5,
                                                         num_replace_workers=self.load_workers
                                                         )
        else:
            self.train_ds = monai.data.CacheDataset(data=train_files, transform=train_transforms,
                                                    num_workers=self.load_workers, cache_rate=1)

        train_loader = monai.data.DataLoader(
            self.train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=self.load_workers,
            pin_memory=torch.cuda.is_available(),
        )
        if args.smartcache or self.n_train > 200:
            self.train_ds.start()  # need it if SmartCacheDataset

        # create a validation data loader
        val_transforms = self.get_xforms("val")
        val_ds = monai.data.CacheDataset(data=val_files, transform=val_transforms, num_workers=self.load_workers)
        val_loader = monai.data.DataLoader(
            val_ds,
            batch_size=1,  # image-level batch to the sliding window method, not the window-level batch
            num_workers=self.load_workers,
            pin_memory=torch.cuda.is_available(),
        )

        return train_loader, val_loader

    def get_evaluator(self):
        keys = ("pred", "label")

        val_post_transform = monai.transforms.Compose(
            [AsDiscreted(keys=keys, argmax=(True, False), to_onehot=True, n_classes=self.n_classes)]
        )
        val_handlers = [
            ProgressBar(),
            CheckpointSaver(save_dir=self.mypath.task_model_dir(),
                            save_dict={"net": self.net},
                            save_key_metric=True,
                            key_metric_n_saved=3),
        ]
        evaluator = monai.engines.SupervisedEvaluator(
            device=self.device,
            val_data_loader=self.val_loader,
            network=self.net,
            inferer=get_inferer(),
            post_transform=val_post_transform,
            key_val_metric={
                "val_mean_dice": MeanDice(include_background=False,
                                          output_transform=lambda x: (x[keys[0]], x[keys[1]]))
            },
            val_handlers=val_handlers,
            amp=self.amp,
        )

        return evaluator

    def run_one_step(self, net_ta_dict, idx: int):
        t1 = time.time()
        x, y = next(self.tra_gen)
        t3 = time.time()
        print(f"load data cost time: {t3 - t1}")
        if self.task == "recon":
            y = x
        x = x.to(self.device)
        y = y.to(self.device)
        # print(x.shape)
        with torch.cuda.amp.autocast():
            self.net.train()
            if "semibyaug" in self.net_name:
                x = next(self.infer_loader)
                y = self.net(x)  # now we have x and its gdth: y
                x = x.to("cpu")
                y = y.to("cpu")
                x_Affined, y_Affined = cu.random_transform(x, y,
                                                           rotation_range=0.1,
                                                           height_shift_range=0.1,
                                                           width_shift_range=0.1,
                                                           shear_range=0.1,
                                                           fill_mode='constant',
                                                           zoom_range=0.2,
                                                           prob=1)
                if random.random() > 0.5:
                    noise = np.random.normal(0, 0.25, x_Affined.shape)
                    x_Affined += noise
                x_Affined = x_Affined.to(self.device)
                y_Affined = y_Affined.to(self.device)
                pred = self.net(x_Affined)
                loss = self.criteria(pred, y_Affined)

            elif "itgt" in self.net_name:
                print("self.netname", self.net_name)
                if "vae" in self.net_name or "VAE" in self.net_name:
                    pred, rec_loss = self.net(x)
                    loss = self.criteria(pred, y) + rec_loss
                
                else:
                    pred, pred_rec = self.net(x)
                    loss = self.criteria(pred, y) + F.mse_loss(pred_rec, x)
            else:
                if self.ds:
                    pred, pred1, pred2 = self.net(x)
                    loss = self.criteria(pred, y)
                    loss1 = self.criteria(pred1, y)
                    loss2 = self.criteria(pred2, y)
                    print(f"loss: {loss}, loss1: {loss1},loss2: {loss2},")
                    loss = 0.5 * loss + 0.25 * loss1 + 0.25 * loss2
                else:
                    pred = self.net(x)
                    loss = self.criteria(pred, y)

        self.opt.zero_grad()
        self.current_loss = loss.item()
        t8 = time.time()
        loss.backward()
        self.opt.step()
        t2 = time.time()
        print(f"load data cost time {t3 - t1}, one step backward training cost time: {t2 - t8}")
        if args.ad_lr and self.main_net_name != self.net_name:  # reset lr for aux nets
            lr = net_ta_dict[self.main_net_name].current_loss / self.current_loss * self.lr
            self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
            print(f"task: {self.task}, lr: {lr}")
        print(f"task: {self.task}, loss: {loss.item()}")
        if (args.smartcache or self.n_train > 200) and (idx % (self.cache_num * args.pps)) == 0:
            print(f"start update cache for task {self.task}")
            self.train_ds.update_cache()
        if (args.smartcache or self.n_train > 200) and idx == args.step_nb - 1:
            train_ds.shutdown()

        # print statistics
        self.accumulate_loss += loss.item()
        if idx % self.steps_per_epoch == 0:  # print every 2000 mini-batches
            ave_tr_loss = self.accumulate_loss / self.steps_per_epoch
            print(f'step: {idx} average training loss: {ave_tr_loss}')
            if not os.path.isfile(self.mypath.train_log_fpath()):
                with open(self.mypath.train_log_fpath(), 'a') as csv_file:
                    writer = csv.writer(csv_file, delimiter=',')
                    writer.writerow(["step ", "ave_tr_loss"])
            with open(self.mypath.train_log_fpath(), 'a') as csv_file:
                writer = csv.writer(csv_file, delimiter=',')
                writer.writerow([idx, ave_tr_loss])
            self.accumulate_loss = 0.0

    def run_all_epochs(self):
        # evaluator as an event handler of the trainer
        # epochs = [int(args.epochs * 0.8), int(args.epochs * 0.2)]
        epochs = [400, 100]
        intervals = [1, 1]
        lrs = [1e-3, 1e-4]
        momentum = 0.95
        for epoch, interval, lr in zip(epochs, intervals, lrs):  # big interval to save time, then small interval refine
            logging.info(f"epochs {epoch}, lr {lr}, momentum {momentum}, interval {interval}")
            opt = torch.optim.Adam(self.net.parameters(), lr=lr)
            train_handlers = [
                ValidationHandler(validator=self.evaluator, interval=interval, epoch_level=True),
                StatsHandler(tag_name="train_loss", output_transform=lambda x: x["loss"]),
            ]
            trainer = monai.engines.SupervisedTrainer(
                device=self.device,
                max_epochs=epoch,
                train_data_loader=self.tra_loader,
                network=self.net,
                optimizer=opt,
                loss_function=DiceCELoss(),
                inferer=get_inferer(),
                key_train_metric=None,
                train_handlers=train_handlers,
                amp=self.amp
            )

            # trainer.add_event_handler(Events.ITERATION_COMPLETED, logfile)

            trainer.run()

    def do_vilidation_if_need(self, net_ta_dict, idx_: int):
        if idx_ < int(args.step_nb * 0.8):
            valid_period = args.valid_period1 * net_ta_dict[self.main_net_name].steps_per_epoch
        else:
            valid_period = args.valid_period2 * net_ta_dict[self.main_net_name].steps_per_epoch

        if idx_ % valid_period == 0:
            print("start do validation")
            if self.net_name != "net_recon":
                self.evaluator.run()

    def get_infer_loader(self):
        data_folder = args.infer_data_dir

        images = sorted(glob.glob(os.path.join(data_folder, "*_ct.nii.gz")))
        logging.info(f"infer: image ({len(images)}) folder: {data_folder}")
        infer_files = [{"image": img} for img in images]

        keys = ("image",)
        infer_transforms = self.get_xforms("infer", keys)
        infer_ds = monai.data.Dataset(data=infer_files, transform=infer_transforms)
        infer_loader = monai.data.DataLoader(
            infer_ds,
            batch_size=1,  # image-level batch to the sliding window method, not the window-level batch
            num_workers=self.load_workers,
            pin_memory=torch.cuda.is_available(),
        )

        return infer_loader

    def infer(self, write_pbb_maps=True):
        """
        run inference, the output folder will be "./output"
        :param write_pbb_maps: write probabilities maps to the disk for future boosting
        """
        keys = ("image",)
        self.net.eval()
        prediction_folder = self.mypath.infer_pred_dir(self.ld_name)
        inferer = get_inferer()
        saver = monai.data.NiftiSaver(output_dir=prediction_folder, mode="nearest")  # todo: change mode
        with torch.no_grad():
            for infer_data in self.infer_loader:
                logging.info(f"segmenting {infer_data['image_meta_dict']['filename_or_obj']}")
                preds = inferer(infer_data[keys[0]].to(self.device), self.net)
                n = 1.0
                for _ in range(4):
                    # test time augmentations
                    _img = RandGaussianNoised(keys[0], prob=1.0, std=0.01)(infer_data)[keys[0]]
                    pred = inferer(_img.to(self.device), self.net)
                    preds = preds + pred
                    n = n + 1.0
                    for dims in [[2], [3]]:
                        flip_pred = inferer(torch.flip(_img.to(self.device), dims=dims), self.net)
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
            new_name = new_name[len("volume-covid19-A-0"):]
            new_name = new_name[: -len("_ct_seg.nii.gz")] + ".nii.gz"
            to_name = os.path.join(submission_dir, new_name)
            shutil.copy(f, to_name)
        logging.info(f"predictions copied to {submission_dir}.")
        if "lung" in self.net_name:
            import find_connect_parts as ff
            ff.write_connected_lobes(preds_dir, workers=5, target_dir=preds_dir + "/biggest_5")


def get_netname_ta_dict(netname_label_dict: Dict[str, List],
                        all_nets: Dict[str, nn.Module]) -> Dict[str, TaskArgs]:
    ta_dict: Dict[str, TaskArgs] = {}
    for net_name, label in netname_label_dict.items():
        if "net_lesion" in net_name:
            ta = TaskArgs(
                task="lesion",
                labels=label,
                net_name=net_name,
                ld_name=args.ld_ls,
                tr_nb=args.tr_nb_ls,
                ds=args.ds_ls,
                tsp=args.tsp_ls,
                lr=args.lr_ls,
                main_net_name=args.main_net_name,
                all_nets=all_nets,
                sub_dir=args.sub_dir_ls
            )
        elif "net_recon" in net_name:
            ta = TaskArgs(
                task="recon",
                labels=label,
                net_name=net_name,
                ld_name=args.ld_rc,
                tr_nb=args.tr_nb_rc,
                ds=args.ds_rc,
                tsp=args.tsp_rc,
                lr=args.lr_rc,
                main_net_name=args.main_net_name,
                all_nets=all_nets,
                sub_dir=args.sub_dir_rc
            )
        elif "net_vessel" in net_name:
            ta = TaskArgs(
                task="vessel",
                labels=label,
                net_name=net_name,
                ld_name=args.ld_vs,
                tr_nb=args.tr_nb_vs,
                ds=args.ds_vs,
                tsp=args.tsp_vs,
                lr=args.lr_vs,
                main_net_name=args.main_net_name,
                all_nets=all_nets,
                sub_dir=args.sub_dir_vs
            )
        elif "net_lobe" in net_name:
            ta = TaskArgs(
                task="lobe",
                labels=label,
                net_name=net_name,
                ld_name=args.ld_lb,
                tr_nb=args.tr_nb_lb,
                ds=args.ds_lb,
                tsp=args.tsp_lb,
                lr=args.lr_lb,
                main_net_name=args.main_net_name,
                all_nets=all_nets,
                sub_dir=args.sub_dir_lb
            )
        elif "net_lung" in net_name:
            ta = TaskArgs(
                task="lung",
                labels=label,
                net_name=net_name,
                ld_name=args.ld_lu,
                tr_nb=args.tr_nb_lu,
                ds=args.ds_lu,
                tsp=args.tsp_lu,
                lr=args.lr_lu,
                main_net_name=args.main_net_name,
                all_nets=all_nets,
                sub_dir=args.sub_dir_lu
            )
        elif "net_airway" in net_name:
            ta = TaskArgs(
                task="airway",
                labels=label,
                net_name=net_name,
                ld_name=args.ld_aw,
                tr_nb=args.tr_nb_aw,
                ds=args.ds_aw,
                tsp=args.tsp_aw,
                lr=args.lr_aw,
                main_net_name=args.main_net_name,
                all_nets=all_nets,
                sub_dir=args.sub_dir_aw
            )
        else:
            raise Exception(f"net name {net_name} is not correct")
        ta_dict[net_name] = ta

    return ta_dict


def get_net_ta_dict(net_names: List[str], args) -> Dict[str, TaskArgs]:
    netname_label_dict: Dict[str, List] = get_netname_label_dict(net_names)
    netname_ds_dict: Dict[str, int] = get_netname_ds_dict(net_names)
    all_nets: Dict[str, nn.Module] = get_mtnet(netname_label_dict, netname_ds_dict, args.base)
    ta_dict: Dict[str, TaskArgs] = get_netname_ta_dict(netname_label_dict, all_nets)

    return ta_dict


def get_fat_ta_list(net_ta_dict: Dict, main_name, idx_):
    """
    for each epoch, only return [main_net, one_aux_net]
    """
    net_names, ta_list = list(net_ta_dict.keys()), list(net_ta_dict.values())
    if len(net_names) == 1:  # only one net
        return ta_list
    elif len(net_names) > 1:

        nb_tasks = len(net_names)
        main_index = net_names.index(main_name)
        main_ta = ta_list[main_index]

        aux_index = list(range(nb_tasks))
        aux_index.remove(main_index)
        aux_ta_list = [ta_list[i] for i in aux_index]

        one_idx = idx_ % (nb_tasks - 1)
        aux_ta = aux_ta_list[one_idx]
        return [main_ta, aux_ta]
    else:
        raise Exception("net names are empty")


def get_tr_ta_list(net_ta_dict, idx_):
    if args.fat:
        tr_ta_list: List[TaskArgs] = get_fat_ta_list(net_ta_dict, args.main_net_name, idx_)
    else:
        tr_ta_list: List[TaskArgs] = net_ta_dict
    return tr_ta_list


def get_model_path(model_folder):
    ckpts = sorted(glob.glob(os.path.join(model_folder, "*.pt")))
    ckpt = ckpts[-1]
    for x in ckpts:
        logging.info(f"available model file: {x}.")
    logging.info("----")
    logging.info(f"using {ckpt}.")
    return ckpt


def train_mtnet():
    net_names: List[str] = get_net_names(args)
    net_ta_dict: Dict[str, TaskArgs] = get_net_ta_dict(net_names, args)

    if args.mode == "train":
        if args.train_mode == "stepbystep":
            for idx_ in range(args.step_nb):
                print('step number: ', idx_)
                tr_tas: List[TaskArgs] = get_tr_ta_list(net_ta_dict, idx_)
                for ta in tr_tas:
                    ta.run_one_step(net_ta_dict, idx_)
                    ta.do_vilidation_if_need(net_ta_dict, idx_)
        else:
            tr_tas: List[TaskArgs] = get_tr_ta_list(net_ta_dict, 0)
            for ta in tr_tas:
                ta.run_all_epochs()
    else:
        for net_name, ta in net_ta_dict.items():
            ta.infer()


if __name__ == '__main__':
    train_mtnet()
