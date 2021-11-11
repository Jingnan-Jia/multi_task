# -*- coding: utf-8 -*-
# @Time    : 9/2/21 10:14 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import csv
import glob
import logging
import os
import random
import shutil
import time
from shutil import copy2
from typing import List, Optional, Union, Dict, Tuple
import seg_metrics.seg_metrics as sg
import pathlib
import monai
import numpy as np
import torch
import jjnutils.util as cu
from ignite.contrib.handlers import ProgressBar
from monai.data import NibabelReader, ITKReader, Dataset
from monai.handlers import CheckpointSaver, MeanDice, ValidationHandler, StatsHandler
from monai.transforms import LoadImaged, AddChanneld, Orientationd, Spacingd, ScaleIntensityRanged, SpatialPadd, \
    RandAffined, RandCropByPosNegLabeld, RandGaussianNoised, RandSpatialCropd, CastToTyped, ToTensord, AsDiscreted
from torch import nn as nn
from mt.mymodules.tool import Tracker

from mt.mymodules.mypath import Mypath
from mt.mymodules.set_args_mtnet import get_args
from mt.mymodules.data import inifite_generator
from mt.mymodules.archive import get_loss_of_multi_output, get_loss_of_1_output, get_loss_of_seg_rec, get_model_path
from torch.nn.modules.loss import _Loss
from typing import Callable, List, Optional, Sequence, Union
from monai.utils import LossReduction, Weight
import warnings
from monai.networks import one_hot

args = get_args()


def get_all_ct_names(path, number=None, prefix=None, name_suffix=None):
    print(f'data path: {os.path.abspath(path)  }')
    suffix_list = [".nrrd", ".mhd", ".mha", ".nii", ".nii.gz"]  # todo: more suffix

    if prefix and name_suffix:
        files = glob.glob(path + '/' + prefix + "*" + name_suffix + suffix_list[0])
        for suffix in suffix_list[1:]:
            files.extend(glob.glob(path + '/' + prefix + "*" + name_suffix + suffix))
    elif prefix:
        files = glob.glob(path + '/' + prefix + "*" + suffix_list[0])
        for suffix in suffix_list[1:]:
            files.extend(glob.glob(path + '/' + prefix + "*" + suffix))
    elif name_suffix:
        if 'SSc' in path:
            files = glob.glob(path + '/*/' + "*" + name_suffix + '.mha')
        else:
            files = glob.glob(path + '/' + "*" + name_suffix + suffix_list[0])
            for suffix in suffix_list[1:]:
                files.extend(glob.glob(path + '/' + "*" + name_suffix + suffix))

    else:
        files = glob.glob(path + '/*' + suffix_list[0])
        for suffix in suffix_list[1:]:
            files.extend(glob.glob(path + '/*' + suffix))

    scan_files = sorted(files)
    if len(scan_files) == 0:
        raise Exception(f'Scan files are None, please check the data directory: {path}')
    if isinstance(number, int) and number!=0:
        scan_files = scan_files[:number]
    elif isinstance(number, list):  # number = [3,7]
        scan_files = scan_files[number[0]:number[1]]

    return scan_files


class WeightedDiceLoss(_Loss):
    """Weighted Dice for multi-class segmentation. Small objects would be bigger weights.

    """

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = True,
        sigmoid: bool = False,
        softmax: bool = True,
        other_act: Optional[Callable] = None,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: Union[LossReduction, str] = LossReduction.MEAN,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
    ) -> None:
        """
        Args:
            include_background: if False, channel index 0 (background category) is excluded from the calculation.
            to_onehot_y: whether to convert `y` into the one-hot format. Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction.
            softmax: if True, apply a softmax function to the prediction.
            other_act: if don't want to use `sigmoid` or `softmax`, use other callable function to execute
                other activation layers, Defaults to ``None``. for example:
                `other_act = torch.tanh`.
            squared_pred: use squared versions of targets and predictions in the denominator or not.
            jaccard: compute Jaccard Index (soft IoU) instead of dice or not.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

            smooth_nr: a small constant added to the numerator to avoid zero.
            smooth_dr: a small constant added to the denominator to avoid nan.
            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, a Dice loss value is computed independently from each item in the batch
                before any `reduction`.

        Raises:
            TypeError: When ``other_act`` is not an ``Optional[Callable]``.
            ValueError: When more than 1 of [``sigmoid=True``, ``softmax=True``, ``other_act is not None``].
                Incompatible values.

        """
        super().__init__(reduction=LossReduction(reduction).value)
        if other_act is not None and not callable(other_act):
            raise TypeError(f"other_act must be None or callable but is {type(other_act).__name__}.")
        if int(sigmoid) + int(softmax) + int(other_act is not None) > 1:
            raise ValueError("Incompatible values: more than 1 of [sigmoid=True, softmax=True, other_act is not None].")
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act
        self.squared_pred = squared_pred
        self.jaccard = jaccard
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)
        self.batch = batch

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD], where N is the number of classes.
            target: the shape should be BNH[WD] or B1H[WD], where N is the number of classes.

        Raises:
            AssertionError: When input and target (after one hot transform if setted)
                have different shapes.
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].

        """
        if self.sigmoid:
            input = torch.sigmoid(input)

        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)

        if self.other_act is not None:
            input = self.other_act(input)

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # if skipping background, removing first channel
                target = target[:, 1:]
                input = input[:, 1:]

        if target.shape != input.shape:
            raise AssertionError(f"ground truth has different shape ({target.shape}) from input ({input.shape})")

        # reducing only spatial dimensions (not batch nor channels)
        reduce_axis: List[int] = torch.arange(2, len(input.shape)).tolist()
        if self.batch:
            # reducing spatial dimensions and batch
            reduce_axis = [0] + reduce_axis

        intersection = torch.sum(target * input, dim=reduce_axis)

        if self.squared_pred:
            target = torch.pow(target, 2)
            input = torch.pow(input, 2)

        ground_o = torch.sum(target, dim=reduce_axis)
        pred_o = torch.sum(input, dim=reduce_axis)

        denominator = ground_o + pred_o

        if self.jaccard:
            denominator = 2.0 * (denominator - intersection)

        f: torch.Tensor = 1.0 - (2.0 * intersection + self.smooth_nr) / (denominator + self.smooth_dr)


        volume_tot = torch.sum(ground_o)
        ratio_per_class = ground_o / volume_tot
        weight_per_class = 1 / (ratio_per_class + self.smooth_nr)
        weight_tot = torch.sum(weight_per_class)
        normalized_weight_per_class = weight_per_class / weight_tot
        print(f'normalize weights: {normalized_weight_per_class}')
        f = f * normalized_weight_per_class
        f = torch.mean(f)  # the batch and channel average

        return f


class WeightedCELoss(nn.Module):
    """
    this is soft CrossEntropyLoss
    """

    def __init__(self, mode='fnfp', adap_weights=None):

        super().__init__()
        # self.nllloss = nn.NLLLoss()
        # self.softmax = nn.functional.softmax(dim=1)
        # return
        self.mode = mode
        self.adap_weights = adap_weights

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1
            target = target[:, 0]
        # loss = nn.CrossEntropyLoss()
        # celoss = loss(input,target.long())

        target = target.unsqueeze(1)
        # print(input.shape)
        # print(target.shape)
        batch = input.shape[0]
        cls = input.shape[1]
        target_onehot = torch.FloatTensor(input.shape)
        if target.device.type == "cuda":
            target_onehot = target_onehot.cuda(target.device.index)
        target_onehot.zero_()
        target_onehot.scatter_(1, target.type(torch.int64), 1)
        if self.adap_weights == None:
            reduce_axis: List[int] = torch.arange(2, len(input.shape)).tolist()
            self.weights = torch.sum(target_onehot, dim=reduce_axis)
            self.weights = self.weights / torch.sum(self.weights)

            self.weights = 1 / self.weights
            self.weights = self.weights / torch.sum(self.weights)

            print(f'weights for CE: {self.weights}')
        input_pro = nn.functional.softmax(input, dim=1)
        fn_logpro = torch.log(input_pro + 1e-6)
        fp_logpro = torch.log(1 - input_pro + 1e-6)

        ce_fn = -1 * fn_logpro * target_onehot
        ce_fn = ce_fn.view(batch, cls, -1)
        ce_fn = torch.mean(ce_fn, dim=2)
        ce_fn = self.weights * ce_fn
        ce_fn = torch.sum(ce_fn, dim=1)
        ce_fn = torch.mean(ce_fn, dim=0)
        if self.mode == 'fn':
            return ce_fn
        else:
            ce_fp = -1 * fp_logpro * (1 - target_onehot)
            ce_fp = ce_fp.view(batch, cls, -1)
            ce_fp = torch.mean(ce_fp, dim=2)
            ce_fp = self.weights * ce_fp
            ce_fp = torch.sum(ce_fp, dim=1)
            ce_fp = torch.mean(ce_fp, dim=0)

            return ce_fn + ce_fp


class DiceCELoss(nn.Module):
    """Dice and Xentropy loss"""

    def __init__(self):
        super().__init__()
        self.dice = monai.losses.DiceLoss(include_background=True, to_onehot_y=True, softmax=True)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, y_pred, y_true):
        dice = self.dice(y_pred, y_true)
        # CrossEntropyLoss target needs to have shape (B, D, H, W)
        # Target from pipeline has shape (B, 1, D, H, W)
        cross_entropy = self.cross_entropy(y_pred, torch.squeeze(y_true, dim=1).long())
        print(f"dice loss: {dice}, CE loss: {cross_entropy}")
        return dice + cross_entropy


class CELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, y_pred, y_true):
        # CrossEntropyLoss target needs to have shape (B, D, H, W)
        # Target from pipeline has shape (B, 1, D, H, W)
        cross_entropy = self.cross_entropy(y_pred, torch.squeeze(y_true, dim=1).long())
        return cross_entropy


def get_loss(task: str) -> nn.Module:
    """Return loss function from its name.

    Args:
        task: task name

    """
    loss_fun: nn.Module
    print(f'loss: {args.loss}')
    if task == "recon":
        loss_fun = nn.MSELoss()  # do not forget parenthesis
    else:
        if args.loss == "dice":
            loss_fun = monai.losses.DiceLoss(to_onehot_y=True, softmax=True)
        elif args.loss == "CE":
            loss_fun = CELoss()
        elif args.loss == "dice_CE":
            loss_fun = DiceCELoss()  # or FocalLoss
        elif args.loss == "weighted_dice":
            loss_fun = WeightedDiceLoss(to_onehot_y=True, softmax=True)
        elif args.loss == "weighted_CE_fnfp":
            loss_fun = WeightedCELoss()
        elif args.loss == "weighted_CE_fn":
            loss_fun = WeightedCELoss(mode='fn')
        else:
            raise ValueError(f"loss_fun should be 'dice', 'CE', 'dice_CE', 'weighted_dice', 'weighted_CE', but got {args.loss}")
    return loss_fun


def from_engine(keys, first: bool = False, device=torch.device("cpu")):
    """
    Utility function to simplify the `batch_transform` or `output_transform` args of ignite components
    when handling dictionary or list of dictionaries(for example: `engine.state.batch` or `engine.state.output`).
    Users only need to set the expected keys, then it will return a callable function to extract data from
    dictionary and construct a tuple respectively.

    If data is a list of dictionaries after decollating, extract expected keys and construct lists respectively,
    for example, if data is `[{"A": 1, "B": 2}, {"A": 3, "B": 4}]`, from_engine(["A", "B"]): `([1, 3], [2, 4])`.

    It can help avoid a complicated `lambda` function and make the arg of metrics more straight-forward.
    For example, set the first key as the prediction and the second key as label to get the expected data
    from `engine.state.output` for a metric::

        from monai.handlers import MeanDice, from_engine

        metric = MeanDice(
            include_background=False,
            output_transform=from_engine(["pred", "label"])
        )

    Args:
        keys: specified keys to extract data from dictionary or decollated list of dictionaries.
        first: whether only extract sepcified keys from the first item if input data is a list of dictionaries,
            it's used to extract the scalar data which doesn't have batch dim and was replicated into every
            dictionary when decollating, like `loss`, etc.


    """
    def _wrapper(data):
        if isinstance(data, dict):
            return tuple(data[k].to(device) for k in keys)
        elif isinstance(data, list) and isinstance(data[0], dict):
            # if data is a list of dictionaries, extract expected keys and construct lists,
            # if `first=True`, only extract keys from the first item of the list
            ret = [data[0][k].to(device) if first else [i[k].to(device) for i in data] for k in keys]
            # ret = [x.to(device) for x in ret]
            return tuple(ret) if len(ret) > 1 else ret[0]

    return _wrapper


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
        device=torch.device('cpu')  # avoid CUDA out of memory
    )
    return inferer


class TaskArgs:
    """A container. The network's necessary parameters, datasets, training steps are in this container.

    """
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amp = True
        self.task: str = task
        self.main_task = None
        self.labels: List[int] = labels
        self.net_name: str = net_name
        self.net: nn.Module = all_nets[net_name]
        self.net.to(self.device)
        self.tr_nb: Union[int] = tr_nb
        self.tr_nb_cache: int = 200
        self.current_step = 0

        self.ds: int = ds
        self.tsp_xy = float(tsp.split("_")[0])
        self.tsp_z = float(tsp.split("_")[1])
        self.sub_dir = sub_dir
        self.load_workers = 3
        self.scaler = torch.cuda.amp.GradScaler()

        self.lr: float = lr
        self.main_net_name: str = main_net_name
        self.main_net = all_nets[self.main_net_name]
        self.keys: Tuple[str, str] = ("pred", "label")

        self.tracker = Tracker(task_name=self.task, data_path=args.data_path, ld_name=ld_name)  # record super parameters and metrics
        self.id = self.tracker.record_start(args)  # first record, get the id, then have the path.
        self.mypath = Mypath(self.id, task, data_path=args.data_path, check_id_dir=False)
        self.ld_name = ld_name  # for fine-tuning and inference
        self.ld_path = Mypath(self.ld_name, task, data_path=args.data_path, check_id_dir=False)
        if ld_name:  # fine-tuning
            self.trained_model_folder = self.ld_path.id_dir
            ckpt = get_model_path(self.trained_model_folder)
            self.net.load_state_dict(torch.load(ckpt, map_location=self.device))
            print(f'load model from {ckpt}')

        self.n_classes = len(self.labels)

        if args.mode=="train":  # if in ínfer mode, do not need to load tra_loader or val_loader at all.
            self.tra_loader, self.val_loader = self._dataloader()
            if args.fluent_ds:
                self.tra_loader2 = self._dataloader(require_val=False) # do not second need val data
                self.tra_gen = inifite_generator(self.tra_loader, self.tra_loader2)
            else:
                self.tra_gen = inifite_generator(self.tra_loader)

            self.evaluator = self._get_evaluator()

            self.loss_fun = get_loss(task)
            self.opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)

            # self.main_net_enc_grad = [layer.grad for layer in self.net.enc]
            # self.main_net_enc_grad_norm = [torch.norm(grad) for grad in self.main_net_enc_grad]
            self.criteria = self.loss_fun
            self.accumulate_loss: float = 0.0
            self.current_loss: float = 100.0
            self.steps_per_epoch = self.n_train * args.pps
            self.tracker.record('steps_per_epoch', self.steps_per_epoch)


        if args.mode == "infer":
            self.infer_loader = self.get_infer_loader(transformmode="infer")
        elif "yichao" in self.net_name:
            self.infer_loader = self.get_infer_loader(transformmode="infer_patches")
            self.ini_loader = inifite_generator(self.infer_loader, keys=("image",))

    def _get_enc_parameters(self):
        enc_parameters = {}
        enc_gradients = {}
        for name, param in self.net.enc.named_parameters():
            if param.requires_grad:
                enc_parameters[name] = param
                enc_gradients[name] = param.grad
        return enc_parameters, enc_gradients

    def _get_xforms(self, mode: str = "train", keys=("image", "label")):
        """returns a composed transform for train/val/infer."""
        nibabel_reader = NibabelReader()
        itk_reader = ITKReader()
        xforms = [
            LoadImaged(keys),  # .nii, .nii.gz [.mhd, .mha LoadImage]
            AddChanneld(keys),
            Orientationd(keys, axcodes="LPS"),
            Spacingd(keys, pixdim=(self.tsp_xy, self.tsp_xy, self.tsp_z), mode=("bilinear", "nearest")[: len(keys)]),
            ScaleIntensityRanged(keys[0], a_min=-1000.0, a_max=500.0, b_min=0.0, b_max=1.0, clip=True),
        ]
        if mode == "train":
            xforms.extend(
                [
                    # SpatialPadd(keys, spatial_size=(args.patch_xy, args.patch_xy, args.patch_z), mode="minimum"),
                    # ensure at least HTxHT*z
                    RandAffined(
                        keys,
                        prob=0.3,
                        rotate_range=(0.05, 0.05),
                        scale_range=(0.1, 0.1, 0.1),
                        mode=("bilinear", "nearest"),
                        as_tensor_output=False,
                    ),
                    SpatialPadd(keys, spatial_size=(args.patch_xy, args.patch_xy, args.patch_z), mode="minimum"),
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
        elif mode == "infer_patches":
            keys = ("image")
            xforms.extend(
                [
                    SpatialPadd(keys, spatial_size=(args.patch_xy, args.patch_xy, args.patch_z), mode="minimum"),
                    # ensure at least HTxHT*z
                    RandAffined(
                        keys,
                        prob=0.3,
                        rotate_range=(-0.1, 0.1),
                        scale_range=(-0.1, 0.1),
                        mode=("bilinear",),
                        as_tensor_output=False,
                    ),
                    RandSpatialCropd(keys,
                                     roi_size=(args.patch_xy, args.patch_xy, args.patch_z),
                                     random_center=True,
                                     random_size=False),
                    RandGaussianNoised(keys, prob=0.5, std=0.01),
                ]
            )
            dtype = (np.float32,)
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

    def _norm_enc_gradients(self):
        self.target_enc_grad_norm = {name:norm * args.ratio_norm_gradients
                                     for name, norm in self.main_task.enc_grad_norm.items()}
        self.ratio_norm = {name:self.target_enc_grad_norm[name] / norm for name, norm in self.enc_grad_norm.items()}

        for name, parameter in self.net.enc.named_parameters():
            # print("before norm, grad.norm", torch.norm(parameter.grad))
            parameter.grad *= self.ratio_norm[name]
            # print("after norm, grad.norm", torch.norm(parameter.grad))

            # self.net.enc.named_parameters()[name].grad *= ratio
            # self.enc_parameters[name] *= ratio


    def _update_gradients(self, loss, amp):
        self.opt.zero_grad()
        if amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # t = time.time()
        self.enc_parameters, self.enc_gradients = self._get_enc_parameters()
        self.enc_grad_norm = {name: torch.norm(gradient) for name, gradient in self.enc_gradients.items()}
        if args.ratio_norm_gradients and self.net_name != self.main_net_name:
            self._norm_enc_gradients()
        # print(f"update gradients cost time: {time.time()-t}")

        if amp:
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            self.opt.step()

    def _get_file_names(self):
        """Return 2 lists of training and validation file names."""
        keys = ("image", "label")
        data_dir = self.mypath.data_task_dir
        # data_dir = "/data/jjia/monai/COVID-19-20_v2/Train"
        print(f'data dir: {data_dir}')

        ct_names: List[str] = get_all_ct_names(data_dir, name_suffix="_ct")

        if self.task != "recon":
            gdth_names: List[str] = get_all_ct_names(data_dir, name_suffix="_seg")
        else:
            gdth_names = ct_names
        if self.main_net_name!='lobe': # pat_28 should be in testing dataset.
            SEED = 47
            random.seed(SEED)
            random.shuffle(ct_names)
            random.seed(SEED)
            random.shuffle(gdth_names)

        train_frac, val_frac, test_frac = 0.7, 0.1, 0.2
        data_len = len(gdth_names)
        tr_nb = int(train_frac * data_len)
        vd_nb = int(val_frac * data_len)
        ts_nb = int(test_frac * data_len)
        rest = data_len - tr_nb - vd_nb - ts_nb
        ts_nb += rest  # assign the rest patients to testing dataset
        if vd_nb==0:
            vd_nb = 1
            tr_nb -= 1
        if ts_nb == 0:
            ts_nb = 1
            tr_nb -= 1
        if tr_nb==0:
            raise Exception(f"training number is 0 !")

        if self.tr_nb != 0:
        #     total_nb = min(self.tr_nb, tr_nb)  # if set tr_nb
        # else:
        #     total_nb = len(ct_names)
            self.n_train: int = min(tr_nb, self.tr_nb)
        else:
            self.n_train: int = tr_nb

        if self.net_name != self.main_net_name:
            self.n_val: int = min(5, vd_nb)
        else:
            # self.n_val: int = min(total_nb - self.n_train, int(val_frac * total_nb))
            self.n_val: int = vd_nb
        # self.n_train, self.n_val = 2, 2 # todo: change it.

        logging.info(f"In task {self.task}, training number:  {self.n_train} valid number: {self.n_val}")

        train_files = [{keys[0]: img, keys[1]: seg} for img, seg in
                       zip(ct_names[:self.n_train], gdth_names[:self.n_train])]
        val_files = [{keys[0]: img, keys[1]: seg} for img, seg in zip(ct_names[self.n_train: self.n_train+self.n_val],
                                                                      gdth_names[self.n_train: self.n_train+self.n_val])]
        ts_files = [{keys[0]: img, keys[1]: seg} for img, seg in zip(ct_names[-ts_nb:], gdth_names[-ts_nb:])]
        print(f"In task {self.task}")
        print(f"train_files: {train_files}")
        print(f"valid_files: {val_files}")
        print(f"test_files: {ts_files}")

        return train_files, val_files

    def _dataloader(self, require_val=True):
        """Return train (and valid) dataloader.

        Args:
            require_val: if return valid_dataloader.

        """
        ct_name_list: List[str]
        gdth_name_list: List[str]
        train_files, val_files = self._get_file_names()
        # train_files, val_files = train_files[:2], val_files[:2]

        train_transforms = self._get_xforms("train")
        if args.cache:
            if args.smartcache :
                # if args.smartcache:
                self.cache_num = min(self.tr_nb_cache, self.n_train - 1)
                # else:
                #     self.cache_num = self.tr_nb_cache
                # cache_num must be smaller than dataset length to support replacement.
                self.train_ds = monai.data.SmartCacheDataset(data=train_files,
                                                             transform=train_transforms,
                                                             replace_rate=1,
                                                             cache_num=self.cache_num,
                                                             num_init_workers=5,
                                                             num_replace_workers=self.load_workers,
                                                             ) #or self.n_train > self.tr_nb_cache
            else:
                self.train_ds = monai.data.CacheDataset(data=train_files, transform=train_transforms,
                                                        num_workers=self.load_workers, cache_rate=1)
        else:
            self.train_ds = Dataset(data=train_files, transform=train_transforms,)

        train_loader = monai.data.DataLoader(
            self.train_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=self.load_workers,
            # pin_memory=torch.cuda.is_available(),
            pin_memory=False,
            persistent_workers=True,

        )
        if args.smartcache : #or self.n_train > self.tr_nb_cache
            self.train_ds.start()  # need it if SmartCacheDataset

        if not require_val:
            return train_loader
        else:
            # create a validation data loader
            val_transforms = self._get_xforms("val")
            print('valid files:', val_files)
            val_ds = monai.data.CacheDataset(data=val_files, transform=val_transforms, num_workers=self.load_workers)
            val_loader = monai.data.DataLoader(
                val_ds,
                batch_size=1,  # image-level batch to the sliding window method, not the window-level batch
                num_workers=self.load_workers,
                # pin_memory=torch.cuda.is_available(),
                pin_memory=False,
                persistent_workers=True,

            )
            return train_loader, val_loader

    def _get_evaluator(self):
        """Return evaluator. record_val_metrics after one evaluation.

        """
        keys = ("pred", "label")

        val_post_transform = monai.transforms.Compose(
            [ToTensord(keys=("pred", "label")), AsDiscreted(keys=keys, argmax=(True, False), to_onehot=True, n_classes=self.n_classes)]
        )
        val_handlers = [
            ProgressBar(),
            CheckpointSaver(save_dir=self.mypath.id_dir,
                            save_dict={"net": self.net},
                            save_key_metric=True,
                            key_metric_n_saved=3),
        ]
        evaluator = monai.engines.SupervisedEvaluator(
            device=self.device,
            val_data_loader=self.val_loader,
            network=self.net,
            inferer=get_inferer(),
            postprocessing=val_post_transform,
            key_val_metric={"dice_ex_bg": MeanDice(include_background=False,
                                          # output_transform=lambda x: (x[keys[0]].to(torch.device('cpu')),
                                          #                             x[keys[1]].to(torch.device('cpu'))))
                                            output_transform = from_engine(["pred", "label"]))
            },
            additional_metrics={"dice_inc_bg": MeanDice(include_background=True,
                                                        output_transform = from_engine(["pred", "label"]))},
            val_handlers=val_handlers,
            amp=self.amp,
        )

        def record_val_metrics(engine):
            engine.state.epoch = int(self.current_step / (self.steps_per_epoch + 0.1))
            val_log_dir = self.mypath.metrics_fpath('valid')
            if os.path.exists(val_log_dir):
                val_log = np.genfromtxt(val_log_dir, dtype='str', delimiter=',')
            else:
                val_log = ['epoch', 'step', 'dice_ex_bg', 'dice_inc_bg']
            val_log = np.vstack([val_log, [engine.state.epoch,
                                           self.current_step,
                                           round(engine.state.metrics["dice_ex_bg"], 3),
                                           round(engine.state.metrics["dice_inc_bg"], 3),]])
            np.savetxt(val_log_dir, val_log, fmt='%s', delimiter=',')

        from ignite.engine import Events
        evaluator.add_event_handler(Events.COMPLETED, record_val_metrics)

        return evaluator

    def stop_data_iter(self):
        print(f'how to stop ?')
        # self.val_loader

    def run_one_step(self, net_ta_dict, idx: int):
        """Run one step.

        Args:
            net_ta_dict: a dick with net name as key, task as value.
            idx: index of step

        Q: why we need net_ta_dict?
        A: Main task was not be instiated during the instiatation of other tasks.


        """

        self.current_step = idx
        self.net.train()
        t1 = time.time()
        if "yichao" in self.net_name:
            print("yichao's net")
            x = next(self.ini_loader)
            print(f"x.shape: {x.shape}")
            t3 = time.time()
            # print(f"load data cost time: {t3 - t1}")
            x = x.to(self.device)
            y = self.net(x)  # now we have x and its gdth: y

            t4 = time.time()
            print(f"forward data cost time: {int(t4 - t3)}")
            x = x.to("cpu")
            if len(y) > 1:
                y = y[0]
            y = torch.argmax(y, dim=1)  # one hot decoding
            y = torch.unsqueeze(y, dim=1)

            y = y.to("cpu")
            print(f"y.shape: {y.shape}")
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
            x_Affined = x_Affined.float()
            y_Affined = y_Affined.float()

            t5 = time.time()
            print(f"transform data cost time: {int(t5 - t4)}")

            pred = self.net(x_Affined)
            if len(pred) > 1:
                pred = pred[0]
            print(f"pred.shape: {pred.shape}, y_Affined.shape: {y_Affined.shape}")
            loss = self.criteria(pred, y_Affined)
        else:

            x, y = next(self.tra_gen)
            t3 = time.time()
            # print(f"load data cost time: {t3 - t1}")
            if self.task == "recon":
                y = x
            x = x.to(self.device)
            y = y.to(self.device)

            if "itgt" in self.net_name:
                # print("self.netname", self.net_name)
                if "vae" in self.net_name or "VAE" in self.net_name:
                    pred, rec_loss = self.net(x)
                    loss = self.criteria(pred, y) + rec_loss
                else:
                    loss = get_loss_of_seg_rec(x, y, self.net, self.criteria, args.amp)
            else:
                if (self.ds and "recon" not in self.net_name) or "unetpp" in self.net_name:
                    loss = get_loss_of_multi_output(x, y, self.net, self.criteria, args.amp)
                else:
                    loss = get_loss_of_1_output(x, y, self.net, self.criteria, args.amp)
        t8 = time.time()
        self._update_gradients(loss, args.amp)

        self.current_loss = loss.item()

        # for
        t2 = time.time()
        print(f"load data cost time {int(t3 - t1)}, one step backward training cost time: {t2 - t8}")
        if (args.ad_lr!=0) and self.main_net_name != self.net_name:  # reset lr for aux nets
            lr = net_ta_dict[self.main_net_name].current_loss / self.current_loss * self.lr * args.ad_lr
            self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
            print(f"task: {self.task}, lr: {lr}")
        print(f"task: {self.task}, loss: {loss.item()}")

        if args.smartcache and (idx % (self.cache_num * args.pps)) == 0: #or self.n_train > self.tr_nb_cache
            print(f"start update cache for task {self.task}")
            self.train_ds.update_cache()
        if args.smartcache and idx == args.step_nb - 1: #or self.n_train > self.tr_nb_cache
            print(f'shutdown training dataloader')
            self.train_ds.shutdown()

        # print statistics
        self.accumulate_loss += loss.item()
        if idx % self.steps_per_epoch == 0:  # print every 2000 mini-batches
            ave_tr_loss = self.accumulate_loss / self.steps_per_epoch
            print(f'step: {idx} average training loss: {ave_tr_loss}')
            if not os.path.isfile(self.mypath.metrics_fpath('train')):
                with open(self.mypath.metrics_fpath('train'), 'a') as csv_file:
                    writer = csv.writer(csv_file, delimiter=',')
                    writer.writerow(["step", "ave_loss_in_epoch"])
            with open(self.mypath.metrics_fpath('train'), 'a') as csv_file:
                writer = csv.writer(csv_file, delimiter=',')
                writer.writerow([idx, ave_tr_loss])
            self.accumulate_loss = 0.0

    def run_all_epochs(self):
        # evaluator as an event handler of the trainer
        # epochs = [int(args.epochs * 0.8), int(args.epochs * 0.2)]
        epochs = [200, 100]
        intervals = [1, 1]
        lrs = [1e-4, 1e-4]
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

    def do_validation_if_need(self, net_ta_dict, idx_: int):
        if idx_ < int(args.step_nb * 0.8):
            valid_period = args.valid_period1 * net_ta_dict[self.main_net_name].steps_per_epoch
        else:
            valid_period = args.valid_period2 * net_ta_dict[self.main_net_name].steps_per_epoch
        # print(f"vallid period: {valid_period}")
        print('idx_', idx_)
        if idx_ % valid_period == (valid_period-1):
            print("start do validation")
            if "net_recon" not in self.net_name:
                print('start evaluate')
                t1 = time.time()
                self.evaluator.run()
                print(f"evaluation cost time : {int(time.time() - t1)} seconds")

    def get_infer_loader(self, transformmode="infer"):
        data_folder = args.infer_data_dir
        if data_folder in ['None', None]:
            train_files, infer_files = self._get_file_names()
            print('infered files:', infer_files)
        else:
            images = sorted(glob.glob(os.path.join(data_folder, "*_ct*")))
            logging.info(f"infer: image ({len(images)}) folder: {data_folder}")
            infer_files = [{"image": img} for img in images]

        keys = ("image",)
        infer_transforms = self._get_xforms(transformmode, keys)
        infer_ds = monai.data.Dataset(data=infer_files, transform=infer_transforms)
        infer_loader = monai.data.DataLoader(
            infer_ds,
            batch_size=1,  # image-level batch to the sliding window method, not the window-level batch
            num_workers=self.load_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=False,
        )

        return infer_loader

    def infer(self, write_pbb_maps=True):
        """
        run inference, the output folder will be "./output"
        :param write_pbb_maps: write probabilities maps to the disk for future boosting
        """
        print("start infer")
        keys = ("image",)
        self.net.eval()
        if args.infer_data_dir is None:  # prediction may be GLUCOLD or LUNA16 or LOLA11
            prediction_folder = os.path.join(self.ld_path.infer_pred_dir(), self.ld_path.data_sub_dir())
        else:
            prediction_folder = os.path.join(self.ld_path.infer_pred_dir(), args.infer_data_dir.split("/")[-1])
        print(prediction_folder)
        inferer = get_inferer()
        saver = monai.data.NiftiSaver(output_dir=prediction_folder, mode="nearest")  # todo: change mode
        with torch.no_grad():
            for infer_data in self.infer_loader:
                logging.info(f"segmenting {infer_data['image_meta_dict']['filename_or_obj']}")
                print(f"segmenting {infer_data['image_meta_dict']['filename_or_obj']}")
                preds = inferer(infer_data[keys[0]].to(self.device), self.net)
                if args.infer_4_times:
                    n = 1.0
                    for _ in range(4):
                        # test time augmentations
                        _img = RandGaussianNoised(keys[0], prob=1.0, std=0.01)(infer_data)[keys[0]]
                        pred = inferer(_img.to(self.device), self.net)
                        preds = preds + pred
                        n = n + 1.0
                        if 'lesion' in self.task:
                            for dims in [[2], [3]]:
                                flip_pred = inferer(torch.flip(_img.to(self.device), dims=dims), self.net)
                                pred = torch.flip(flip_pred, dims=dims)
                                preds = preds + pred
                                n = n + 1.0
                    preds = preds / n
                else:
                    preds = inferer(infer_data[keys[0]].to(self.device), self.net)
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
        if not os.path.exists(prediction_folder):
            os.makedirs(prediction_folder)
        # files = glob.glob(os.path.join(prediction_folder,"*", "*.nii.gz"))
        files = get_all_ct_names(os.path.join(prediction_folder,"*"), name_suffix="_ct_seg")

        for f in files:
            new_name = os.path.basename(f)
            new_name = new_name.split("_ct")[0] + new_name.split("_ct")[1]
            # new_name = new_name[len("volume-covid19-A-0"):]
            # new_name = new_name[: -len("_ct_seg.nii.gz")] + ".nii.gz"
            to_name = os.path.join(prediction_folder, new_name )
            shutil.move(f, to_name)
            parent_dir = pathlib.Path(f).parent.absolute()  # remove the empty directory after move its file
            if len(os.listdir(parent_dir)) == 0:
                # removing the file using the os.remove() method
                os.rmdir(parent_dir)

        logging.info(f"predictions copied to {prediction_folder}.")

        # if "lobe" in self.net_name:  # other tasks do not have lobe ground truth
        #     gdth_file = self.mypath.data_task_dir
        #     pred_file = prediction_folder
        #     csv_file = os.path.join(prediction_folder, 'metrics.csv')
        #
        #     metrics = sg.write_metrics(labels=self.labels[1:],  # exclude background
        #                                gdth_path=gdth_file,
        #                                pred_path=pred_file,
        #                                csv_file=csv_file)
        #     print('metrics: ', metrics)

        # if "lung" in self.net_name:
        #
        #     write_connected_lobes(preds_dir, workers=5, target_dir=preds_dir + "/biggest_parts")