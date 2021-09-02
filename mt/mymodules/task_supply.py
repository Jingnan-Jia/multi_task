# -*- coding: utf-8 -*-
# @Time    : 9/2/21 10:19 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import argparse
from typing import Dict, List

import monai
import numpy as np
import torch
from torch import nn as nn

from mt.mymodules.custom_net import Encoder, DecoderRec, EnsembleEncRec, DecoderSegRec, EnsembleEncSeg, DecoderSeg
from mt.mymodules.networks.Generic_UNetPlusPlus import Generic_UNetPlusPlus
from mt.mymodules.networks.saharnet import Saharnet_encoder, Saharnet_decoder
from mt.mymodules.networks.unet_att_dsv import unet_CT_single_att_dsv_3D
from mt.mymodules.set_args_mtnet import args
from mt.mymodules.task import TaskArgs


def model_summary(model, model_name):
    print(f"=================model_summary: {model_name}==================")
    # model_parameters = [layer for layer in model.parameters() if layer.requires_grad]
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"Total Params:{params}")
    print("=" * 100)

    # total_params = 0
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         p = np.prod(param.shape)
    #         print(name, p)
    #         total_params+=p
    # print("=" * 100)
    # print(f"Total Params:{total_params}")


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


def mt_netname_net(netname_label_dict: Dict[str, List], netname_ds_dict: Dict[str, int], base: int = 1) -> Dict[
    str, nn.Module]:
    """Get multi-task net.

    Args:
        netname_label_dict: A dict with key of net name, value of label.
        netname_ds_dict: A dict with key of net name, value of 'number of deep supervision'.
        base:

    Returns:
        A dict with key of net name, value of network.

    """
    nets = {}
    for net_name, label in netname_label_dict.items():
        if "saharnet" in net_name:
            sahar_flag = True
        else:
            sahar_flag = False
    if sahar_flag:
        enc = Saharnet_encoder()
    else:
        enc = Encoder(features=(32 * base, 32 * base, 64 * base, 128 * base, 256 * base, 32 * base), dropout=0.1)
    for net_name, label in netname_label_dict.items():
        if "att_unet" in net_name:
            net = unet_CT_single_att_dsv_3D(
                in_channels=1,
                n_classes=len(label),
                base=base
            )
        elif "resnet" in net_name:
            net = monai.networks.nets.SegResNet(
                init_filters=32 * args.base,  # todo: could change to base
                out_channels=len(label),
                dropout_prob=0.1
            )
        elif "unetpp" in net_name:
            net = Generic_UNetPlusPlus(1, args.base * 32, len(label), 5)
        elif "vae" in net_name or "VAE" in net_name:  # vae as the reconnet
            net = monai.networks.nets.SegResNetVAE(
                input_image_size=(args.patch_xy, args.patch_xy, args.patch_z),
                spatial_dims=3,
                init_filters=32 * args.base,  # todo: could change to base
                in_channels=1,
                out_channels=len(label),
                dropout_prob=0.1
            )
        elif net_name == "net_recon":  # reconstruction
            dec = DecoderRec(features=(32 * base, 32 * base, 64 * base, 128 * base, 256 * base, 32 * base),
                             dropout=0.1)
            net = EnsembleEncRec(enc, dec)
        else:  # segmentation output different channels
            if "itgt" in net_name:  # 2 outputs: pred and rec_loss
                dec = DecoderSegRec(out_channels=len(label),
                                    features=(32 * base, 32 * base, 64 * base, 128 * base, 256 * base, 32 * base),
                                    dropout=0.1, )
                net = EnsembleEncSeg(enc, dec)
            else:
                if sahar_flag:
                    dec = Saharnet_decoder(out_channels=len(label))
                else:
                    dec = DecoderSeg(out_channels=len(label),
                                     features=(32 * base, 32 * base, 64 * base, 128 * base, 256 * base, 32 * base),
                                     dropout=0.1,
                                     ds=netname_ds_dict[net_name])
                net = EnsembleEncSeg(enc, dec)
        # net = get_net()
        # print(net)
        model_summary(net, model_name=net_name)
        nets[net_name] = net

    return nets


def mt_netnames(myargs: argparse.Namespace) -> List[str]:
    """Get net names from arguments.

    Define the Model, use dash to separate multi net names, do not use ',' to separate it, because ',' can lead to
    unknown error during parse arguments

    Args:
        myargs:

    Returns:
        A list of net names

    """
    #
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


def mt_netname_label(net_names: List[str]) -> Dict[str, List[int]]:
    """Return a dict with net name as keys and net labels as values.

    Args:
        net_names: A list of net names

    """
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


def mt_netname_ds(net_names: List[str]) -> Dict[str, int]:
    """Return a dict with net name as keys and 'number of deep supervision path' as values.

    Args:
        net_names: A list of net names


    """
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


def get_loss(task: str) -> nn.Module:
    """Return loss function from its name.

    Args:
        task: task name

    """
    loss_fun: nn.Module
    if task == "recon":
        loss_fun = nn.MSELoss()  # do not forget parenthesis
    else:
        loss_fun = DiceCELoss()  # or FocalLoss
    return loss_fun


def _mt_netname_ta(netname_label_dict: Dict[str, List],
                   all_nets: Dict[str, nn.Module]) -> Dict[str, TaskArgs]:
    """Return a dict with net name as keys and task as values.

    Args:
        netname_label_dict: a dict with net name as keys and net labels as values.
        all_nets: a dict with net name as keys and nets as values.

    """
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

    for net_name, ta in ta_dict.items():
        ta.main_task = ta_dict[args.main_net_name]

    return ta_dict


def mt_netname_ta(net_names: List[str], args: argparse.Namespace) -> Dict[str, TaskArgs]:
    """Return a dick with keys of net names, values of tasks.

    Args:
        net_names: A list of net names.
        args: arguments

    """
    netname_label_dict: Dict[str, List] = mt_netname_label(net_names)
    netname_ds_dict: Dict[str, int] = mt_netname_ds(net_names)
    all_nets: Dict[str, nn.Module] = mt_netname_net(netname_label_dict, netname_ds_dict, args.base)
    ta_dict: Dict[str, TaskArgs] = _mt_netname_ta(netname_label_dict, all_nets)

    return ta_dict


def mt_fat_ta_list(net_ta_dict: Dict, main_name, idx_):
    """return [main_net, one_aux_net] for each epoch.

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


def mt_tr_ta_list(net_ta_dict, idx_, main_net_name, fat=1):
    """Return training task list for this training step.

    Args:
        net_ta_dict: A dick including net name and its corresponding task
        idx_: index of training step

    Returns:

    """
    if fat:  # focus-alternative-training
        tr_ta_list: List[TaskArgs] = mt_fat_ta_list(net_ta_dict, main_net_name, idx_)
    else:
        tr_ta_list: List[TaskArgs] = net_ta_dict
    return tr_ta_list


