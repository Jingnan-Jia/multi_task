# -*- coding: utf-8 -*-
# @Time    : 9/2/21 10:25 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import glob
import logging
import os

import monai
import torch
from torch.nn import functional as F


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


def get_loss_of_multi_output(x, y, net, criteria, amp=True):
    if amp:
        with torch.cuda.amp.autocast():
            preds = net(x)  # tuple
            losses = [criteria(pred, y) for pred in preds]
            # print(f"losses: {losses}")
            loss = 0.5 * losses[0] + 0.5 * torch.mean(torch.stack(losses[1:]))
            # print(f"weighted loss: {loss}")
            # pred0, pred1, pred2 = net(x)
            # loss0 = criteria(pred0, y)
            # loss1 = criteria(pred1, y)
            # loss2 = criteria(pred2, y)
            # print(f"loss: {loss0}, loss1: {loss1},loss2: {loss2},")
            # loss = 0.5 * loss0 + 0.25 * loss1 + 0.25 * loss2
    else:
        preds = net(x)  # tuple
        losses = [criteria(pred, y) for pred in preds]
        # print(f"losses: {losses}")
        loss = 0.5 * losses[0] + 0.5 * torch.mean(torch.stack(losses[1:]))
        # print(f"weighted loss: {loss}")
        #
        # pred0, pred1, pred2 = net(x)
        # loss0 = criteria(pred0, y)
        # loss1 = criteria(pred1, y)
        # loss2 = criteria(pred2, y)
        # print(f"loss: {loss0}, loss1: {loss1},loss2: {loss2},")
        # loss = 0.5 * loss0 + 0.25 * loss1 + 0.25 * loss2
    return loss


def get_loss_of_1_output(x, y, net, criteria, amp=True):
    if amp:
        with torch.cuda.amp.autocast():
            pred = net(x)
            loss = criteria(pred, y)
    else:
        pred = net(x)
        loss = criteria(pred, y)

    return loss


def get_loss_of_seg_rec(x, y, net, criteria, amp=True):
    if amp:
        with torch.cuda.amp.autocast():
            pred, pred_rec = net(x)
            loss = criteria(pred, y) + F.mse_loss(pred_rec, x)
    else:
        pred, pred_rec = net(x)
        loss = criteria(pred, y) + F.mse_loss(pred_rec, x)
    return loss


def get_model_path(model_folder):
    ckpts = sorted(glob.glob(os.path.join(model_folder, "*.pt")))
    ckpt = ckpts[-1]
    for x in ckpts:
        logging.info(f"available model file: {x}.")
    logging.info("----")
    logging.info(f"using {ckpt}.")
    return ckpt


def listdirs(path):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]