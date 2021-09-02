# -*- coding: utf-8 -*-
# @Time    : 12/1/20 12:09 AM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
from typing import Sequence, Union

import torch
import torch.nn as nn

from monai.networks.blocks import Convolution, UpSample
from monai.networks.layers.factories import Conv, Pool
from monai.utils import ensure_tuple_rep
from monai.networks.nets.basic_unet import TwoConv, Down, UpCat


def mycat(upsampled, bypass):
    return torch.cat((upsampled, bypass), 1)


class Myconvolution(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = Convolution(dimensions=3,
                                in_channels=in_ch,
                                out_channels=out_ch,
                                act=("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
                                norm=("instance", {"affine": True}),
                                dropout=0.1)

    def forward(self, x):
        out = self.conv(x)
        return out


class TwoConv_sahar(nn.Module):
    def __init__(self, in_chn, out_chn1, out_chn2):
        super().__init__()
        self.conv1 = Convolution(dimensions=3,
                                 in_channels=in_chn,
                                 out_channels=out_chn1,
                                 kernel_size=1,
                                 act=("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
                                 norm=("instance", {"affine": True}),
                                 dropout=0)
        self.conv2 = Convolution(dimensions=3,
                                 in_channels=out_chn1,
                                 out_channels=out_chn2,
                                 kernel_size=3,
                                 dilation=2,  # with dilation
                                 act=("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
                                 norm=("instance", {"affine": True}),
                                 dropout=0)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        return x2


class DenseSubModule(nn.Module):
    def __init__(self, in_chn, chn1, chn2):
        super().__init__()
        self.convs = TwoConv_sahar(in_chn, chn1, chn2)

    def forward(self, x):
        out = self.convs(x)
        out = mycat(x, out)
        return out

class Mybroadcast(nn.Module):
    def __init__(self, chn):
        super().__init__()
        self.chn = chn
    def forward(self, x):
        out = torch.cat(self.chn * [x], dim=1)
        return out
    
    
class Pixel_attention(nn.Module):
    def __init__(self, chn):
        super().__init__()
        # self.max_pool = Pool["MAX", 3](kernel_size=1)
        # self.ave_pool = Pool["AVG", 3](kernel_size=1)
        self.conv = Convolution(dimensions=3,
                                in_channels=2,
                                out_channels=1,
                                kernel_size=3,
                                act=("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
                                norm=("instance", {"affine": True}),
                                dropout=0.1)
        self.sigmoid = nn.Sigmoid()
        self.broadcast = Mybroadcast(chn)

    def forward(self, x):
        # max_pool = self.max_pool(x)
        # ave_pool = self.ave_pool(x)
        max_pool, _ = torch.max(x, dim=1)
        ave_pool = torch.mean(x, dim=1)
        max_pool = torch.unsqueeze(max_pool, 1)
        ave_pool = torch.unsqueeze(ave_pool, 1)

        cat = mycat(max_pool, ave_pool)
        att = self.conv(cat)
        att = self.sigmoid(att)
        att = self.broadcast(att)
        # print(f"x.shape: {x.shape}")
        # print(f"att.shape: {att.shape}")

        out = att * x
        return out


class DenseLoop(nn.Sequential):
    def __init__(self, in_chn, chn1, chn2, nb_loops):
        super().__init__()
        all_filters = in_chn + chn2 * nb_loops
        out_filters = int(all_filters / 2)

        for i in range(nb_loops):
            if i == 0:
                _in_chn = in_chn  # first convolution accept in_chn channels
            else:
                _in_chn = _in_chn + chn2  # afterwards, convolution accepts concatenated channels
            self.add_module("block" + str(i), DenseSubModule(_in_chn, chn1, chn2))
            if i == nb_loops - 1:
                self.add_module("pixel_attention", Pixel_attention(all_filters))
                self.add_module("transition_layer", Convolution(dimensions=3,
                                                                in_channels=all_filters,
                                                                out_channels=out_filters,
                                                                kernel_size=3,
                                                                act=(
                                                                    "LeakyReLU",
                                                                    {"negative_slope": 0.1, "inplace": True}),
                                                                norm=("instance", {"affine": True}),
                                                                dropout=0.1))


class Down_sahar(nn.Module):
    def __init__(self, chn):
        super().__init__()

        self.conv = Convolution(dimensions=3,
                                in_channels=chn,
                                out_channels=chn,
                                kernel_size=1,
                                act=("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
                                norm=("instance", {"affine": True}),
                                dropout=0)

        self.max_pooling = Pool["MAX", 3](kernel_size=2)
        # self.layer1 = Cov


    def forward(self, x):
            
        out = self.max_pooling(self.conv(x))
        return out


class Up_sahar(nn.Module):
    def __init__(self, chn):
        super().__init__()

        self.conv = Convolution(dimensions=3,
                                in_channels=chn,
                                out_channels=chn,
                                kernel_size=3,
                                act=("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
                                norm=("instance", {"affine": True}),
                                dropout=0)
        self.deconv = UpSample(3, chn, chn, 2, mode="deconv")

    def forward(self, x):
        out = self.deconv(self.conv(x))
        return out


class ChAtt_sahar(nn.Module):
    def __init__(self, input_chn):
        super().__init__()

        self.maxpool = nn.AdaptiveMaxPool3d((1, 1, 1))
        self.linear1 = nn.Linear(input_chn, input_chn)
        self.linear2 = nn.Linear(input_chn, input_chn)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.maxpool(x)  # (batch_size, chn, 1, 1, 1)
        out = out.squeeze(2)  # (batch_size, chn, 1, 1)
        out = out.squeeze(2)  # (batch_size, chn,  1)
        out = out.squeeze(2)  # (batch_size, chn)

        out = self.linear1(out)  # (batch_size, chn)
        out = self.linear2(out)  # (batch_size, chn)

        out = torch.unsqueeze(out, -1)  # (batch_size, chn,  1)
        out = torch.unsqueeze(out, -1)  # (batch_size, chn, 1, 1)
        out = torch.unsqueeze(out, -1)  # (batch_size, chn, 1, 1, 1)

        out = self.sigmoid(out)

        out = x * out

        return out


class Saharnet_decoder(nn.Module):
    def __init__(self, out_channels=2):
        super().__init__()
        nb_filters = 32
        self.nb_loops = [2, 2, 2, 2, 2]
        self.f2 = [8, 16, 32, 16, 8]
        self.f1 = [i * 4 for i in self.f2]  # bottlenet channels number is 4 times. https://arxiv.org/pdf/2001.02394.pdf

        self.ft0 = get_transition_filters(nb_filters * 2, self.f2[0], self.nb_loops[0])
        self.ft1 = get_transition_filters(self.ft0, self.f2[1], self.nb_loops[1])
        self.ft2 = get_transition_filters(self.ft1, self.f2[2], self.nb_loops[2])
        self.ft3 = get_transition_filters(self.ft2 + self.ft1, self.f2[3], self.nb_loops[3])
        self.ft4 = get_transition_filters(self.ft3 + self.ft0, self.f2[4], self.nb_loops[4])
        print("ft:", self.ft0, self.ft1, self.ft2, self.ft3, self.ft4)

        self.up0 = Up_sahar(self.ft2)
        self.ch_att0 = ChAtt_sahar(self.ft1)
        self.denseloop3 = DenseLoop(self.ft2 + self.ft1, self.f1[3], self.f2[3], self.nb_loops[3])

        self.up1 = Up_sahar(self.ft3)
        self.ch_att1 = ChAtt_sahar(self.ft0)
        self.denseloop4 = DenseLoop(self.ft3 + self.ft0, self.f1[4], self.f2[4], self.nb_loops[4])

        self.conv_4 = Myconvolution(self.ft4, self.f2[4])
        self.final_conv = Conv["conv", 3](self.f2[4], out_channels, kernel_size=1)

    def forward(self, dl0, dl1, dl2):
        # decoder part
        up0 = self.up0(dl2)
        # print(f"up0.shape: {up0.shape}")
        at0 = self.ch_att0(dl1)
        # print(f"at0.shape: {at0.shape}")

        ct0 = mycat(up0, at0)
        # print(f"ct0.shape: {ct0.shape}")

        dl3 = self.denseloop3(ct0)
        # print(f"dl3.shape: {dl3.shape}")

        up1 = self.up1(dl3)
        # print(f"up1.shape: {up1.shape}")
        at1 = self.ch_att1(dl0)
        # print(f"at1.shape: {at1.shape}")
        ct1 = mycat(up1, at1)
        # print(f"ct1.shape: {ct1.shape}")
        dl4 = self.denseloop4(ct1)
        # print(f"dl4.shape: {dl4.shape}")

        c4 = self.conv_4(dl4)
        # print(f"c4.shape: {c4.shape}")

        out = self.final_conv(c4)
        # print(f"out.shape: {out.shape}")

        return out


def get_transition_filters(in_chn, chn2, nb_loops):
    all_filters = in_chn + chn2 * nb_loops
    out_filters = int(all_filters / 2)
    return out_filters

class SaharAbsNet(nn.Module):
    def __init__(self):
        super().__init__()

class Saharnet_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        nb_filters = 32
        self.nb_loops = [2, 2, 2, 2, 2]
        self.f2 = [8, 16, 32, 16, 8]
        self.f1 = [i * 4 for i in self.f2]  # bottlenet channels number is 4 times. https://arxiv.org/pdf/2001.02394.pdf

        self.ft0 = get_transition_filters(nb_filters * 2, self.f2[0], self.nb_loops[0])
        self.ft1 = get_transition_filters(self.ft0, self.f2[1], self.nb_loops[1])
        self.ft2 = get_transition_filters(self.ft1, self.f2[2], self.nb_loops[2])
        self.ft3 = get_transition_filters(self.ft2, self.f2[3], self.nb_loops[3])
        self.ft4 = get_transition_filters(self.ft3, self.f2[4], self.nb_loops[4])

        self.conv_1 = Myconvolution(1, nb_filters)
        self.conv_2 = Myconvolution(nb_filters, nb_filters)
        self.conv_3 = Myconvolution(nb_filters * 2, nb_filters)

        self.denseloop0 = DenseLoop(nb_filters * 2, self.f1[0], self.f2[0], self.nb_loops[0])
        self.down0 = Down_sahar(self.ft0)

        self.denseloop1 = DenseLoop(self.ft0, self.f1[1], self.f2[1], self.nb_loops[1])
        self.down1 = Down_sahar(self.ft1)

        self.denseloop2 = DenseLoop(self.ft1, self.f1[2], self.f2[2], self.nb_loops[2])

    def forward(self, x):
        # encoder part
        c1 = self.conv_1(x)  # out chn = 8
        c2 = self.conv_2(c1)  # out chn = 8
        ct = mycat(c1, c2)  # out chn = 16
        c3 = self.conv_3(ct)  # out chn = 8
        cat2 = mycat(c1, c3)  # out chn = 16

        dl0 = self.denseloop0(cat2)  # out chn = 32
        dw0 = self.down0(dl0)

        dl1 = self.denseloop1(dw0)
        dw1 = self.down1(dl1)

        dl2 = self.denseloop2(dw1)
        # print(f"denseloop0.shape:{dl0.shape}")
        # print(f"down0.shape:{dw0.shape}")
        # print(f"denseloop1.shape:{dl1.shape}")
        # print(f"down1.shape:{dw1.shape}")
        # print(f"denseloop2.shape:{dl2.shape}")

        return dl0, dl1, dl2
