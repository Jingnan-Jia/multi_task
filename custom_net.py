# -*- coding: utf-8 -*-
# @Time    : 11/20/20 10:39 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
from typing import Sequence, Union

import torch
import torch.nn as nn

from monai.networks.blocks import Convolution, UpSample
from monai.networks.layers.factories import Conv, Pool
from monai.utils import ensure_tuple_rep
from monai.networks.nets.basic_unet import TwoConv, Down, UpCat


class Up(nn.Module):
    """upsampling, concatenation with the encoder feature map, two convolutions"""

    def __init__(
        self,
        dim: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        halves: bool = True,
    ):
        """
        Args:
            dim: number of spatial dimensions.
            in_chns: number of input channels to be upsampled.
            cat_chns: number of channels from the decoder.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            dropout: dropout ratio. Defaults to no dropout.
            upsample: upsampling mode, available options are
                ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.
            halves: whether to halve the number of channels during upsampling.
        """
        super().__init__()

        up_chns = in_chns // 2 if halves else in_chns
        self.upsample = UpSample(dim, in_chns, up_chns, 2, mode=upsample)
        self.convs = TwoConv(dim, up_chns, out_chns, act, norm, dropout)

    def forward(self, x: torch.Tensor):
        """

        Args:
            x: features to be upsampled.
        """
        x_0 = self.upsample(x)
        x = self.convs(x_0)  # input channels: (cat_chns)
        return x


class Encoder(nn.Module):
    def __init__(
            self,
            dimensions: int = 3,
            in_channels: int = 1,
            features: Sequence[int] = (32, 32, 64, 128, 256, 32),
            act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
            norm: Union[str, tuple] = ("instance", {"affine": True}),
            dropout: Union[float, tuple] = 0.0,
    ):
        """
        A UNet implementation with 1D/2D/3D supports.

        Based on:

            Falk et al. "U-Net – Deep Learning for Cell Counting, Detection, and
            Morphometry". Nature Methods 16, 67–70 (2019), DOI:
            http://dx.doi.org/10.1038/s41592-018-0261-2

        Args:
            dimensions: number of spatial dimensions. Defaults to 3 for spatial 3D inputs.
            in_channels: number of input channels. Defaults to 1.
            features: six integers as numbers of features.
                Defaults to ``(32, 32, 64, 128, 256, 32)``,

                - the first five values correspond to the five-level encoder feature sizes.
                - the last value corresponds to the feature size after the last upsampling.

            act: activation type and arguments. Defaults to LeakyReLU.
            norm: feature normalization type and arguments. Defaults to instance norm.
            dropout: dropout ratio. Defaults to no dropout.
            upsample: upsampling mode, available options are
                ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.

        Examples::

            # for spatial 2D
            >>> net = BasicUNet(dimensions=2, features=(64, 128, 256, 512, 1024, 128))

            # for spatial 2D, with group norm
            >>> net = BasicUNet(dimensions=2, features=(64, 128, 256, 512, 1024, 128), norm=("group", {"num_groups": 4}))

            # for spatial 3D
            >>> net = BasicUNet(dimensions=3, features=(32, 32, 64, 128, 256, 32))

        See Also

            - :py:class:`monai.networks.nets.DynUNet`
            - :py:class:`monai.networks.nets.UNet`

        """
        super().__init__()

        fea = ensure_tuple_rep(features, 6)
        print(f"BasicUNet features: {fea}.")

        self.conv_0 = TwoConv(dimensions, in_channels, features[0], act, norm, dropout)
        self.down_1 = Down(dimensions, fea[0], fea[1], act, norm, dropout)
        self.down_2 = Down(dimensions, fea[1], fea[2], act, norm, dropout)
        self.down_3 = Down(dimensions, fea[2], fea[3], act, norm, dropout)
        self.down_4 = Down(dimensions, fea[3], fea[4], act, norm, dropout)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: input should have spatially N dimensions
                ``(Batch, in_channels, dim_0[, dim_1, ..., dim_N])``, N is defined by `dimensions`.
                It is recommended to have ``dim_n % 16 == 0`` to ensure all maxpooling inputs have
                even edge lengths.

        Returns:
            A torch Tensor of "raw" predictions in shape
            ``(Batch, out_channels, dim_0[, dim_1, ..., dim_N])``.
        """
        x0 = self.conv_0(x)

        x1 = self.down_1(x0)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        x4 = self.down_4(x3)


        return x0, x1, x2, x3, x4


class DecoderSegRec(nn.Module):
    def __init__(
            self,
            dimensions: int = 3,
            out_channels: int = 1,
            features: Sequence[int] = (32, 32, 64, 128, 256, 32),
            act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
            norm: Union[str, tuple] = ("instance", {"affine": True}),
            dropout: Union[float, tuple] = 0.0,
            upsample: str = "deconv"
    ):
        """
        A UNet implementation with 1D/2D/3D supports.

        Based on:

            Falk et al. "U-Net – Deep Learning for Cell Counting, Detection, and
            Morphometry". Nature Methods 16, 67–70 (2019), DOI:
            http://dx.doi.org/10.1038/s41592-018-0261-2

        Args:
            dimensions: number of spatial dimensions. Defaults to 3 for spatial 3D inputs.
            in_channels: number of input channels. Defaults to 1.
            out_channels: number of output channels. Defaults to 2.
            features: six integers as numbers of features.
                Defaults to ``(32, 32, 64, 128, 256, 32)``,

                - the first five values correspond to the five-level encoder feature sizes.
                - the last value corresponds to the feature size after the last upsampling.

            act: activation type and arguments. Defaults to LeakyReLU.
            norm: feature normalization type and arguments. Defaults to instance norm.
            dropout: dropout ratio. Defaults to no dropout.
            upsample: upsampling mode, available options are
                ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.

        Examples::

            # for spatial 2D
            >>> net = BasicUNet(dimensions=2, features=(64, 128, 256, 512, 1024, 128))

            # for spatial 2D, with group norm
            >>> net = BasicUNet(dimensions=2, features=(64, 128, 256, 512, 1024, 128), norm=("group", {"num_groups": 4}))

            # for spatial 3D
            >>> net = BasicUNet(dimensions=3, features=(32, 32, 64, 128, 256, 32))

        See Also

            - :py:class:`monai.networks.nets.DynUNet`
            - :py:class:`monai.networks.nets.UNet`

        """
        super().__init__()
        fea = ensure_tuple_rep(features, 6)
        print(f"BasicUNet features: {fea}.")

        self.upcat_4 = UpCat(dimensions, fea[4], fea[3], fea[3], act, norm, dropout, upsample)
        self.upcat_3 = UpCat(dimensions, fea[3], fea[2], fea[2], act, norm, dropout, upsample)
        self.upcat_2 = UpCat(dimensions, fea[2], fea[1], fea[1], act, norm, dropout, upsample)
        self.upcat_1 = UpCat(dimensions, fea[1], fea[0], fea[5], act, norm, dropout, upsample, halves=False)

        self.up_4 = Up(dimensions, fea[4], fea[3], act, norm, dropout, upsample)
        self.up_3 = Up(dimensions, fea[3], fea[2], act, norm, dropout, upsample)
        self.up_2 = Up(dimensions, fea[2], fea[1], act, norm, dropout, upsample)
        self.up_1 = Up(dimensions, fea[1], fea[5], act, norm, dropout, upsample, halves=False)

        self.final_conv = Conv["conv", dimensions](fea[5], out_channels, kernel_size=1)
        self.final_conv1 = Conv["conv", dimensions](fea[5], out_channels, kernel_size=1)

    def forward(self,
                x0: torch.Tensor,
                x1: torch.Tensor,
                x2: torch.Tensor,
                x3: torch.Tensor,
                x4: torch.Tensor):
        """
        Args:
            x: input should have spatially N dimensions
                ``(Batch, in_channels, dim_0[, dim_1, ..., dim_N])``, N is defined by `dimensions`.
                It is recommended to have ``dim_n % 16 == 0`` to ensure all maxpooling inputs have
                even edge lengths.

        Returns:
            A torch Tensor of "raw" predictions in shape
            ``(Batch, out_channels, dim_0[, dim_1, ..., dim_N])``.
        """

        u4 = self.upcat_4(x4, x3)
        u3 = self.upcat_3(u4, x2)
        u2 = self.upcat_2(u3, x1)
        u1 = self.upcat_1(u2, x0)

        u_out = self.final_conv(u1)

        if self.training:
            r4 = self.up_4(x4)
            r3 = self.up_3(r4)
            r2 = self.up_2(r3)
            r1 = self.up_1(r2)
            r_out = self.final_conv1(r1)

            
            return u_out, r_out

        return u_out


class DecoderSeg(nn.Module):
    def __init__(
            self,
            dimensions: int = 3,
            out_channels: int = 1,
            features: Sequence[int] = (32, 32, 64, 128, 256, 32),
            act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
            norm: Union[str, tuple] = ("instance", {"affine": True}),
            dropout: Union[float, tuple] = 0.0,
            upsample: str = "deconv",
    ):
        """
        A UNet implementation with 1D/2D/3D supports.

        Based on:

            Falk et al. "U-Net – Deep Learning for Cell Counting, Detection, and
            Morphometry". Nature Methods 16, 67–70 (2019), DOI:
            http://dx.doi.org/10.1038/s41592-018-0261-2

        Args:
            dimensions: number of spatial dimensions. Defaults to 3 for spatial 3D inputs.
            in_channels: number of input channels. Defaults to 1.
            out_channels: number of output channels. Defaults to 2.
            features: six integers as numbers of features.
                Defaults to ``(32, 32, 64, 128, 256, 32)``,

                - the first five values correspond to the five-level encoder feature sizes.
                - the last value corresponds to the feature size after the last upsampling.

            act: activation type and arguments. Defaults to LeakyReLU.
            norm: feature normalization type and arguments. Defaults to instance norm.
            dropout: dropout ratio. Defaults to no dropout.
            upsample: upsampling mode, available options are
                ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.

        Examples::

            # for spatial 2D
            >>> net = BasicUNet(dimensions=2, features=(64, 128, 256, 512, 1024, 128))

            # for spatial 2D, with group norm
            >>> net = BasicUNet(dimensions=2, features=(64, 128, 256, 512, 1024, 128), norm=("group", {"num_groups": 4}))

            # for spatial 3D
            >>> net = BasicUNet(dimensions=3, features=(32, 32, 64, 128, 256, 32))

        See Also

            - :py:class:`monai.networks.nets.DynUNet`
            - :py:class:`monai.networks.nets.UNet`

        """
        super().__init__()
        fea = ensure_tuple_rep(features, 6)
        print(f"BasicUNet features: {fea}.")

        self.upcat_4 = UpCat(dimensions, fea[4], fea[3], fea[3], act, norm, dropout, upsample)
        self.upcat_3 = UpCat(dimensions, fea[3], fea[2], fea[2], act, norm, dropout, upsample)
        self.upcat_2 = UpCat(dimensions, fea[2], fea[1], fea[1], act, norm, dropout, upsample)
        self.upcat_1 = UpCat(dimensions, fea[1], fea[0], fea[5], act, norm, dropout, upsample, halves=False)

        self.final_conv = Conv["conv", dimensions](fea[5], out_channels, kernel_size=1)


    def forward(self,
                x0: torch.Tensor,
                x1: torch.Tensor,
                x2: torch.Tensor,
                x3: torch.Tensor,
                x4: torch.Tensor):
        """
        Args:
            x: input should have spatially N dimensions
                ``(Batch, in_channels, dim_0[, dim_1, ..., dim_N])``, N is defined by `dimensions`.
                It is recommended to have ``dim_n % 16 == 0`` to ensure all maxpooling inputs have
                even edge lengths.

        Returns:
            A torch Tensor of "raw" predictions in shape
            ``(Batch, out_channels, dim_0[, dim_1, ..., dim_N])``.
        """

        u4 = self.upcat_4(x4, x3)
        u3 = self.upcat_3(u4, x2)
        u2 = self.upcat_2(u3, x1)
        u1 = self.upcat_1(u2, x0)

        u_out = self.final_conv(u1)

        return u_out


class DecoderRec(nn.Module):
    def __init__(
            self,
            dimensions: int = 3,
            out_channels: int = 1,
            features: Sequence[int] = (32, 32, 64, 128, 256, 32),
            act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
            norm: Union[str, tuple] = ("instance", {"affine": True}),
            dropout: Union[float, tuple] = 0.0,
            upsample: str = "deconv",
    ):
        """
        A UNet implementation with 1D/2D/3D supports.

        Based on:

            Falk et al. "U-Net – Deep Learning for Cell Counting, Detection, and
            Morphometry". Nature Methods 16, 67–70 (2019), DOI:
            http://dx.doi.org/10.1038/s41592-018-0261-2

        Args:
            dimensions: number of spatial dimensions. Defaults to 3 for spatial 3D inputs.
            in_channels: number of input channels. Defaults to 1.
            out_channels: number of output channels. Defaults to 2.
            features: six integers as numbers of features.
                Defaults to ``(32, 32, 64, 128, 256, 32)``,

                - the first five values correspond to the five-level encoder feature sizes.
                - the last value corresponds to the feature size after the last upsampling.

            act: activation type and arguments. Defaults to LeakyReLU.
            norm: feature normalization type and arguments. Defaults to instance norm.
            dropout: dropout ratio. Defaults to no dropout.
            upsample: upsampling mode, available options are
                ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.

        Examples::

            # for spatial 2D
            >>> net = BasicUNet(dimensions=2, features=(64, 128, 256, 512, 1024, 128))

            # for spatial 2D, with group norm
            >>> net = BasicUNet(dimensions=2, features=(64, 128, 256, 512, 1024, 128), norm=("group", {"num_groups": 4}))

            # for spatial 3D
            >>> net = BasicUNet(dimensions=3, features=(32, 32, 64, 128, 256, 32))

        See Also

            - :py:class:`monai.networks.nets.DynUNet`
            - :py:class:`monai.networks.nets.UNet`

        """
        super().__init__()

        fea = ensure_tuple_rep(features, 6)
        print(f"BasicUNet features: {fea}.")

        self.up_4 = Up(dimensions, fea[4], fea[3], act, norm, dropout, upsample)
        self.up_3 = Up(dimensions, fea[3], fea[2], act, norm, dropout, upsample)
        self.up_2 = Up(dimensions, fea[2], fea[1], act, norm, dropout, upsample)
        self.up_1 = Up(dimensions, fea[1], fea[5], act, norm, dropout, upsample, halves=False)

        self.final_conv = Conv["conv", dimensions](fea[5], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: input should have spatially N dimensions
                ``(Batch, in_channels, dim_0[, dim_1, ..., dim_N])``, N is defined by `dimensions`.
                It is recommended to have ``dim_n % 16 == 0`` to ensure all maxpooling inputs have
                even edge lengths.

        Returns:
            A torch Tensor of "raw" predictions in shape
            ``(Batch, out_channels, dim_0[, dim_1, ..., dim_N])``.
        """

        r4 = self.up_4(x)
        r3 = self.up_3(r4)
        r2 = self.up_2(r3)
        r1 = self.up_1(r2)
        r_out = self.final_conv(r1)
        return r_out


class EnsembleEncSeg(nn.Module):
    def __init__(self, enc, dec,):
        super().__init__()
        self.enc = enc
        self.dec = dec

    def forward(self, x):
        x0, x1, x2, x3, x4 = self.enc(x)
        out = self.dec(x0, x1, x2, x3, x4)

        return out


class EnsembleEncRec(nn.Module):
    def __init__(self, enc, dec, ):
        super().__init__()
        self.enc = enc
        self.dec = dec

    def forward(self, x):
        x0, x1, x2, x3, x4 = self.enc(x)
        out = self.dec(x4)
        return out
    

