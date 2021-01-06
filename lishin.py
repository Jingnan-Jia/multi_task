# -*- coding: utf-8 -*-
# @Time    : 12/1/20 9:20 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
class Unet(nn.Module):
    def successive_convs_pad(self, in_channels, out_channels, kernel_size=3):
        block = torch.nn.Sequential(
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1),
                    torch.nn.BatchNorm2d(out_channels),
                    torch.nn.LeakyReLU(),
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels, padding=1),
                    torch.nn.BatchNorm2d(out_channels),
                    torch.nn.LeakyReLU(),
                )
        return block
    def concat(self, upsampled, bypass):
        return torch.cat((upsampled, bypass), 1)
    def __init__(self, in_channel, out_channel, start_channel=64):
        super(Unet, self).__init__()
        #Encode
        c = start_channel
        self.encode1 = self.successive_convs_pad(in_channels=in_channel, out_channels=c)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.encode2 = self.successive_convs_pad(c, 2*c)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.encode3 = self.successive_convs_pad(2*c, 4*c)
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.encode4 = self.successive_convs_pad(4*c, 8*c)
        self.maxpool4 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        # Bottleneck
        self.bottleneck = self.successive_convs_pad(8*c, 16*c)
        self.adpool = torch.nn.AdaptiveAvgPool2d((1,1))
        self.fc = torch.nn.Linear(16*c, 1)
        # Decode
        self.upconv4 = torch.nn.ConvTranspose2d(in_channels=16*c, out_channels=8*c, kernel_size=2, stride=2)
        self.decode4 = self.successive_convs_pad(16*c, 8*c)
        self.upconv3 = torch.nn.ConvTranspose2d(in_channels=8*c, out_channels=4*c, kernel_size=2, stride=2)
        self.decode3 = self.successive_convs_pad(8*c, 4*c)
        self.upconv2 = torch.nn.ConvTranspose2d(in_channels=4*c, out_channels=2*c, kernel_size=2, stride=2)
        self.decode2 = self.successive_convs_pad(4*c, 2*c)
        self.upconv1 = torch.nn.ConvTranspose2d(in_channels=2*c, out_channels=c, kernel_size=2, stride=2)
        self.decode1 = self.successive_convs_pad(2*c, c)
        self.output_logit = torch.nn.Conv2d(kernel_size=1, in_channels=c, out_channels=out_channel)
        self.output  = torch.nn.Softmax(1)
    def forward(self, x):
        # Encode
        encode1 = self.encode1(x)
        maxpool1 = self.maxpool1(encode1)
        encode2 = self.encode2(maxpool1)
        maxpool2 = self.maxpool2(encode2)
        encode3 = self.encode3(maxpool2)
        maxpool3 = self.maxpool3(encode3)
        encode4 = self.encode4(maxpool3)
        maxpool4 = self.maxpool4(encode4)
        # Bottleneck
        bottleneck = self.bottleneck(maxpool4)
        adpool = self.adpool(bottleneck)
        adpool = adpool.squeeze(3)
        adpool = adpool.squeeze(2)
        output_volume = self.fc(adpool)
        # Decode
        upconv4 = self.upconv4(bottleneck)
        cat4 = self.concat(upconv4, encode4)
        decode4 = self.decode4(cat4)
        upconv3 = self.upconv3(decode4)
        cat3 = self.concat(upconv3, encode3)
        decode3 = self.decode3(cat3)
        upconv2 = self.upconv2(decode3)
        cat2 = self.concat(upconv2, encode2)
        decode2 = self.decode2(cat2)
        upconv1 = self.upconv1(decode2)
        cat1 = self.concat(upconv1, encode1)
        decode1 = self.decode1(cat1)
        output_logit = self.output_logit(decode1)
        output = self.output(output_logit)
        return  output, output_logit, output_volume, adpool