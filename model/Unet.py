import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from thop import profile

class double_conv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.dconv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )
    def forward(self, x):
        x = self.dconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, mode='bilinear', size=False):
        super(up, self).__init__()
        if mode == 'bilinear':
            if not size:
                self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            else:
                 self.up = nn.Upsample(size=size, mode='bilinear', align_corners=True)               
        elif mode == 'pixelshuffle':
            self.up = nn.Sequential(nn.Conv2d(in_ch, 4*in_ch, kernel_size=3, padding=1, dilation=1),
              			nn.PixelShuffle(2))
        elif mode == 'ConvTrans':
            self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)
        elif mode == 'CARA':
            self.up = CARAFE(in_ch)

    def forward(self, x):
        x = self.up(x)
        return x

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(channels,channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Unet(nn.Module):#U-NET模型
    def __init__(self, in_channel, n_class, channel_reduction=2, aux=False):
        super().__init__()
        self.aux = aux
        channels = [64, 128, 256, 512, 1024]
        channels = [int(c / channel_reduction) for c in channels]

        self.donv1 = double_conv(in_channel, channels[0])
        self.donv2 = double_conv(channels[0], channels[1])
        self.donv3 = double_conv(channels[1], channels[2])
        self.donv4 = double_conv(channels[2], channels[3])
        self.down_pool = nn.MaxPool2d(kernel_size=2)

        self.donv_mid = double_conv(channels[3], channels[3])

        self.donv5 = double_conv(channels[4], channels[2])
        self.donv6 = double_conv(channels[3], channels[1])
        self.donv7 = double_conv(channels[2], channels[0])
        self.donv8 = double_conv(channels[1], channels[0])

        self.out_conv = nn.Conv2d(channels[0], n_class, kernel_size=1)


    def forward(self, x):
        outputs = dict()
        size = x.size()[2:]

        x1 = self.donv1(x)
        x2 = self.donv2(self.down_pool(x1))
        x3 = self.donv3(self.down_pool(x2))
        x4 = self.donv4(self.down_pool(x3))
        x_mid = self.donv_mid(self.down_pool(x4))
        att_map = [x1.mean(1), x3.mean(1), x_mid.mean(1)]
        x = F.interpolate(x_mid, size=x4.size()[-2:], mode="bilinear", align_corners=True)
        x = self.donv5(torch.cat((x, x4), dim=1))

        x = F.interpolate(x, size=x3.size()[-2:], mode="bilinear", align_corners=True)
        x = self.donv6(torch.cat((x, x3), dim=1))

        x = F.interpolate(x, size=x2.size()[-2:], mode="bilinear", align_corners=True)
        x = self.donv7(torch.cat((x, x2), dim=1))

        x = F.interpolate(x, size=x1.size()[-2:], mode="bilinear", align_corners=True)
        x = self.donv8(torch.cat((x, x1), dim=1))

        x = self.out_conv(x)
        outputs.update({'att': att_map})
        outputs.update({"main_out": x})



        return outputs

###
class Unet_(nn.Module):#U-NET模型
    def __init__(self, in_channel, n_class, channel_reduction=1, aux=False):
        super().__init__()
        self.aux = aux
        channels = [64, 128, 256, 512, 1024]
        channels = [int(c / channel_reduction) for c in channels]

        self.donv1 = double_conv(in_channel, channels[0])
        self.donv2 = double_conv(channels[0], channels[1])
        self.donv3 = double_conv(channels[1], channels[2])
        self.donv4 = double_conv(channels[2], channels[3])
        self.down_pool = nn.MaxPool2d(kernel_size=2)
        self.donv_mid = double_conv(channels[3], channels[3])

        self.donv12 = double_conv(in_channel, channels[0])
        self.donv22 = double_conv(channels[0], channels[1])
        self.donv32 = double_conv(channels[1], channels[2])
        self.donv42 = double_conv(channels[2], channels[3])
        self.down_pool2 = nn.MaxPool2d(kernel_size=2)
        self.donv_mid2 = double_conv(channels[3], channels[3])

        self.conv1 = nn.Conv2d(2*channels[0], channels[0], 1, stride=1, bias=False)
        self.conv2 = nn.Conv2d(2*channels[1], channels[1],  1, stride=1, bias=False)
        self.conv3 = nn.Conv2d(2*channels[2], channels[2],  1, stride=1, bias=False)
        self.conv4 = nn.Conv2d(2*channels[3], channels[3],  1, stride=1, bias=False)
        self.conv5 = nn.Conv2d(2*channels[3], channels[3],  1, stride=1, bias=False)

        self.donv5 = double_conv(channels[4], channels[2])
        self.donv6 = double_conv(channels[3], channels[1])
        self.donv7 = double_conv(channels[2], channels[0])
        self.donv8 = double_conv(channels[1], channels[0])

        self.out_conv_1 = nn.Conv2d(channels[0], n_class, kernel_size=1)

    def forward(self, x):
        outputs = dict()
        x1 = x['image']
        size = x1.size()[2:]
        x2 = x['aux'] 
        #x2 = x1
        x11 = self.donv1(x1)

        x21 = self.donv2(self.down_pool(x11))

        x31 = self.donv3(self.down_pool(x21))

        x41 = self.donv4(self.down_pool(x31))

        x_mid1 = self.donv_mid(self.down_pool(x41))

        x12 = self.donv12(x2)

        x22 = self.donv22(self.down_pool2(x12))

        x32 = self.donv32(self.down_pool2(x22))

        x42 = self.donv42(self.down_pool2(x32))

        x_mid2 = self.donv_mid2(self.down_pool2(x42))

        x1 = self.conv1(torch.cat((x11, x12), dim=1))
        x2 = self.conv2(torch.cat((x21, x22), dim=1))
        x3 = self.conv3(torch.cat((x31, x32), dim=1))
        x4 = self.conv4(torch.cat((x41, x42), dim=1))
        x_mid = self.conv5(torch.cat((x_mid1, x_mid2), dim=1))

        x = F.interpolate(x_mid, size=x4.size()[-2:], mode="bilinear", align_corners=True)
        x = self.donv5(torch.cat((x, x4), dim=1))

        x = F.interpolate(x, size=x3.size()[-2:], mode="bilinear", align_corners=True)
        x = self.donv6(torch.cat((x, x3), dim=1))

        x = F.interpolate(x, size=x2.size()[-2:], mode="bilinear", align_corners=True)
        x = self.donv7(torch.cat((x, x2), dim=1))

        x = F.interpolate(x, size=x1.size()[-2:], mode="bilinear", align_corners=True)
        x = self.donv8(torch.cat((x, x1), dim=1))

        x = self.out_conv_(x)
        outputs.update({"main_out": x})



        return outputs




if __name__ == "__main__":
    net = Unet(1, 2)
    inputs = torch.randn(1, 1, 224, 224)
    flops, params = profile(net, (inputs,))
    print('flops: ', flops, 'params: ', params)
    print("%.2f | %.2f" % (params / (1000 ** 2), flops / (1000 ** 3)))#这里除以1000的平方，是为了化成M的单位
























