import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from timm.models.layers import trunc_normal_

from thop import profile

class CB(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.dconv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.Dropout(0.2),
            nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x):
        x = self.dconv(x)
        return x

class PoolAttention(nn.Module):
    def __init__(self, in_channel, kernel, reduction=4):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d((kernel, kernel))
        self.conv1 = nn.Conv2d(in_channel, in_channel//reduction, kernel_size=kernel, stride=1, padding=kernel//2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channel//reduction, in_channel, kernel_size=kernel, stride=1, padding=kernel//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        tmp = x
        x = self.pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)

        x = F.interpolate(x, size=tmp.size()[-2], mode='bilinear')
        x = self.sigmoid(x)

        return x*tmp

class PSE(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        self.Path1 = PoolAttention(in_channel, kernel=1)
        self.Path2 = PoolAttention(in_channel, kernel=2)
        self.Path3 = PoolAttention(in_channel, kernel=3)
        self.out_conv = nn.Sequential(
            nn.Conv2d(3*in_channel, in_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channel),
        )
    
    def forward(self, x):
        tmp = x
        x = torch.cat((self.Path1(x), self.Path2(x), self.Path3(x)), dim=1)
        x = self.out_conv(x)
        assert x.size() == tmp.size(), "input size mismatch output in pse"
        return x

class C2F(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        self.conv_bn = nn.Sequential(
            nn.Conv2d(2*in_channel, in_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channel),
        )

        self.cb1 = CB(2*in_channel, in_channel)
        self.pse1 = PSE(in_channel)

        self.cb2 = CB(2*in_channel, in_channel)
        self.pse2 = PSE(in_channel)
   
    def forward(self, x1, x2):
        x2 = self.conv_bn(x2)
        assert x1.size() == x2.size(), "coarse fine feature size mismatch"
        res = x2 - x1
        x = self.cb1(torch.cat((x1, x2), dim=1))
        x = self.pse1(x)
        x = torch.cat((x, res), dim=1)
        x = self.cb2(x)
        x = self.pse2(x)

        return x

class ConvBN(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.conv_bn = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )

    def forward(self, x):
        x = self.conv_bn(x)
        return x

class OL(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        self.ol = nn.Sequential(
            nn.Conv2d(in_channel, 1, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.ol(x)
        return x

class MFINet(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        channels = [64, 128, 256]
        self.b1 = nn.Sequential(
            CB(in_channel, channels[0]),
            PSE(channels[0])
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.b2 = nn.Sequential(
            CB(channels[0], channels[1]),
            PSE(channels[1])
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.b3 = nn.Sequential(
            CB(channels[1], channels[2]),
            PSE(channels[2])
        )

        self.c2f_1 = C2F(channels[0])
        self.c2f_2 = C2F(channels[1])

        self.conv_bn_2 = ConvBN(channels[1], channels[0])
        self.conv_bn_3 = ConvBN(channels[2], channels[0])

        self.ol1 = OL(3*channels[0])
        self.ol2 = OL(channels[1])
        self.ol3 = OL(channels[2])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02) 
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        outputs = []
        size = x.size()[2:]
        x1 = self.b1(x)

        x2 = self.pool1(x1)
        x2 = self.b2(x2)

        x3 = self.pool2(x2)
        x3 = self.b3(x3)

        c3 = x3
        c2 = self.c2f_2(x2, F.interpolate(c3, size=x2.size()[-2:], mode="bilinear", align_corners=True))
        c1 = self.c2f_1(x1, F.interpolate(c2, size=x1.size()[-2:], mode="bilinear", align_corners=True))

        c3_cat = self.conv_bn_3(c3)
        c2_cat = self.conv_bn_2(c2)

        c1_ol = self.ol1(torch.cat((c1, F.interpolate(c2_cat, size=c1.size()[-2:], mode="bilinear", align_corners=True), 
                                    F.interpolate(c3_cat, size=c1.size()[-2:], mode="bilinear", align_corners=True)), dim=1))
        c2_ol = self.ol2(c2)
        c3_ol = self.ol3(c3)

        outputs = [c1_ol, c2_ol, c3_ol]

        return outputs

if __name__ == "__main__":
    net = MFINet(in_channel=1)
    inputs = torch.randn(1, 1, 224, 224)
    flops, params = profile(net, (inputs,))
    # print('flops: ', flops, 'params: ', params)
    print('params: ', params, 'flops: ', flops, )

    print("%.2f | %.2f" % (params / (1000 ** 2), flops / (1000 ** 3)))#这里除以1000的平方，是为了化成M的单位
    inputs = torch.randn(2, 128, 224, 224)
    # x2 = torch.randn(2, 1, 91, 91)
    # net = MFINet(in_channel=1)
    # out = net(x2)
    # for i in out:
    #     print(i.size())






















