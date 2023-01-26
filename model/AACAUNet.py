"""
    References:
        "Multi-level Attention Network for Retinal Vessel Segmentation"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from thop import profile


class DDB(nn.Module):
    def __init__(self, in_channels, bottle=False, growth_rate=12, num_layers=4, dropout_prob=0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.num_layers = num_layers

        for i in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.BatchNorm2d(in_channels + i * growth_rate),
                nn.ReLU(),
                nn.Conv2d(in_channels + i * growth_rate, growth_rate, kernel_size=3, padding=1),
                nn.Dropout2d(dropout_prob)
            ))
        
        self.bottle = bottle

    def forward(self, x):
        tmp_feature = []
        for i, layer in enumerate(self.layers):
            new_features = layer(x)
            tmp_feature.append(new_features)
            x = torch.cat([x, new_features], dim=1)
        if self.bottle:
            x = torch.cat(tmp_feature, dim=1)
        return x

class AACA(nn.Module):
    def __init__(self, in_channels, k):
        super().__init__()
        r = (k + 1)//2
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.atrous_conv1 = nn.Conv1d(in_channels, in_channels, k, dilation=r, padding=r)
        self.relu = nn.ReLU()
        self.atrous_conv2 = nn.Conv1d(in_channels, in_channels, k, dilation=r, padding=r)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        tmp = x
        x = self.gap(x).squeeze(-1)
        x = self.atrous_conv1(x)
        x = self.relu(x)
        x = self.atrous_conv2(x)
        x = self.sigmoid(x).unsqueeze(-1)
        x = tmp * x + tmp

        return x

class AFM(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels):
        super().__init__()
        self.linear1 = nn.Conv2d(in_channels1, out_channels, 1, bias=False)
        self.linear2 = nn.Conv2d(in_channels2, out_channels, 1, bias=True)
        self.relu = nn.ReLU()
        self.linear3 = nn.Conv2d(out_channels, 1, 1, bias=True)
        self.Sigmoid = nn.Sigmoid()

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels1+in_channels2, in_channels2, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(in_channels2, in_channels2, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(in_channels2, 1, kernel_size=1, bias=False)
        )

    def forward(self, x1, x2):
        x1_tmp = x1
        x2_tmp = x2
        x1 = self.linear1(x1)
        x2 = self.linear2(x2)
        x = self.relu(x1 + x2)
        x = self.linear3(x)
        x = self.Sigmoid(x)
        x1_tmp = x * x1_tmp
        x = torch.cat((x2_tmp, x1_tmp), dim=1)
        x = self.out_conv(x)

        return x


class Pool(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        self.pool = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, in_channel, kernel_size=1, bias=False),
            nn.Dropout(0.2),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.pool(x)
        return x

class Up(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.up = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channel, out_channel, 2, stride=2)
        )

    def forward(self, x):
        x = self.up(x)
        return x


def asa(C):
    C = torch.tensor(C)
    k = torch.log2(C)/2 + 1/2
    nearest_int = torch.floor(k)
    is_even = nearest_int % 2 == 0
    k = nearest_int - is_even.to(torch.int8)
    return int(k)

class AACAUNet(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        channels = [64, 128, 256]
        self.conv_stem = nn.Conv2d(in_channel, 48, kernel_size=3, padding=1, bias=False)

        self.e1 = nn.Sequential(
            DDB(48),
            AACA(96, asa(96))
        )
        self.p1 = Pool(96)

        self.e2 = nn.Sequential(
            DDB(96),
            AACA(144, asa(144))
        )
        self.p2 = Pool(144)

        self.e3 = nn.Sequential(
            DDB(144),
            AACA(192, asa(192))
        )
        self.p3 = Pool(192)

        self.e4 = nn.Sequential(
            DDB(192),
            AACA(240, asa(240))
        )
        self.p4 = Pool(240)

        self.e5 = nn.Sequential(
            DDB(240, bottle=True),
            AACA(48, asa(48))
        )

        self.up1 = Up(48, 48)
        self.d1 = DDB(288)

        self.up2 = Up(336, 48)
        self.d2 = DDB(240)

        self.up3 = Up(288, 48)
        self.d3 = DDB(192)

        self.up4 = Up(240, 48)
        self.d4 = DDB(144)

        self.ds_conv1 = nn.Conv2d(336, 48, kernel_size=1, bias=False)
        self.ds_conv2 = nn.Conv2d(288, 48, kernel_size=1, bias=False)
        self.ds_conv3 = nn.Conv2d(240, 48, kernel_size=1, bias=False)
        self.ds_conv4 = nn.Conv2d(192, 48, kernel_size=1, bias=False)

        self.att_conv = nn.Conv2d(48*4, 48, kernel_size=1, bias=False)

        self.afm1 = AFM(48, 48, 48)
        self.afm2 = AFM(48, 48, 48)
        self.afm3 = AFM(48, 48, 48)
        self.afm4 = AFM(48, 48, 48)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.size()
        x = self.conv_stem(x)

        e1 = self.e1(x)
        e1_pool = self.p1(e1)

        e2 = self.e2(e1_pool)
        e2_pool = self.p2(e2)

        e3 = self.e3(e2_pool)
        e3_pool = self.p3(e3)

        e4 = self.e4(e3_pool)
        e4_pool = self.p4(e4)

        e5 = self.e5(e4_pool)

        d1 = self.d1(torch.cat((e4, F.interpolate(self.up1(e5), size=e4.size()[-2:], mode='bilinear')), dim=1))
        d2 = self.d2(torch.cat((e3, self.up2(d1)), dim=1))
        d3 = self.d3(torch.cat((e2, F.interpolate(self.up3(d2), size=e2.size()[-2:], mode='bilinear')), dim=1))
        d4 = self.d4(torch.cat((e1, F.interpolate(self.up4(d3), size=e1.size()[-2:], mode='bilinear')), dim=1))

        d1 = F.interpolate(self.ds_conv1(d1), size=(h, w), mode='bilinear')
        d2 = F.interpolate(self.ds_conv2(d2), size=(h, w), mode='bilinear')
        d3 = F.interpolate(self.ds_conv3(d3), size=(h, w), mode='bilinear')
        d4 = F.interpolate(self.ds_conv4(d4), size=(h, w), mode='bilinear')

        d_att = self.att_conv(torch.cat((d1, d2, d3, d4), dim=1))
        ds1 = [self.sigmoid(d1.mean(dim=1)), 
            self.sigmoid(d2.mean(dim=1)), 
            self.sigmoid(d3.mean(dim=1)), 
            self.sigmoid(d4.mean(dim=1))]

        d1 = self.sigmoid(self.afm1(d_att, d1))
        d2 = self.sigmoid(self.afm2(d_att, d2))
        d3 = self.sigmoid(self.afm3(d_att, d3))
        d4 = self.sigmoid(self.afm4(d_att, d4))
        
        ds2 = [d1, d2, d3, d4]

        avg_out = (d1+d2+d3+d4)/4
        # print(avg_out.size())
        return ds1, ds2, avg_out




if __name__ == "__main__":
    # inputs = torch.randn(1, 1, 224, 224)
    inputs = torch.randn(1, 1, 224, 224)
    net = AACAUNet(1)
    flops, params = profile(net, (inputs,))
    print( 'params: ', params, 'flops: ', flops,)
    print("%.2f | %.2f" % (params / (1000 ** 2), flops / (1000 ** 3)))#这里除以1000的平方，是为了化成M的单位
    # inputs = torch.randn(2, 128, 224, 224)
    x2 = torch.randn(2, 1, 224, 224)
    net = AACAUNet(1)
    # net = MFINet(in_channel=1)
    out = net(x2)
    # print(out.size())
    # for i in out:
    #     print(i.size())






















