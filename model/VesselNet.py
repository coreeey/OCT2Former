import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from thop import profile
import torch.autograd.profiler as profiler


class IRBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(IRBlock, self).__init__()

    self.conv1x1_1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1),
                                nn.ReLU(),
                                nn.BatchNorm2d(out_channels))

    self.conv1x1_2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1),
                                nn.ReLU(),
                                nn.BatchNorm2d(out_channels))

    self.conv3x3_1 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=2, padding=2),
                                nn.ReLU(),
                                nn.BatchNorm2d(out_channels))
    
    self.conv3x3_2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=3, padding=3),
                                nn.ReLU(),
                                nn.BatchNorm2d(out_channels))

    self.out_conv = nn.Conv2d(out_channels*2+in_channels, out_channels, kernel_size=3, padding=1)

  def forward(self, x):
    x_path1 = self.conv1x1_1(x)

    x_path2 = self.conv1x1_2(x)
    x_path2 = x_path2 + self.conv3x3_1(x_path2)
    x_path2 = x_path2 + self.conv3x3_2(x_path2)

    out = torch.cat((x, x_path1, x_path2), dim=1)

    out = self.out_conv(out)

    return out

class ConvRL(nn.Module):
    def __init__(self, in_channel, out_channel, kernel=2, stride=2):
        super().__init__()

        self.conv_rl = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel, stride=stride),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv_rl(x)
        return x

class Up(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, 2, stride=2)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class VesselNet(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        base_channels = 32
        self.ir1 = IRBlock(in_channels, base_channels)
        self.pool1 = ConvRL(base_channels, base_channels)

        self.ir2 = IRBlock(base_channels, base_channels*2)
        self.pool2 = ConvRL(base_channels*2, base_channels*2)

        self.ir3 = IRBlock(base_channels*2, base_channels*4)

        self.ir4 = IRBlock(base_channels*6, base_channels*2)

        self.ir5 = IRBlock(base_channels*3, base_channels)

        self.out1 = ConvRL(base_channels, 2, 1, 1)
        self.out2 = ConvRL(base_channels*2, 1, 1, 1)
        self.out3 = ConvRL(base_channels*4, 1, 1, 1)
        self.out4 = ConvRL(base_channels*2, 1, 1, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        outputs = []
        b, c, h, w = x.size()
        x1 = self.ir1(x)

        x2 = self.pool1(x1)
        x2 = self.ir2(x2)

        x3 = self.pool2(x2)
        x3 = self.ir3(x3)

        x4 = self.ir4(
            torch.cat([x2, F.interpolate(x3, size=x2.size()[-2:], mode='bilinear')], dim=1))

        x5 = self.ir5(
            torch.cat([x1, F.interpolate(x4, size=x1.size()[-2:], mode='bilinear')], dim=1))

        # out1 = self.sigmoid(self.out1(x5))
        out1 = self.out1(x5)

        out2 = self.sigmoid(self.out2(torch.cat([x1, x5], dim=1)))
        out3 = self.sigmoid(self.out3(x3))
        out4 = self.sigmoid(self.out4(x4))
        
        outputs = [out1, out2, out4, out3]
        return outputs

if __name__ == "__main__":
    net = VesselNet(1)
    # net = IRBlock(1, 32)
    inputs = torch.randn(1, 1, 224, 224)
    flops, params = profile(net, (inputs,))
    # print('flops: ', flops, 'params: ', params)
    print( 'params: ', params, 'flops: ', flops)

    print("%.2f | %.2f" % (params / (1000 ** 2), flops / (1000 ** 3)))#这里除以1000的平方，是为了化成M的单位
    # inputs = torch.randn(2, 128, 224, 224)
    # x2 = torch.randn(2, 1, 48, 48)
    # net = VesselNet(1)
    # out = net(x2)
    # # print(out.size())
    # for i in out:
    #     print(i.size())
    # from torchsummary import summary
    # from torchvision.models import resnet18
    # model = VesselNet(1)

    # summary(model, input_size=[(1, 224, 224)], batch_size=2, device="cpu")






















