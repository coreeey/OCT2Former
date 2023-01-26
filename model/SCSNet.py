"""
modify by https://github.com/Qsingle/imed_vision/blob/3b12f854d39158e4e2b7acf3a11d42219299cd96/models/segmentation/scsnet.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import collections.abc
import functools
from itertools import repeat

def to_tuple(input, n):
    """
    Get a tuple with n data
    Args:
        input (Union[int, tuple]): the input
        n (int): the number of data
    Returns:
        A tuple with n data
    """
    if isinstance(input, collections.abc.Iterable):
        assert len(input) == n, "tuple len is not equal to n: {}".format(input)
        spatial_axis = map(int, input)
        value = tuple(spatial_axis)
        return value
    return tuple(repeat(input, n))

_pair = functools.partial(to_tuple, n=2)

class Conv2d(nn.Module):
    def __init__(self, in_ch, out_ch, ksize=1, stride=1, padding=0, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d,
                 activation=nn.ReLU(inplace=True), dropout_rate=0.0, gn_groups=32,
                 **kwargs):
        """
        The conv2d with normalization layer and activation layer.
        Args:
            in_ch (int): the number of channels for input
            out_ch (int): the number of channels for output
            ksize (Union[int,tuple]): the size of conv kernel, default is 1
            stride (Union[int,tuple]): the stride of the slide window, default is 1
            padding (Union[int, tuple]): the padding size, default is 0
            dilation (Union[int,tuple]): the dilation rate, default is 1
            groups (int): the number of groups, default is 1
            bias (bool): whether use bias, default is False
            norm_layer (nn.Module): the normalization module
            activation (nn.Module): the nonlinear module
            dropout_rate (float): dropout rate
        """
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=ksize, stride=stride,
                              padding=padding, dilation=dilation, groups=groups,
                              bias=bias, **kwargs)
        self.norm_layer = norm_layer
        if not norm_layer is None:
            if isinstance(norm_layer, nn.GroupNorm):
                self.norm_layer = norm_layer(gn_groups, out_ch)
            else:
                self.norm_layer = norm_layer(out_ch)
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)


    def forward(self, x):
        net = self.conv(x)
        if self.norm_layer is not None:
            net = self.norm_layer(net)
        if self.activation is not None:
            net = self.activation(net)
        if self.dropout_rate > 0:
            self.dropout(net)
        return net


class SpatialFusion(nn.Module):
    def __init__(self, sr_ch, seg_ch, hidden_state=32):
        super(SpatialFusion, self).__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(sr_ch + seg_ch, hidden_state, 1, 1),
            nn.ReLU()
        )
        self.fusion_1 = nn.Sequential(
            nn.Conv2d(hidden_state, hidden_state, (7, 1), (1, 1), padding=(3, 0)),
        )

        self.fusion_2 = nn.Sequential(
            nn.Conv2d(hidden_state, hidden_state, (1, 7), (1, 1), padding=(0, 3))
        )
        self.fusion = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(hidden_state, seg_ch, 1, 1),
            # nn.Softmax(dim=1)
            nn.Sigmoid()
        )

    def forward(self, sr_fe, seg_fe):
        proj = self.proj(torch.cat([sr_fe, seg_fe], dim=1))
        fusion_1 = self.fusion_1(proj)
        fusion_2 = self.fusion_2(proj)
        fusion = self.fusion(fusion_1+fusion_2)
        return fusion

class SFA(nn.Module):
    def __init__(self, in_ch):
        """
            Implementation of Scale-aware feature aggregation module
            References:
                "SCS-Net: A Scale and Context Sensitive Network for Retinal Vessel Segmentation"
                <https://www.sciencedirect.com/science/article/pii/S1361841521000712>
            Parameters:
                in_ch (int): number of input channels
        """
        super(SFA, self).__init__()
        self.dilation_1 = nn.Conv2d(in_ch, in_ch, 3, 1, padding=1, dilation=1, bias=False)
        self.dilation_3 = nn.Conv2d(in_ch, in_ch, 3, 1, padding=3, dilation=3, bias=False)
        self.dilation_5 = nn.Conv2d(in_ch, in_ch, 3, 1, padding=5, dilation=5, bias=False)

        self.fusion_12 = nn.Sequential(
            nn.Conv2d(in_ch*2, in_ch, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, 2, 1, 1)
        )

        self.fusion_23 = nn.Sequential(
            nn.Conv2d(in_ch*2, in_ch, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, 2, 1, 1)
        )

        self.att_fusion = nn.Conv2d(in_ch, in_ch, 1, 1)


    def forward(self, x):
        f1 = self.dilation_1(x)
        f2 = self.dilation_3(x)
        f3 = self.dilation_5(x)

        f12 = torch.cat([f1, f2], dim=1)
        f23 = torch.cat([f2, f3], dim=1)

        fusion_12 = self.fusion_12(f12)
        fusion_23 = self.fusion_23(f23)

        att_12 = torch.softmax(fusion_12, dim=1)
        w_alpha1, w_beta1 = torch.split(att_12, 1, dim=1)

        att_23 = torch.softmax(fusion_23, dim=1)
        w_alpha2, w_beta2 = torch.split(att_23, 1, dim=1)

        att_1 = w_alpha1*f1 + w_beta1*f2
        att_2 = w_alpha2*f2 + w_beta2*f3
        out = att_1 + att_2 + x
        out = self.att_fusion(out)
        return out

class AFF(nn.Module):
    def __init__(self, in_ch, reduction=16):
        """
            Implementation of Adaptive feature fusion module
            References:
                "SCS-Net: A Scale and Context Sensitive Network for Retinal Vessel Segmentation"
                <https://www.sciencedirect.com/science/article/pii/S1361841521000712>
            Parameters
            ----------
            in_ch (int): number of channels of input
            reduction (int): reduction rate for squeeze
        """
        super(AFF, self).__init__()
        in_ch1 = in_ch*2
        hidden_ch = (in_ch*2) // reduction
        self.se = nn.Sequential(
            nn.Conv2d(in_ch1, hidden_ch, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_ch, in_ch1, 1),
            nn.Sigmoid()
        )
        self.conv1x1 = nn.Conv2d(in_ch1, in_ch, 1)

    def forward(self, x1, x2):
        """
        Parameters
        ----------
        x1 (Tensor): low level feature, (n,c,h,w)
        x2 (Tensor): high level feature, (n,c,h,w)
        Returns
        -------
            Tensor, fused feature
        """
        x12 = torch.cat([x1, x2], dim=1)
        se = self.se(x12)
        se = self.conv1x1(se)
        se = F.adaptive_avg_pool2d(se, 1)
        se = torch.sigmoid(se)
        w1 = se * x1
        out = w1 + x2
        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm_layer=nn.BatchNorm2d, activation=nn.ReLU()):
        """
            Implementation of the residual block in SCS-Net, we follow the structure
            described in ResNet to build this block.
            References:
                "SCS-Net: A Scale and Context Sensitive Network for Retinal Vessel Segmentation"
                <https://www.sciencedirect.com/science/article/pii/S1361841521000712#!>
                "Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>
            Parameters:
            ----------
                in_ch (int): number of channels for input
                out_ch (int): number of channels for output
                norm_layer (nn.Module): normalization layer
                activation (nn.Module): activation layer
        """
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            Conv2d(in_ch, out_ch, ksize=3, stride=1, padding=1, norm_layer=norm_layer, activation=activation),
            Conv2d(out_ch, out_ch, ksize=3, stride=1, padding=1, norm_layer=norm_layer, activation=None)
        )
        self.activation = activation
        self.identity = nn.Identity()
        if in_ch != out_ch:
            self.identity = nn.Conv2d(in_ch, out_ch, 1, 1, bias=False)

    def forward(self, x):
        identity = self.identity(x)
        net = self.conv(x)
        net = net + identity
        net = self.activation(net)
        return net


class SCSNet(nn.Module):
    def __init__(self, in_ch=1, num_classes=1,  out_size=None,upscale_rate=2, alphas=[0.6, 0.3, 0.1]):
        """
        Implementation of SCSNet.
        References:
                "SCS-Net: A Scale and Context Sensitive Network for Retinal Vessel Segmentation"
                <https://www.sciencedirect.com/science/article/pii/S1361841521000712#!>
        Args:
            in_ch (int): number of channels for input
            num_classes (int): number of classes
            super_reso (bool): whether use super-resolution
            out_size (Union[int,tuple]:
            upscale_rate:
            alphas:
        """
        super(SCSNet, self).__init__()
        base_ch = 64
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder1 = ResidualBlock(in_ch, base_ch)
        self.encoder2 = ResidualBlock(base_ch, base_ch*2)
        self.encoder3 = ResidualBlock(base_ch*2, base_ch*4)

        self.sfa = SFA(base_ch*4)

        self.aff3 = AFF(base_ch*4)
        self.aff2 = AFF(base_ch*2)
        self.aff_conv3 = nn.Sequential(
            nn.Conv2d(base_ch*4, base_ch*2, 3, stride=1, padding=1, bias=False),
            nn.ReLU()
        )
        self.aff1 = AFF(base_ch)
        self.aff_conv2 = nn.Sequential(
            nn.Conv2d(base_ch*2, base_ch, 3, stride=1, padding=1, bias=False),
            nn.ReLU()
        )
        self.aff_conv1 =  nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, stride=1, padding=1, bias=False),
            nn.ReLU()
        )

        self.side_l3 = nn.Conv2d(base_ch*2, num_classes, 1)
        self.side_l2 = nn.Conv2d(base_ch, num_classes, 1)
        self.side_l1 = nn.Conv2d(base_ch, num_classes, 1)

        self.alpha_l3 = alphas[2]
        self.alpha_l2 = alphas[1]
        self.alpha_l1 = alphas[0]
        self.upscale_rate = upscale_rate
        self.out_size = _pair(out_size)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        en1 = self.encoder1(x)
        down1 = self.down(en1)
        en2 = self.encoder2(down1)
        down2 = self.down(en2)
        en3 = self.encoder3(down2)
        down3 = self.down(en3)
        sfa = self.sfa(down3)
        sfa = F.interpolate(sfa, size=en3.shape[2:], mode="bilinear", align_corners=True)
        aff3 = self.aff3(en3, sfa)
        aff3 = self.aff_conv3(aff3)
        aff3_up = F.interpolate(aff3, size=en2.shape[2:], mode="bilinear", align_corners=True)
        aff2 = self.aff2(en2, aff3_up)
        aff2 = self.aff_conv2(aff2)
        aff2_up = F.interpolate(aff2, size=en1.shape[2:], mode="bilinear", align_corners=True)
        aff1 = self.aff1(en1, aff2_up)
        aff1 = self.aff_conv1(aff1)
        side1 = self.side_l1(aff1)
        side2 = self.side_l2(F.interpolate(aff2, size=x.shape[2:], mode="bilinear",
                                                  align_corners=True))
        side3 = self.side_l3(F.interpolate(aff3, size=x.shape[2:], mode="bilinear",
                                                  align_corners=True))

        out = self.alpha_l1 * self.sigmoid(side1) + self.alpha_l2 * self.sigmoid(side2) + self.alpha_l3 * self.sigmoid(side3)
 
        return out

if __name__ == "__main__":
    import torch
    from thop import profile

    # x = torch.randn(1, 3, 224, 224)
    # model = SCSNet(3)
    # model.eval()
    # out = model(x)
    # print(out.shape)
    inputs = torch.randn(1, 1, 224, 224)
    net = SCSNet(1)
    net.eval()
    flops, params = profile(net, (inputs,))
    print('flops: ', flops, 'params: ', params)
    print("%.2f | %.2f" % (params / (1000 ** 2), flops / (1000 ** 3)))