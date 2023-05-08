import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from torchvision import models
from einops import rearrange, repeat
from torch import einsum
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
import math
from torch.nn.modules.batchnorm import _BatchNorm
from einops import rearrange

from thop import profile

nonlinearity = partial(F.relu, inplace=False)

def init_weight(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:

        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        #nn.init.orthogonal_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Batch') != -1:
        m.weight.data.normal_(1,0.02)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Embedding') != -1:
        nn.init.orthogonal_(m.weight, gain=1)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim,  k, stage_num=3, num_heads=8, attn_drop=0., proj_drop=0.):
        super(Attention, self).__init__()
        self.stage_num = stage_num
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim*4, bias=False)
        self.k = nn.Linear(dim, dim*4, bias=False)
        self.v = nn.Linear(dim, dim*4, bias=False)
        self.DA = DTA(k, stage_num)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim*4, dim)
        self.proj_drop = nn.Dropout(proj_drop)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, _BatchNorm):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Parameter):
                m.weight.data.normal_(0, math.sqrt(2. / k)) 


    def forward(self, x):
        idn = x
        B, N, C = x.size()
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        v = rearrange(v, 'b n c -> b c n')
        k = rearrange(k, 'b n c -> b c n')
        q = rearrange(q, 'b n c -> b c n')

        q, k = self.DA(q, k)
        #mutihead
        v = rearrange(v, 'b (n_h head_dim) n -> b n_h head_dim n', n_h = self.num_heads)
        k = rearrange(k, 'b (n_h head_dim) n -> b n_h n head_dim', n_h = self.num_heads)
        q = rearrange(q, 'b (n_h head_dim) n -> b n_h n head_dim', n_h = self.num_heads)

        att = torch.matmul(q.permute(0, 1, 3, 2), k) * self.scale
        att = F.softmax(att, dim=-1)   
        x = torch.matmul(att, v)  # b * c * n
        out = rearrange(x, 'b n_h head_dim n-> b n (n_h head_dim)')

        out = self.proj(out)
        out = self.proj_drop(out)
        out = F.relu(out, inplace=True)
        return out

    def _l2norm(self, inp, dim):
        return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))

class DABlock(nn.Module):
    def __init__(self, dim, num_heads, k=128, mlp_ratio=4.,drop=0. ,attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,  local=False, div=1):
        super().__init__()

        self.norm1 = norm_layer(div*dim)
        self.attn = Attention(div*dim,k,stage_num=6,num_heads=num_heads, attn_drop=attn_drop,  proj_drop=drop)
        self.local = local
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.conv_aux = double_conv(dim, dim)
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x):
        b,c,H,W = x.size()
        x_conv = self.conv_aux(x)
        x = rearrange(x,'b c h w -> b (h w) c ')
        x = self.attn(self.norm1(x))
        x = x + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = rearrange(x,'b (h w) c -> b c h w ', h = H)
        x = x + self.alpha * x_conv
        return x

class GroupEmbed(nn.Module):

    def __init__(self, in_ch=3, patch=3, stride=2, out_ch=768, with_pos=False):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=patch, stride=stride, padding=1, groups=1)
        self.norm = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.conv(x)
        x = self.norm(x)
        return x

class BasicStem(nn.Module):
    def __init__(self, in_ch=3, out_ch=64, with_pos=False, act=nn.LeakyReLU()):
        super(BasicStem, self).__init__()
        hidden_ch = out_ch // 2
        self.conv1 = nn.Conv2d(in_ch, hidden_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(hidden_ch)
        self.conv2 = nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(hidden_ch)
        self.conv3 = nn.Conv2d(hidden_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)

        self.act = act

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)

        x = self.conv3(x)
        return x

class DTA(nn.Module):
    def __init__(self, k=130,stage_num=3):
        super(DTA, self).__init__()
        self.pool = nn.AdaptiveMaxPool1d(k)
        self.stage_num = stage_num

    def forward(self, x1, x2):
        k = self.pool(x1)
        k = self._l2norm(k, dim=1)
        q = self.pool(x2)
        q = self._l2norm(q, dim=1)
        x1 = rearrange(x1, 'b n c -> b c n')
        x2 = rearrange(x2, 'b n c -> b c n')
	    #with torch.no_grad():
        for i in range(self.stage_num):
            z1 = torch.bmm(x1, k)  
            z1 = F.softmax(z1, dim=2) 
            z1_ = self._l2norm(z1, dim=1) 
            x1_ = x1.permute(0, 2, 1) 
            k = torch.bmm(x1_, z1_) 
            k = self._l2norm(k, dim=1)

            z2 = torch.bmm(x2, q)  
            z2 = F.softmax(z2, dim=2) 
            z2_ = self._l2norm(z2, dim=1)  
            x2_ = x2.permute(0, 2, 1) 
            q = torch.bmm(x2_, z2_) 
            q = self._l2norm(q, dim=1)
        return k, q

    def _l2norm(self, inp, dim):
        return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))

class double_conv(nn.Module):
    def __init__(self, in_channel, out_channel, act=nn.ReLU):
        super().__init__()

        self.dconv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            act(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            act(),
        )       

    def forward(self, x):
        x = self.dconv(x)

        return x

class single_conv(nn.Module):
    def __init__(self, in_channel, out_channel, act=nn.ReLU):
        super().__init__()

        self.dconv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            act(),
        )        

    def forward(self, x):
        x = self.dconv(x)

        return x

class triple_conv(nn.Module):
    def __init__(self, in_channel, out_channel, act=nn.ReLU):
        super().__init__()

        self.dconv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            act(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            act(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            act(),
        )    
         
    def forward(self, x):
        x = self.dconv(x)

        return x

class conv3x3(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, dilation=1),
        )
    def forward(self, inputs):
        x = self.conv(inputs)
        return x

class conv1x1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1,dilation=1),
        )
    def forward(self, inputs):
        x = self.conv(inputs)
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
        # elif mode == 'CARA':
        #     self.up = CARAFE(in_ch)

    def forward(self, x):
        x = self.up(x)
        return x

class DA_encoder(nn.Module):
    def __init__(self, in_chans=2, embed_dims=[64, 128, 256, 512, 1024],k=[128, 128, 128, 64, 64],num_heads=[2, 4, 4, 8, 16], mlp_ratios=[4, 4, 4, 4, 4],
                 norm_layer=nn.LayerNorm):
        super().__init__()

        self.k = k

        self.stem = BasicStem(in_ch=in_chans, out_ch=embed_dims[0], with_pos=True)
        self.pe2 = GroupEmbed(in_ch=embed_dims[0], out_ch=embed_dims[1], with_pos=True)
        self.pe3 = GroupEmbed(in_ch=embed_dims[1], out_ch=embed_dims[2], with_pos=True)
        # self.pe4 = GroupEmbed(in_ch=embed_dims[2], out_ch=embed_dims[3], with_pos=True)
        # self.pe5 = GroupEmbed(in_ch=embed_dims[3], out_ch=embed_dims[4], with_pos=True)

        self.fea1_1 =DABlock(embed_dims[0], num_heads[0], k=self.k[0], mlp_ratio=mlp_ratios[0],norm_layer=norm_layer)

        self.fea2_1 =DABlock(embed_dims[1], num_heads[1], k=self.k[1], mlp_ratio=mlp_ratios[1],norm_layer=norm_layer)

        self.fea3_1 =DABlock(embed_dims[2], num_heads[2], k=self.k[2], mlp_ratio=mlp_ratios[2],norm_layer=norm_layer)
        self.fea3_2 =DABlock(embed_dims[2], num_heads[2], k=self.k[2], mlp_ratio=mlp_ratios[2],norm_layer=norm_layer)

        # self.fea4_1 =DABlock(embed_dims[3], num_heads[3], k=self.k[3], mlp_ratio=mlp_ratios[3],norm_layer=norm_layer)
        # self.fea4_2 =DABlock(embed_dims[3], num_heads[3], k=self.k[3], mlp_ratio=mlp_ratios[3],norm_layer=norm_layer)

        # self.fea5_1 =DABlock(embed_dims[4], num_heads[4], k=self.k[4], mlp_ratio=mlp_ratios[4],norm_layer=norm_layer)

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(embed_dims[3], embed_dims[3])
        # self.conv2 = nn.Conv2d(embed_dims[3], embed_dims[3], 3, 1, 1)
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
        outputs = dict()
        size = x.size()[2:]
        B, _, H, W = x.shape
        
        feature = []
        #1
        x = self.stem(x)
        x = self.fea1_1(x)
        v1 = x

        #2
        x= self.pe2(x)
        x= self.fea2_1(x)
        v2 = x
       #3
        x = self.pe3(x)
        x = self.fea3_1(x)
        x = self.fea3_2(x)
        v3 = x

        # x = self.pe4(x)
        # x = self.fea4_1(x)
        # x = self.fea4_2(x)
        # v4 = x

        
        # x = self.pe5(x)
        # x = self.fea5_1(x)
        # v5 = x


        return v1, v2, v3#, v4, v5#, x#, dense_x

class OCT2Former(nn.Module):
    def __init__(self, in_chans=3, num_classes=2, embed_dims=[64, 128, 256, 512, 1024],k=[128, 128, 128, 128, 128],num_heads=[1, 2, 4, 8, 16], mlp_ratios=[4, 4, 4, 4, 4],
                 depths=[1, 1, 1, 2, 1],norm_layer=nn.LayerNorm, aux=False, spec_inter=False):
        super().__init__()
        #k=[128,128,128,128,128]
        filters = [256, 512, 1024, 2048, 4096]
        filters = [int(c / 4) for c in filters]
        self.k = k
        self.num_classes = num_classes
        self.depths = depths
        self.aux = aux

        # self.base_encoder = DA_encoder(in_chans, embed_dims, [16, 16, 16], num_heads, mlp_ratios, norm_layer)

        self.base_encoder = DA_encoder(in_chans, embed_dims, k, num_heads, mlp_ratios, norm_layer)

        if spec_inter:
            self.up0 = up(embed_dims[3], size=(23, 23))
            self.up1 = up(embed_dims[2]) #, 'CARA')
            self.up2 = up(embed_dims[1], size=(91, 91))#, 'CARA')
        else:
            self.up0 = up(embed_dims[3])
            self.up1 = up(embed_dims[2])
            self.up2 = up(embed_dims[1])


        # self.donv6 = double_conv(embed_dims[2]+embed_dims[3], embed_dims[2])
        # self.donv7 = double_conv(embed_dims[1]+embed_dims[2], embed_dims[1])
        # self.donv8 = double_conv(embed_dims[0]+embed_dims[1], filters[0])

        # self.donv6 = triple_conv(embed_dims[2]+embed_dims[3], embed_dims[2])
        # self.donv7 = triple_conv(embed_dims[1]+embed_dims[2], embed_dims[1])
        # self.donv8 = triple_conv(embed_dims[0]+embed_dims[1], filters[0])


        # self.donv7 = nn.Sequential(DABlock(embed_dims[1]+embed_dims[2],num_heads[1], k=128),
        #             nn.Conv2d(embed_dims[1]+embed_dims[2],  embed_dims[1], 3, padding=1))
        # self.donv8 = nn.Sequential(DABlock(embed_dims[0]+embed_dims[1], num_heads[0], k=128),
        #             nn.Conv2d(embed_dims[0]+embed_dims[1],  filters[0], 3, padding=1))
#decoder analyize
        self.donv6 = single_conv(embed_dims[2]+embed_dims[3], embed_dims[2])
        self.donv7 = single_conv(embed_dims[1]+embed_dims[2], embed_dims[1])
        self.donv8 = single_conv(embed_dims[0]+embed_dims[1], filters[0])

        self.final_out = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, num_classes, 3, padding=1)
        )
        # init weights
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
        outputs = dict()
        size = x.size()[2:]
        B, _, H, W = x.shape

        
#center
        aux_feature = []
        feature = []
        v1, v2, v3= self.base_encoder(x)
# 1

        x = self.up1(v3)
        x = self.donv7(torch.cat((x, v2), dim=1))
#2
        x = self.up2(x)
        x = self.donv8(torch.cat((x, v1), dim=1))
        feature.append(x.mean(1))
        out = self.final_out(x)

        outputs.update({'main_out': out})
        return outputs # 1 2 2 4 2

class OCT2Former_DCD(nn.Module):#decoder of unet3+

    def __init__(self, in_channels,channels=64,n_classes=2):
        super(OCT2Former_DCD, self).__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.n_classes=n_classes

        ## -------------Encoder--------------
        self.conv1 = double_conv(self.in_channels, self.channels)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = double_conv(self.channels, self.channels*2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = double_conv(self.channels*2, self.channels*4)

        self.base_encoder = DA_encoder(1, [64, 128, 256], [128,128,128], num_heads=[1, 2, 4])

        ## -------------Decoder--------------
        self.CatChannels = self.channels
        self.CatBlocks = 3
        self.UpChannels = self.CatChannels * self.CatBlocks

        self.h1_PT_hd3 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv = nn.Conv2d(self.channels, self.CatChannels, 3, padding=1)
        self.h1_PT_hd3_relu = nn.ReLU(inplace=True)

        self.h2_PT_hd3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = nn.Conv2d(self.channels*2, self.CatChannels, 3, padding=1)
        self.h2_PT_hd3_relu = nn.ReLU(inplace=True)

        self.h3_Cat_hd3_conv = nn.Conv2d(self.channels*4, self.CatChannels, 3, padding=1)
        self.h3_Cat_hd3_relu = nn.ReLU(inplace=True)


        self.conv3d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)
        self.relu3d_1 = nn.ReLU(inplace=True)

        self.h1_PT_hd2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = nn.Conv2d(self.channels, self.CatChannels, 3, padding=1)
        self.h1_PT_hd2_relu = nn.ReLU(inplace=True)

        self.h2_Cat_hd2_conv = nn.Conv2d(self.channels*2, self.CatChannels, 3, padding=1)
        self.h2_Cat_hd2_relu = nn.ReLU(inplace=True)

        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.hd3_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd2_relu = nn.ReLU(inplace=True)

        self.conv2d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.relu2d_1 = nn.ReLU(inplace=True)


        self.h1_Cat_hd1_conv = nn.Conv2d(self.channels, self.CatChannels, 3, padding=1)
        self.h1_Cat_hd1_relu = nn.ReLU(inplace=True)

        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd2_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd2_UT_hd1_relu = nn.ReLU(inplace=True)

        self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd3_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd1_relu = nn.ReLU(inplace=True)

        self.conv1d_1 = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)  # 16
        self.relu1d_1 = nn.ReLU(inplace=True)

        self.outconv1 = nn.Conv2d(self.CatChannels, n_classes, 3, padding=1)


    def forward(self, inputs):
        ## -------------Encoder-------------
        h1, h2, h3= self.base_encoder(inputs)
        # h1 = self.conv1(inputs)  # h1->304*304

        # h2 = self.maxpool1(h1)
        # h2 = self.conv2(h2)  # h2->152*152

        # h3 = self.maxpool2(h2)
        # h3 = self.conv3(h3)  # h3->76*76


        ## -------------Decoder-------------

        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1)))
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2)))
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_conv(h3))
        hd3 = self.relu3d_1(self.conv3d_1(torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3), 1)))

        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1)))
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_conv(h2))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3)))
        hd2 = self.relu2d_1(self.conv2d_1(torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2), 1)))

        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_conv(h1))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2)))
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3)))
        x = self.relu1d_1(self.conv1d_1(torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1), 1)))

        d1 = self.outconv1(x)  # d1->320*320*n_classes
        return d1,x

if __name__ == "__main__":
    net = OCT2Former(1, 2)
    inputs = torch.randn(1, 1, 224, 224)
    flops, params = profile(net, (inputs,))
    print('flops: ', flops, 'params: ', params)
    print("%.2f | %.2f" % (params / (1000 ** 2), flops / (1000 ** 3)))