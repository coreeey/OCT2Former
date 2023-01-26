import torch
import torch.nn as nn

#from .unet_utils import up_block, down_block
from .unet_utils import up_block, down_block
from .conv_trans_utils import *
#from .conv_trans_utils import *

import pdb

from thop import profile

class UTNet(nn.Module):
    
    def __init__(self, in_chan=1, base_chan=64, num_classes=1, reduce_size=8, block_list='234', num_blocks=[1, 2, 4], projection='interp', num_heads=[2,4,8], attn_drop=0., proj_drop=0., bottleneck=False, maxpool=True, rel_pos=True, aux_loss=False):
        super().__init__()

        self.aux_loss = aux_loss
        self.inc = [BasicBlock(in_chan, base_chan)]
        if '0' in block_list:
            self.inc.append(BasicTransBlock(base_chan, heads=num_heads[-5], dim_head=base_chan//num_heads[-5], attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size, projection=projection, rel_pos=rel_pos))
            self.up4 = up_block_trans(2*base_chan, base_chan, num_block=0, bottleneck=bottleneck, heads=num_heads[-4], dim_head=base_chan//num_heads[-4], attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size, projection=projection, rel_pos=rel_pos)
        
        else:
            self.inc.append(BasicBlock(base_chan, base_chan))
            self.up4 = up_block(2*base_chan, base_chan, scale=(2,2), num_block=2)
        self.inc = nn.Sequential(*self.inc)


        if '1' in block_list:
            self.down1 = down_block_trans(base_chan, 2*base_chan, num_block=num_blocks[-4], bottleneck=bottleneck, maxpool=maxpool, heads=num_heads[-4], dim_head=2*base_chan//num_heads[-4], attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size, projection=projection, rel_pos=rel_pos)
            self.up3 = up_block_trans(4*base_chan, 2*base_chan, num_block=0, bottleneck=bottleneck, heads=num_heads[-3], dim_head=2*base_chan//num_heads[-3], attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size, projection=projection, rel_pos=rel_pos)
        else:
            self.down1 = down_block(base_chan, 2*base_chan, (2,2), num_block=2)
            self.up3 = up_block(4*base_chan, 2*base_chan, scale=(2,2), num_block=2)

        if '2' in block_list:
            self.down2 = down_block_trans(2*base_chan, 4*base_chan, num_block=num_blocks[-3], bottleneck=bottleneck, maxpool=maxpool, heads=num_heads[-3], dim_head=4*base_chan//num_heads[-3], attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size, projection=projection, rel_pos=rel_pos)
            self.up2 = up_block_trans(8*base_chan, 4*base_chan, num_block=0, bottleneck=bottleneck, heads=num_heads[-2], dim_head=4*base_chan//num_heads[-2], attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size, projection=projection, rel_pos=rel_pos)

        else:
            self.down2 = down_block(2*base_chan, 4*base_chan, (2, 2), num_block=2)
            self.up2 = up_block(8*base_chan, 4*base_chan, scale=(2,2), num_block=2)

        if '3' in block_list:
            self.down3 = down_block_trans(4*base_chan, 8*base_chan, num_block=num_blocks[-2], bottleneck=bottleneck, maxpool=maxpool, heads=num_heads[-2], dim_head=8*base_chan//num_heads[-2], attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size, projection=projection, rel_pos=rel_pos)
            self.up1 = up_block_trans(16*base_chan, 8*base_chan, num_block=0, bottleneck=bottleneck, heads=num_heads[-1], dim_head=8*base_chan//num_heads[-1], attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size, projection=projection, rel_pos=rel_pos)

        else:
            self.down3 = down_block(4*base_chan, 8*base_chan, (2,2), num_block=2)
            self.up1 = up_block(16*base_chan, 8*base_chan, scale=(2,2), num_block=2)

        if '4' in block_list:
            self.down4 = down_block_trans(8*base_chan, 16*base_chan, num_block=num_blocks[-1], bottleneck=bottleneck, maxpool=maxpool, heads=num_heads[-1], dim_head=16*base_chan//num_heads[-1], attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size, projection=projection, rel_pos=rel_pos)
        else:
            self.down4 = down_block(8*base_chan, 16*base_chan, (2,2), num_block=2)


        self.outc = nn.Conv2d(base_chan, num_classes, kernel_size=1, bias=True)

        if aux_loss:
            self.out1 = nn.Conv2d(8*base_chan, num_classes, kernel_size=1, bias=True)
            self.out2 = nn.Conv2d(4*base_chan, num_classes, kernel_size=1, bias=True)
            self.out3 = nn.Conv2d(2*base_chan, num_classes, kernel_size=1, bias=True)
            


    def forward(self, x):
        size = x.size()[2:]
        #x = F.interpolate(x, size=(256, 256))
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        if self.aux_loss:
            out = self.up1(x5, x4)
            out1 = F.interpolate(self.out1(out), size=x.shape[-2:], mode='bilinear', align_corners=True)

            out = self.up2(out, x3)
            out2 = F.interpolate(self.out2(out), size=x.shape[-2:], mode='bilinear', align_corners=True)

            out = self.up3(out, x2)
            out3 = F.interpolate(self.out3(out), size=x.shape[-2:], mode='bilinear', align_corners=True)

            out = self.up4(out, x1)
            out = self.outc(out)

            return out, out3, out2, out1

        else:
            out = self.up1(x5, x4)
            out = self.up2(out, x3)
            out = self.up3(out, x2)

            out = self.up4(out, x1)
            out = self.outc(out)
        output = dict()
        #out = F.interpolate(out, size=size)
        output["main_out"] = out
        return output



        



class UTNet_Encoderonly(nn.Module):
    
    def __init__(self, in_chan, base_chan, num_classes=1, reduce_size=8, block_list='234', num_blocks=[1, 2, 4], projection='interp', num_heads=[2,4,8], attn_drop=0., proj_drop=0., bottleneck=False, maxpool=True, rel_pos=True, aux_loss=False):
        super().__init__()

        self.aux_loss = aux_loss
        
        self.inc = [BasicBlock(in_chan, base_chan)]
        if '0' in block_list:
            self.inc.append(BasicTransBlock(base_chan, heads=num_heads[-5], dim_head=base_chan//num_heads[-5], attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size, projection=projection, rel_pos=rel_pos))
        
        else:
            self.inc.append(BasicBlock(base_chan, base_chan))
        self.inc = nn.Sequential(*self.inc)


        if '1' in block_list:
            self.down1 = down_block_trans(base_chan, 2*base_chan, num_block=num_blocks[-4], bottleneck=bottleneck, maxpool=maxpool, heads=num_heads[-4], dim_head=2*base_chan//num_heads[-4], attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size, projection=projection, rel_pos=rel_pos)
        else:
            self.down1 = down_block(base_chan, 2*base_chan, (2,2), num_block=2)


        if '2' in block_list:
            self.down2 = down_block_trans(2*base_chan, 4*base_chan, num_block=num_blocks[-3], bottleneck=bottleneck, maxpool=maxpool, heads=num_heads[-3], dim_head=4*base_chan//num_heads[-3], attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size, projection=projection, rel_pos=rel_pos)
        else:
            self.down2 = down_block(2*base_chan, 4*base_chan, (2, 2), num_block=2)


        if '3' in block_list:
            self.down3 = down_block_trans(4*base_chan, 8*base_chan, num_block=num_blocks[-2], bottleneck=bottleneck, maxpool=maxpool, heads=num_heads[-2], dim_head=8*base_chan//num_heads[-2], attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size, projection=projection, rel_pos=rel_pos)
        else:
            self.down3 = down_block(4*base_chan, 8*base_chan, (2,2), num_block=2)

        if '4' in block_list:
            self.down4 = down_block_trans(8*base_chan, 16*base_chan, num_block=num_blocks[-1], bottleneck=bottleneck, maxpool=maxpool, heads=num_heads[-1], dim_head=16*base_chan//num_heads[-1], attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size, projection=projection, rel_pos=rel_pos)
        else:
            self.down4 = down_block(8*base_chan, 16*base_chan, (2,2), num_block=2)


        self.up1 = up_block(16*base_chan, 8*base_chan, scale=(2,2), num_block=2)
        self.up2 = up_block(8*base_chan, 4*base_chan, scale=(2,2), num_block=2)
        self.up3 = up_block(4*base_chan, 2*base_chan, scale=(2,2), num_block=2)
        self.up4 = up_block(2*base_chan, base_chan, scale=(2,2), num_block=2)

        self.outc = nn.Conv2d(base_chan, num_classes, kernel_size=1, bias=True)

        if aux_loss:
            self.out1 = nn.Conv2d(8*base_chan, num_classes, kernel_size=1, bias=True)
            self.out2 = nn.Conv2d(4*base_chan, num_classes, kernel_size=1, bias=True)
            self.out3 = nn.Conv2d(2*base_chan, num_classes, kernel_size=1, bias=True)
            


    def forward(self, x):
        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        if self.aux_loss:
            out = self.up1(x5, x4)
            out1 = F.interpolate(self.out1(out), size=x.shape[-2:], mode='bilinear', align_corners=True)

            out = self.up2(out, x3)
            out2 = F.interpolate(self.out2(out), size=x.shape[-2:], mode='bilinear', align_corners=True)

            out = self.up3(out, x2)
            out3 = F.interpolate(self.out3(out), size=x.shape[-2:], mode='bilinear', align_corners=True)

            out = self.up4(out, x1)
            out = self.outc(out)

            return out, out3, out2, out1

        else:
            out = self.up1(x5, x4)
            out = self.up2(out, x3)
            out = self.up3(out, x2)

            out = self.up4(out, x1)
            out = self.outc(out)

            return out

if __name__ == "__main__":
    net = UTNet(1, 64, 2)
    inputs = torch.randn(1, 1, 224, 224)
    flops, params = profile(net, (inputs,))
    print('flops: ', flops, 'params: ', params)
    print("%.2f | %.2f" % (params / (1000 ** 2), flops / (1000 ** 3)))#这里除以1000的平方，是为了化成M的单位

