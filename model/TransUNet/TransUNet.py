import torch
import torch.nn as nn
import functools
import torch.nn.functional as F
from .vit_seg_modeling import VisionTransformer as ViT_seg
from .vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from thop import profile

def get_transNet(n_classes):
    img_size = 400
    vit_patches_size = 20
    vit_name = 'R50-ViT-B_16'

    config_vit = CONFIGS_ViT_seg[vit_name]
    config_vit.n_classes = n_classes
    config_vit.n_skip = 3
    if vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))
    net = ViT_seg(config_vit, img_size=img_size, num_classes=n_classes)
    return net


if __name__ == '__main__':
    net = get_transNet(2)
    inputs = torch.randn((2, 3, 224, 224))

    flops, params = profile(net, (inputs,))
    print('flops: ', flops, 'params: ', params)
    print("%.2f | %.2f" % (params / (1000 ** 2), flops / (1000 ** 3)))#这里除以1000的平方，
    # for edge in edges:
    #     print(edge.size())
