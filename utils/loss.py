import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.utils import make_one_hot
from torchvision.utils import save_image
import os

class BinaryDiceLoss(nn.Module):

    def __init__(self, smooth=1, p=1.0, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        # print(target.size(), predict.size())
        target = target.contiguous().view(target.shape[0], -1)
        predict = predict.contiguous().view(predict.shape[0], -1) 

        intersect = 2.0 * torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        union = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        dice_loss = torch.sub(1.0, intersect/union)  # (batch_size)

        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        elif self.reduction == 'none':
            return dice_loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))




class DiceLoss(nn.Module):
    def __init__(self, ignore_index=0, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        n_class = predict.size()[1]
        if n_class > 1:
            predict = F.softmax(predict, dim=1)

        dice = BinaryDiceLoss(**self.kwargs)
        dice_loss = 0
        for i in range(n_class):
            if i == self.ignore_index:
                continue
            predict_i = predict[:, i, :, :]
            target_i = (target == i).float()
            dice_loss += dice(predict_i, target_i)

        if self.ignore_index == None:
            dice_loss /= n_class
        else:
            dice_loss /= n_class-1

        return dice_loss


class CrossEntropy(nn.Module):
    def __init__(self, weight=None, label_smooth=None, ignore_label=-1):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight,
                                             ignore_index=ignore_label)
        self.label_smooth = label_smooth
    def forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(
                input=score, size=(h, w), mode='bilinear')

        if self.label_smooth is not None:
            target = (1.0-self.label_smooth)*target + self.label_smooth/2

        loss = self.criterion(score, target.long())

        return loss


class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, thres=0.8,
                 min_kept=100000, weight=None):
        super(OhemCrossEntropy, self).__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight,
                                             ignore_index=ignore_label,
                                             reduction='none')

    def forward(self, score, target, **kwargs):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(input=score, size=(h, w), mode='bilinear')
        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label

        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0  
        pred = pred.gather(1, tmp_target.unsqueeze(1)) 
        pred, ind = pred.contiguous().view(-1, )[mask].contiguous().sort()  
        min_value = pred[min(self.min_kept, pred.numel() - 1)]  #
        threshold = max(min_value, self.thresh)  

        pixel_losses = pixel_losses[mask][ind] 
        pixel_losses = pixel_losses[pred < threshold]  
        return pixel_losses.mean()


class soft_cldice(nn.Module):
    def __init__(self, iter_=3, smooth = 1.):
        super(soft_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth

    def forward(y_true, y_pred):
        skel_pred = self.soft_skel(y_pred, iters)
        skel_true = self.soft_skel(y_true, iters)
        tprec = (torch.sum(torch.multiply(skel_pred, y_true)[:,1:,...])+smooth)/(torch.sum(skel_pred[:,1:,...])+smooth)    
        tsens = (torch.sum(torch.multiply(skel_true, y_pred)[:,1:,...])+smooth)/(torch.sum(skel_true[:,1:,...])+smooth)    
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
        return cl_dice

    def soft_erode(self, img):
        if len(img.shape)==4:
            p1 = -F.max_pool2d(-img, (3,1), (1,1), (1,0))
            p2 = -F.max_pool2d(-img, (1,3), (1,1), (0,1))
            return torch.min(p1,p2)
        elif len(img.shape)==5:
            p1 = -F.max_pool3d(-img,(3,1,1),(1,1,1),(1,0,0))
            p2 = -F.max_pool3d(-img,(1,3,1),(1,1,1),(0,1,0))
            p3 = -F.max_pool3d(-img,(1,1,3),(1,1,1),(0,0,1))
            return torch.min(torch.min(p1, p2), p3)


    def soft_dilate(self, img):
        if len(img.shape)==4:
            return F.max_pool2d(img, (3,3), (1,1), (1,1))
        elif len(img.shape)==5:
            return F.max_pool3d(img,(3,3,3),(1,1,1),(1,1,1))


    def soft_open(self, img):
        return soft_dilate(soft_erode(img))


    def soft_skel(self, img, iter_):
        img1  =  soft_open(img)
        skel  =  F.relu(img-img1)
        for j in range(iter_):
            img  =  soft_erode(img)
            img1  =  soft_open(img)
            delta  =  F.relu(img-img1)
            skel  =  skel +  F.relu(delta-skel*delta)
        return skel




class soft_dice_cldice(nn.Module):
    def __init__(self, iter_=3, alpha=0.5, smooth = 0.01):
        super(soft_dice_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        y_true = y_true.contiguous().unsqueeze(1).to(float)
        #y_pred_ = F.softmax(y_pred, dim=1)
        #y_pred = torch.exp(y_pred).max(dim=1)[1].unsqueeze(1).to(float).requires_grad_()
        y_pred = (y_pred > 0.5).contiguous().to(float).requires_grad_()

        #print('pred', y_pred.size(), y_true.size())
        dice = self.soft_dice(y_true, y_pred)
        skel_pred = self.soft_skel(y_pred, self.iter)
        skel_true = self.soft_skel(y_true, self.iter)

        tprec = (torch.sum(torch.multiply(skel_pred, y_true)[:,1:,...])+self.smooth)/(torch.sum(skel_pred[:,1:,...])+self.smooth)    
        tsens = (torch.sum(torch.multiply(skel_true, y_pred)[:,1:,...])+self.smooth)/(torch.sum(skel_true[:,1:,...])+self.smooth)    
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
  
        return (1.0-self.alpha)*dice+self.alpha*cl_dice

    def soft_skel(self, img, iter_):
        img1  =  soft_open(img)
        skel  =  F.relu(img-img1)
        for j in range(iter_):
            img  =  soft_erode(img)
            img1  =  soft_open(img)
            delta  =  F.relu(img-img1)
            skel  =  skel +  F.relu(delta-skel*delta)
        return skel

    def soft_dice(self, y_true, y_pred):
        """[function to compute dice loss]
        Args:
            y_true ([float32]): [ground truth image]
            y_pred ([float32]): [predicted image]
        Returns:
            [float32]: [loss value]
        """
        smooth = 1
        intersection = torch.sum((y_true * y_pred)[:,1:,...])
        coeff = (2. *  intersection + smooth) / (torch.sum(y_true[:,1:,...]) + torch.sum(y_pred[:,1:,...]) + smooth)
        return (1. - coeff)
import torch
import numpy as np
from torch import einsum
from torch import Tensor
from scipy.ndimage import distance_transform_edt as distance
from scipy.spatial.distance import directed_hausdorff

from typing import Any, Callable, Iterable, List, Set, Tuple, TypeVar, Union

# switch between representations
def probs2class(probs: Tensor) -> Tensor:
    b, _, w, h = probs.shape  # type: Tuple[int, int, int, int]
    assert simplex(probs)

    res = probs.argmax(dim=1)
    assert res.shape == (b, w, h)

    return res

def probs2one_hot(probs: Tensor) -> Tensor:
    _, C, _, _ = probs.shape
    assert simplex(probs)

    res = class2one_hot(probs2class(probs), C)
    assert res.shape == probs.shape
    assert one_hot(res)

    return res


def class2one_hot(seg: Tensor, C: int) -> Tensor:
    if len(seg.shape) == 2:  # Only w, h, used by the dataloader
        seg = seg.unsqueeze(dim=0)
    assert sset(seg, list(range(C)))

    b, w, h = seg.shape  # type: Tuple[int, int, int]

    res = torch.stack([seg == c for c in range(C)], dim=1).type(torch.int32)
    assert res.shape == (b, C, w, h)
    assert one_hot(res)

    return res


def one_hot2dist(seg: np.ndarray) -> np.ndarray:
    assert one_hot(torch.Tensor(seg), axis=0)
    C: int = len(seg)

    res = np.zeros_like(seg)
    for c in range(C):
        posmask = seg[c].astype(np.bool)

        if posmask.any():
            negmask = ~posmask
            # print('negmask:', negmask)
            # print('distance(negmask):', distance(negmask))
            res[c] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
            # print('res[c]', res[c])
    return res


def simplex(t: Tensor, axis=1) -> bool:
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])

    # Assert utils
def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())


def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)

class SurfaceLoss():
    def __init__(self):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = [1]   #这里忽略背景类  https://github.com/LIVIAETS/surface-loss/issues/3

    # probs: bcwh, dist_maps: bcwh
    def __call__(self, probs: Tensor, dist_maps: Tensor, _: Tensor) -> Tensor:
        assert simplex(probs)
        assert not one_hot(dist_maps)
        
        probs = F.softmax(probs, dim=1)
        pc = probs[:, self.idc, ...].type(torch.float32)
        dc = dist_maps[:, self.idc, ...].type(torch.float32)

        print('pc', pc)
        print('dc', dc)

        multipled = einsum("bcwh,bcwh->bcwh", pc, dc)

        loss = multipled.mean()

        return {"loss": loss}


