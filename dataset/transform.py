import torch
from torch.nn import functional as F
from PIL import Image
import numpy as np
import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw, PIL.ImageFilter
import os
import cv2
from albumentations import (Compose, HorizontalFlip, VerticalFlip, Rotate, RandomRotate90,
                            ShiftScaleRotate, ElasticTransform,
                            GridDistortion, RandomSizedCrop, RandomCrop, CenterCrop,
                            RandomBrightnessContrast, HueSaturationValue, IAASharpen,
                            RandomGamma, RandomBrightness, RandomBrightnessContrast,
                            GaussianBlur,CLAHE,
                            Cutout, CoarseDropout, GaussNoise, ChannelShuffle, ToGray, OpticalDistortion,
                            Normalize, OneOf, NoOp)
from albumentations import pytorch as a

Image.MAX_IMAGE_PIXELS = None

def get_transforms_train():
    transform_train = Compose([
        #Basic
        #Rotate(limit=10, p=0.5),
        #RandomRotate90(p=0.5),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        #GaussNoise(var_limit=(0,50.0), mean=0, p=0.5)
        #RandomCrop(91,91)
        #Morphology
        #Rotate(p=0.5)
        # ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=(-30,30), 
        #                   interpolation=1, border_mode=0, value=(0,0,0), p=0.5),
        # GaussNoise(var_limit=(0,50.0), mean=0, p=0.5),
        # GaussianBlur(blur_limit=(3,7), p=0.5),
        
        # #Color
        # RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.5, 
        #                         brightness_by_max=True,p=0.5),
        # HueSaturationValue(hue_shift_limit=30, sat_shift_limit=30, 
        #                    val_shift_limit=0, p=0.5),
        #Normalize(),
        
        #a.transforms.ToTensorV2(),
    ])
    return transform_train

def get_transforms_train_6M():
    transform_train = Compose([
        #Basic
        Rotate(limit=10, p=0.5),
        HorizontalFlip(p=0.5),  
        VerticalFlip(p=0.5),
        #GaussNoise(var_limit=(0,50.0), mean=0, p=0.5),
        #GaussianBlur(blur_limit=(3,7), p=0.5),
    ])
    return transform_train

def get_transforms_train_3M():
    transform_train = Compose([
        #Basic
        Rotate(limit=10, p=0.5),
        HorizontalFlip(p=0.5),  
        VerticalFlip(p=0.5),
        # GaussNoise(var_limit=(0,50.0), mean=0, p=0.5),
        # GaussianBlur(blur_limit=(3,7), p=0.5),
        # RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.5, 
        #                         brightness_by_max=True,p=0.5),
        # HueSaturationValue(hue_shift_limit=30, sat_shift_limit=30, 
        #                    val_shift_limit=0, p=0.5),
    ])
    return transform_train

def get_transforms_train_ROSE():
    transform_train = Compose([
        #Basic
        Rotate(limit=10, p=0.5),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
    ])
    return transform_train

def get_transforms_train_OCTA_RSS():
    transform_train = Compose([
        #Basic
        Rotate(limit=10, p=0.5),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
    ])
    return transform_train

def get_transforms_valid():
    transform_valid = Compose([
        #Normalize(),
	#RandomCrop(91, 91)
        #a.transforms.ToTensorV2(),
    ] )
    return transform_valid

# 随机旋转
def random_Rotate(img, label=None):
    rand = int(float(torch.rand(1)-0.5)*60)
    img = img.rotate(rand)
    if label is not None:
        label = label.rotate(rand)
    return img, label

# 随机对比度
def random_Contrast(img):
    v = float(torch.rand(1)) * 2
    if 0.5 <= v <= 1.5:
        return PIL.ImageEnhance.Contrast(img).enhance(v)
    else:
        return img

# 随机颜色鲜艳或灰暗
def random_Color(img):
    v = float(torch.rand(1)) * 2
    if 0.4 <= v <= 1.5:
        return PIL.ImageEnhance.Color(img).enhance(v)
    else:
        return img

# 随机亮度变换
def random_Brightness(img):  # [0.1,1.9]
    v = float(torch.rand(1)) * 2
    if 0.4 <= v <= 1.5:
        return PIL.ImageEnhance.Brightness(img).enhance(v)
    else:
        return img

# 随机高斯模糊
def random_GaussianBlur(img, img_aux=None):
    p = float(torch.rand(1))
    if p > 0.6:
        v = float(torch.rand(1))+1.2
        return img.filter(PIL.ImageFilter.GaussianBlur(radius=v))
    else:
        return img

# 随机变换
def random_transfrom(image, label=None, img_aux=None):
    #image = random_Color(image)
    image, label = random_Rotate(image, label) #
    #image = random_Contrast(image)#
    #image = random_Brightness(image)#
    image = random_GaussianBlur(image)#
    return image, label


# 读取图片与mask，返回4D张量
def fetch(image_path, label_path=None):
    #image = Image.open(image_path).convert('L')
    image = cv2.imread(image_path, 0) #0灰度
    #(h, w) = image.size
    #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if label_path is not None:
        if os.path.exists(label_path):
            #label = Image.open(label_path)
            label = cv2.imread(label_path, 0)
        # else:
        #     label = None
    else:
        label = "_"
    #tp = False
    #if h > w:
        #image = image.transpose(Image.ROTATE_90)
        #label = label.transpose(Image.ROTATE_90)
        #tp = True
    #print(image.size)
    return image, label

# image转为tensor
def convert_to_tensor(image, label=None):
    #mean = torch.Tensor(np.array([0.65459856,0.48386562,0.69428385]))
    #std = torch.Tensor(np.array([0.15167958,0.23584107,0.13146145]))

    image = torch.FloatTensor(np.array(image)) / 255
    #image = (image - mean)/std
    if len(image.size()) == 2:
        image = image.unsqueeze(0)
    else:
        image = image.permute(2, 0, 1)

    if label is not None:
        label = torch.FloatTensor(np.array(label))

        if len(label.size()) == 2:
            label = label.unsqueeze(0)
    else:
        label = torch.zeros((1, image.size()[1], image.size()[2]))
    return image, label

# 根据比例resize
def scale(crop_ratio, image, label=None):
    size_h, size_w = image.size()[-2:]
    size = (int(crop_ratio*size_h), int(crop_ratio*size_w))

    image = F.interpolate(image.unsqueeze(0), size = size, mode='bilinear', align_corners=True).squeeze(0) #？？？？？？？？？？？？？
    if label is not None:
        label = F.interpolate(label.unsqueeze(0), size = size, mode='nearest').squeeze(0)
    return image, label


def scale_adaptive(crop_size, image, label=None):
    image_size = image.size()[-2:]
    ratio_h = float(crop_size[0] / image_size[0])
    ratio_w = float(crop_size[1] / image_size[1])
    ratio = min(ratio_h, ratio_w)
    size = (int(image_size[0]*ratio), int(image_size[1]*ratio))

    image = F.interpolate(image.unsqueeze(0), size=size, mode='bilinear', align_corners=True).squeeze(0)
    if label is not None:
        label = F.interpolate(label.unsqueeze(0), size=size, mode='nearest').squeeze(0)
    return image, label


# resize
def resize(crop_size, image, label=None):
    image = F.interpolate(image.unsqueeze(0), size=crop_size, mode='bilinear', align_corners=True).squeeze(0)
    if label is not None:
        label = F.interpolate(label.unsqueeze(0), size=crop_size, mode='nearest').squeeze(0)
    return image, label 


# 随机裁剪
def random_crop(crop_size, image, label=None):
    assert len(image.size()) == 3
    h, w = image.size()[-2:]
    delta_h = h-crop_size[0]
    delta_w = w-crop_size[1]

    sh = int(torch.rand(1)*delta_h) if delta_h > 0 else 0
    sw = int(torch.rand(1)*delta_w) if delta_w > 0 else 0
    eh = crop_size[0]+sh if delta_h > 0 else h
    ew = crop_size[1]+sw if delta_w > 0 else w
    
    image = image[:, sh:eh, sw:ew]
    if label is not None:
        assert len(label.size()) == 3
        label = label[:, sh:eh, sw:ew]

    return image, label


# 若原图小于裁剪图，填充
def pad(crop_size, image, label=None, pad_value=0.0):
    h, w = image.size()[-2:]
    pad_h = max(crop_size[0] - h, 0)
    pad_w = max(crop_size[1] - w, 0)
    if pad_h > 0 or pad_w > 0:
        image = F.pad(image, (0, pad_w, 0, pad_h), mode='constant', value=pad_value)
        if label is not None:
            label = F.pad(label, (0, pad_w, 0, pad_h), mode='constant', value=pad_value)
    return image, label


# 随机垂直翻转
def random_Top_Bottom_filp(image, label=None, p=0.5):
    a = float(torch.rand(1))
    if a > p:
        image = torch.flip(image, [1])
        if label is not None:
            if len(label.size()) == 2:
                label = label.unsqueeze(0)
            label = torch.flip(label, [1])
    return image, label


# 随机水平翻转
def random_Left_Right_filp(image, label=None, p=0.5):
    a = float(torch.rand(1))
    if a > p:
        image = torch.flip(image, [2])
        if label is not None:
            if len(label.size()) == 2:
                label = label.unsqueeze(0)
            label = torch.flip(label, [2])
    return image, label
