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

def get_transforms_train_OCTA_SS():
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


def fetch(image_path, label_path=None):
    #image = Image.open(image_path).convert('L')
    image = cv2.imread(image_path, 0)
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
