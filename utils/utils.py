
import torch
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
import csv
import os
import random
from PIL import Image
import SimpleITK as sitk
from sklearn import metrics
import cv2

def fast_hist(label_true, label_pred, n_class, ROSE=False):
    '''
    :param label_true: 0 ~ n_class (batch, h, w)
    :param label_pred: 0 ~ n_class (batch, h, w)
    :param n_class: 类别数
    :return: 对角线上是每一类分类正确的个数，其他都是分错的个数
    '''

#    assert n_class > 1
    if ROSE:
        FP, FN, TP, TN = numeric_score(label_pred, label_true)
        return torch.tensor(np.array([FP, FN, TP, TN])) 
    if n_class == 1:
        n_class = 2
    mask = (label_true >= 0) & (label_true < n_class)
    hist = torch.bincount(
        n_class * label_true[mask].int() + label_pred[mask].int(),
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)

    return hist

# 计算指标
def cal_scores(hist, smooth=0.001):
    if len(hist) == 4:
        FP, FN, TP, TN = hist
    else:
        TP = np.diag(hist)
        FP = hist.sum(axis=0) - TP
        FN = hist.sum(axis=1) - TP
        TN = hist.sum() - TP - FP - FN
    union = TP + FP + FN

    dice = (2*TP+smooth) / (union+TP+smooth)

    iou = (TP+smooth) / (union+smooth)

    # Precision = np.diag(hist).sum() / hist.sum()   # 分类正确的准确率  acc

    Sensitivity = (TP+smooth) / (TP+FN+smooth)  # recall/TPR

    #Specificity = (TN+smooth) / (FP+TN+smooth) #TNR
    Specificity = (TN+smooth) / (FP+TN+smooth) #TNR
    
    BACC = (Sensitivity+Specificity)/2
    # print(dice, iou)
    if len(hist) == 4:
        return dice*100, iou*100,  Sensitivity*100, Sensitivity*100, Specificity*100, BACC*100

    return dice[1:]*100, iou[1:]*100,  Sensitivity[1:]*100, Sensitivity[1:]*100, Specificity[1:]*100, BACC[1:]*100

def numeric_score(pred_arr, gt_arr, kernel_size=(3, 3)):  # DCC & ROSE-2: kernel_size=(3, 3)
    """Computation of statistical numerical scores:

    * FP = False Positives
    * FN = False Negatives
    * TP = True Positives
    * TN = True Negatives

    return: tuple (FP, FN, TP, TN)
    """
    pred_arr = np.array(pred_arr.squeeze(0).cpu().numpy()*255, np.uint8)
    gt_arr = np.array(gt_arr.cpu().numpy()*255, np.uint8)

    # pred_arr = cv2.resize(pred_arr, (512, 512))

    # print(pred_arr.shape, gt_arr.shape)    
    pred_arr[pred_arr > 128] = 1
    gt_arr[gt_arr > 128] = 1

    #print(pred_arr.shape, gt_arr.shape,type(pred_arr))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)#椭圆结构
    # pred_arr = cv2.erode(pred_arr, kernel, iterations=1)
    dilated_gt_arr = cv2.dilate(gt_arr, kernel, iterations=1)#//255
    # pred_arr = pred_arr//255
    # cv2.imwrite(r'/data1/tanxiao/Segmentation-master/runs/ROSE2/dilated.png', dilated_gt_arr)
    # cv2.imwrite(r'/data1/tanxiao/Segmentation-master/runs/ROSE2/pre.png', pred_arr)
    # cv2.imwrite(r'/data1/tanxiao/Segmentation-master/runs/ROSE2/gt.png', gt_arr)
    FP = np.float(np.sum(np.logical_and(pred_arr == 1, dilated_gt_arr == 0)))
    FN = np.float(np.sum(np.logical_and(pred_arr == 0, gt_arr == 1)))
    TP = np.float(np.sum(np.logical_and(pred_arr == 1, dilated_gt_arr == 1)))
    TN = np.float(np.sum(np.logical_and(pred_arr == 0, gt_arr == 0)))
    
    return FP, FN, TP, TN

def calc_dice(pred_arr, gt_arr, kernel_size=(3, 3)):  # DCC & ROSE-2: kernel_size=(3, 3)

    FP, FN, TP, TN = numeric_score(pred_arr, gt_arr, kernel_size)
    dice = 2.0 * TP / (FP + FN + 2.0 * TP + 1e-12)
    
    return dice*100

def extract_mask(pred_arr, gt_arr, mask_arr=None):
    # we want to make them into vectors
    pred_vec = pred_arr.cpu().numpy().flatten()
    gt_vec = gt_arr.cpu().numpy().flatten()
    
    if mask_arr is not None:
        mask_vec = mask_arr.flatten()
        idx = list(np.where(mask_vec == 0)[0])
        
        pred_vec = np.delete(pred_vec, idx)
        gt_vec = np.delete(gt_vec, idx)
    
    return pred_vec, gt_vec


def calc_auc(pred_arr, gt_arr, mask_arr=None):
    pred_vec, gt_vec = extract_mask(pred_arr, gt_arr, mask_arr=mask_arr)
    roc_auc = metrics.roc_auc_score(gt_vec, pred_vec)
    
    return roc_auc*100


# 保存打印指标
def save_print_score(all_dice, all_iou, all_acc, all_auc, all_sen, all_spe, all_bacc, file, label_names):
    all_dice = np.array(all_dice)
    all_iou = np.array(all_iou)
    all_acc = np.array(all_acc)
    all_auc = np.array(all_auc)
    all_sen = np.array(all_sen)
    all_spe = np.array(all_spe)
    all_bacc = np.array(all_bacc)
    test_mean = ["mean"]+[all_dice.mean()] + list(all_dice.mean(axis=0)) + \
                [all_iou.mean()] + list(all_iou.mean(axis=0)) + \
                [all_acc.mean()] + \
                [all_auc.mean()] + \
                [all_sen.mean()] + list(all_sen.mean(axis=0)) + \
                [all_spe.mean()] + list(all_spe.mean(axis=0)) + \
                [all_bacc.mean()] + list(all_bacc.mean(axis=0)) 
    test_std = ["std"]+[all_dice.std()] + list(all_dice.std(axis=0)) + \
               [all_iou.std()] + list(all_iou.std(axis=0)) + \
               [all_acc.std()] + \
               [all_auc.std()] + \
               [all_sen.std()] + list(all_sen.std(axis=0)) + \
               [all_spe.std()] + list(all_spe.std(axis=0)) + \
               [all_bacc.std()] + list(all_bacc.std(axis=0))
    label_names = label_names[1:]
    title = [' ', 'mDice'] + [name + "_dice" for name in label_names] + \
            ['mIoU'] + [name + "_iou" for name in label_names] + \
            ['mAcc'] + \
            ['mAuc'] + \
            ['mSens'] + [name + "_sen" for name in label_names] + \
            ['mSpec'] + [name + "_spe" for name in label_names] + \
            ['mBACC'] + [name + "_bacc" for name in label_names] 
    with open(file, "a") as f:
        w = csv.writer(f)
        w.writerow(["Test Result"])
        w.writerow(title)
        w.writerow(test_mean)
        w.writerow(test_std)

    print("\n##############Test Result##############")
    print(f'mDice: {all_dice.mean()}')
    print(f'mIoU:  {all_iou.mean()}')
    print(f'mAcc:  {all_acc.mean()}')
    print(f'mAuc:  {all_auc.mean()}')
    print(f'mSens: {all_sen.mean()}')
    print(f'mSpec: {all_spe.mean()}')
    print(f'mBAcc: {all_bacc.mean()}')

# def save_print_score_rose(all_dice, all_iou, all_acc, all_auc, all_sen, all_spe, all_bacc, file, label_names):
#     all_dice = np.array(all_dice)
#     all_iou = np.array(all_iou)
#     all_acc = np.array(all_acc)
#     all_auc = np.array(all_auc)
#     all_sen = np.array(all_sen)
#     all_spe = np.array(all_spe)
#     all_bacc = np.array(all_bacc)
#     test_mean = ["mean"]+[all_dice.mean()] + list(all_dice.mean(axis=0)) + \
#                 [all_iou.mean()] + list(all_iou.mean(axis=0)) + \
#                 [all_acc.mean()] + \
#                 [all_auc.mean()] + \
#                 [all_sen.mean()] + list(all_sen.mean(axis=0)) + \
#                 [all_spe.mean()] + list(all_spe.mean(axis=0)) + \
#                 [all_bacc.mean()] + list(all_bacc.mean(axis=0)) 
#     test_std = ["std"]+[all_dice.std()] + list(all_dice.std(axis=0)) + \
#                [all_iou.std()] + list(all_iou.std(axis=0)) + \
#                [all_acc.std()] + \
#                [all_auc.std()] + \
#                [all_sen.std()] + list(all_sen.std(axis=0)) + \
#                [all_spe.std()] + list(all_spe.std(axis=0)) + \
#                [all_bacc.std()] + list(all_bacc.std(axis=0))
#     label_names = label_names[1:]
#     title = [' ', 'mDice'] + [name + "_dice" for name in label_names] + \
#             ['mIoU'] + [name + "_iou" for name in label_names] + \
#             ['mAcc'] + \
#             ['mAuc'] + \
#             ['mSens'] + [name + "_sen" for name in label_names] + \
#             ['mSpec'] + [name + "_spe" for name in label_names] + \
#             ['mBACC'] + [name + "_bacc" for name in label_names] 
#     with open(file, "a") as f:
#         w = csv.writer(f)
#         w.writerow(["Test Result"])
#         w.writerow(title)
#         w.writerow(test_mean)
#         w.writerow(test_std)

#     print("\n##############Test Result##############")
#     print(f'mDice: {all_dice.mean()}')
#     print(f'mIoU:  {all_iou.mean()}')
#     print(f'mAcc:  {all_acc.mean()}')
#     print(f'mAuc:  {all_auc.mean()}')
#     print(f'mSens: {all_sen.mean()}')
#     print(f'mSpec: {all_spe.mean()}')
#     print(f'mBAcc: {all_bacc.mean()}')
def save_print_score_rose(all_dice, all_iou, all_acc, all_auc, all_sen, all_spe, all_bacc, file, label_names):
    all_dice = np.array(all_dice)
    all_iou = np.array(all_iou)
    all_acc = np.array(all_acc)
    all_auc = np.array(all_auc)
    all_sen = np.array(all_sen)
    all_spe = np.array(all_spe)
    all_bacc = np.array(all_bacc)
    test_mean = ["mean"]+[all_dice.mean()] + \
                [all_iou.mean()] + \
                [all_acc.mean()] + \
                [all_auc.mean()] + \
                [all_sen.mean()] + \
                [all_spe.mean()] + \
                [all_bacc.mean()] 
    test_std = ["std"]+[all_dice.std()] + \
               [all_iou.std()] + \
               [all_acc.std()] + \
               [all_auc.std()] + \
               [all_sen.std()] + \
               [all_spe.std()] + \
               [all_bacc.std()]
    label_names = label_names[1:]
    title = [' ', 'mDice'] + [name + "_dice" for name in label_names] + \
            ['mIoU'] + [name + "_iou" for name in label_names] + \
            ['mAcc'] + \
            ['mAuc'] + \
            ['mSens'] + [name + "_sen" for name in label_names] + \
            ['mSpec'] + [name + "_spe" for name in label_names] + \
            ['mBACC'] + [name + "_bacc" for name in label_names] 
    with open(file, "a") as f:
        w = csv.writer(f)
        w.writerow(["Test Result"])
        w.writerow(title)
        w.writerow(test_mean)
        w.writerow(test_std)

    print("\n##############Test Result##############")
    print(f'mDice: {all_dice.mean()}')
    print(f'mIoU:  {all_iou.mean()}')
    print(f'mAcc:  {all_acc.mean()}')
    print(f'mAuc:  {all_auc.mean()}')
    print(f'mSens: {all_sen.mean()}')
    print(f'mSpec: {all_spe.mean()}')
    print(f'mBAcc: {all_bacc.mean()}')

# 从验证指标中选择最优的epoch
def best_model_in_fold(val_result, num_fold, k=2):
    best_epoch = 0
    best_dice = 0
    for row in val_result:
        if str(num_fold) in row:
            if best_dice < float(row[k]):
                best_dice = float(row[k])
                best_epoch = int(row[1])
    return best_epoch


# 读取数据集目录内文件名，保存至csv文件
def get_dataset_filelist(data_root, save_file):
    file_list = os.listdir(data_root)
    random.shuffle(file_list)
    with open(save_file, 'w') as f:
        w = csv.writer(f)
        w.writerow(file_list)


def poly_learning_rate(args, optimizer, epoch):
    """
    Sets the learning rate to the initial LR decayed by 10 every 30 epochs(step = 30)
    """
    step_per_epoch=90
    #if epoch <= 200:
    lr = args.lr * (1 - epoch / args.num_epochs) ** 0.9
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    # else:         
    #     for param_group in optimizer.param_groups:
    #         lr = 0.0002
    #         param_group['lr'] = lr
    return lr




# one hot转成0,1,2,..这样的标签
def make_class_label(mask):
    b = mask.size()[0]
    mask = mask.view(b, -1)
    class_label = torch.max(mask, dim=-1)[0]
    return class_label


# 把0,1,2...这样的类别标签转化为one_hot
def make_one_hot(targets, num_classes):
    targets = targets.unsqueeze(1)
    label = []
    for i in range(num_classes):
        label.append((targets == i).float())
    label = torch.cat(label, dim=1)
    return label



# 保存训练过程中最大的checkpoint
class save_checkpoint_manager:
    def __init__(self, max_save=5):
        self.checkpoints = {}
        self.max_save = max_save

    def save(self, model, opt, path, score):
        if len(self.checkpoints) < self.max_save:
            self.checkpoints[path] = score
            torch.save({'state_dict': model.state_dict(), 'opt':opt.state_dict()}, path)
        else:
            min_value = min(self.checkpoints.values())
            print(min_value)
            if score > min_value:
                for i in self.checkpoints.keys():
                    if self.checkpoints[i] == min_value:
                        min_key = i
                        break
                os.remove(min_key)
                self.checkpoints.pop(min_key)
                self.checkpoints[path] = score
                torch.save({'state_dict': model.state_dict(), 'opt':opt.state_dict()}, path)


def mixup(images, target, alpha=1):
    lam = np.random.beta(alpha,alpha)
    index = torch.randperm(images.size(0)).cuda()
    newinputs = lam*images + (1-lam)*images[index,:]
    targets_a, targets_b = target, target[index]
    return newinputs, targets_a, targets_b, lam

class cutmix():
    def __init__(self,  alpha=1):
        self.alpha = alpha

    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        """1.论文里的公式2，求出B的rw,rh"""
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        """2.论文里的公式2，求出B的rx,ry（bbox的中心点）"""
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        # 限制坐标区域不超过样本大小

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        """3.返回剪裁B区域的坐标值"""
        return bbx1, bby1, bbx2, bby2

    def cutmix(self, image, target):
        lam = np.random.beta(self.alpha, self.alpha)
        index = torch.randperm(image.size(0))
        targets_a, targets_b = target, target[index]
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(image.size(), lam)
        image[:, :, bbx1:bbx2, bby1:bby2] = image[index, :, bbx1:bbx2, bby1:bby2]
        new_input = image
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (image.size()[-1] * image.size()[-2]))
        return new_input, targets_a, targets_b, lam 




def slices2volume_mask(original_volume_dir, pred_mask_dir, out_dir):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    volume_filenames = sorted(os.path.splitext(file)[0] for file in  os.listdir(original_volume_dir))
    mask_files = sorted(os.listdir(pred_mask_dir))

    for vfile in volume_filenames:
        volume_mask = []
        slices_files = [mfile for mfile in mask_files if vfile in mfile]
        num = len(slices_files)
        for i in range(num):
            file = vfile+f"_{i}.png"
            image = np.array(np.array(Image.open(os.path.join(pred_mask_dir, file))) == 255, np.uint16)
            image = np.expand_dims(image, axis=0)
            volume_mask.append(image)
        # 保存volume mask
        volume_mask = np.concatenate(volume_mask, axis=0)
        volume_mask = sitk.GetImageFromArray(volume_mask)
        sitk.WriteImage(volume_mask, os.path.join(out_dir, vfile+'.nii.gz'))