from settings_game import *
import torch.nn.functional as F
from torch.backends import cudnn
from utils import utils
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
from utils.loss import DiceLoss, OhemCrossEntropy, OhemCrossEntropy_per_image
from tqdm import tqdm
import csv
import random
import numpy as np
from PIL import Image
import time
from torchvision.utils import save_image
from DNN_printer import DNN_printer
from model.choose_model import seg_model
#from dataset.transform import resize

def main(args, num_fold=0):

    torch.set_num_threads(1)
    model = seg_model(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    #if args.mode == "train" and num_fold <= 1:
        #DNN_printer(model, (3, args.crop_size[0],args.crop_size[1]),args.batch_size)

    if args.mode == "train":
        train(model, device, args, num_fold=num_fold)

    elif args.mode == "test":
        if args.k_fold is not None:
            return test(model, device, args, num_fold=num_fold)
        else:
            test(model, device, args, num_fold=num_fold)
    else:
        raise NotImplementedError





def train(model, device, args, num_fold=0):
    dataset_train = myDataset(args.data_root, args.target_root, args.crop_size,  "train",
                                 k_fold=args.k_fold, imagefile_csv=args.dataset_file_list, num_fold=num_fold) 
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True, drop_last=True) 
    num_train_data = len(dataset_train) 
    dataset_val = myDataset(args.data_root, args.target_root, args.crop_size, "val",
                               k_fold=args.k_fold, imagefile_csv=args.dataset_file_list, num_fold=num_fold) 
    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True, drop_last=True) 
    num_train_val = len(dataset_val)  
    ####################################################################################################################
    writer = SummaryWriter(log_dir=args.log_dir[num_fold], comment=f'tb_log')

    #opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.finetune == True:
        ckp = torch.load(args.finetune_path, map_location='cpu')
        model.load_state_dict(ckp['state_dict'], strict=True)
        #opt.load_state_dict(ckp['opt'])

 
    if args.OHEM:
        criterion = OhemCrossEntropy(thres=0.8, min_kept=10000)
    else:
        criterion = nn.CrossEntropyLoss().to(device) #交叉熵
    criterion_dice = DiceLoss()#dice

    cp_manager = utils.save_checkpoint_manager(10) 
    #cutmix = utils.cutmix(1)
    step = 0
    for epoch in range(args.num_epochs):
        model.train()
        lr = utils.poly_learning_rate(args, opt, epoch)  

        with tqdm(total=num_train_data, desc=f'[Train] fold[{num_fold}/{args.k_fold}] Epoch[{epoch + 1}/{args.num_epochs} LR={lr:.8f}] ', unit='img') as pbar:
            for batch in dataloader_train:
                step += 1

                image = batch["image"]
                label = batch["label"]
                assert len(image.size()) == 4
                assert len(label.size()) == 3
                image = image.to(device, dtype=torch.float32)
                label = label.to(device, dtype=torch.long)
                #print(image.size(), label.size())
                #mixup
                #image, label_a, label_b, lam = utils.mixup(image, label, alpha=1)
                #image, label_a, label_b, lam = cutmix.cutmix(image, label)
                #print(image.size(), label_a.size())

                # 前向传播
                opt.zero_grad()
                outputs = model(image)
                main_out = outputs["main_out"]
              
                celoss = criterion(main_out, label)
                totall_loss = celoss
                #totall_loss = diceloss

                if "aux_out" in outputs.keys(): 
                    aux_losses = 0
                    # print(len(outputs["aux_out"]))
                    for aux_p in outputs["aux_out"]:
                        auxloss = criterion(aux_p, label) * args.aux_weight
                        totall_loss += auxloss
                        aux_losses += auxloss

                totall_loss.backward()
                opt.step()

                if step % 5 == 0:
                    writer.add_scalar("Train/CE_loss", celoss.item(), step)
                    #writer.add_scalar("Train/Dice_loss", diceloss.item(), step)
                    if args.aux:
                        writer.add_scalar("Train/aux_losses",aux_losses, step)
                    writer.add_scalar("Train/Totall_loss", totall_loss.item(), step)

                pbar.set_postfix(**{'loss': totall_loss.item()})  # 显示loss
                pbar.update(image.size()[0])


        if (epoch+1) % args.val_step == 0:

            mDice, mIoU, mAcc, mSensitivity, mSpecificity = val(model, dataloader_val, num_train_val, device, args)
            writer.add_scalar("Valid/Dice_val", mDice, step)
            writer.add_scalar("Valid/IoU_val", mIoU, step)
            writer.add_scalar("Valid/Acc_val", mAcc, step)
            writer.add_scalar("Valid/Sen_val", mSensitivity, step)
            writer.add_scalar("Valid/Spe_val", mSpecificity, step)

            val_result = [num_fold, epoch+1, mDice, mIoU, mAcc, mSensitivity, mSpecificity]
            with open(args.val_result_file, "a") as f:
                w = csv.writer(f)
                w.writerow(val_result)
            cp_manager.save(model, opt, os.path.join(args.checkpoint_dir[num_fold], f'CP_epoch{epoch + 1}.pth'), float(mAcc))


def val(model, dataloader, num_train_val,  device, args):
    all_dice = []
    all_iou = []
    all_acc = []
    all_sen = []
    all_spe = []
    model.eval()
    with torch.no_grad():
        with tqdm(total=num_train_val, desc=f'VAL', unit='img') as pbar:
            for batch in dataloader:
                image = batch["image"]
                label = batch["label"]
                assert len(image.size()) == 4
                assert len(label.size()) == 3
                image = image.to(device, dtype=torch.float32)
                label = label.to(device, dtype=torch.long)

                outputs = model(image)
                main_out = outputs["main_out"]
                main_out = torch.exp(main_out).max(dim=1)[1] 

                for b in range(image.size()[0]):
                    hist = utils.fast_hist(label[b, :, :], main_out[b, :, :], args.n_class)
                    dice, iou, acc, Sensitivity, Specificity = utils.cal_scores(hist.cpu().numpy())
                    all_dice.append(list(dice))
                    all_iou.append(list(iou))
                    all_acc.append([acc])
                    all_sen.append(list(Sensitivity))
                    all_spe.append(list(Specificity))
                pbar.update(image.size()[0])

    mDice = np.array(all_dice).mean()
    mIoU = np.array(all_iou).mean()
    mAcc = np.array(all_acc).mean()
    mSensitivity = np.array(all_sen).mean()
    mSpecificity = np.array(all_spe).mean()

    dice = list(np.array(all_dice).mean(axis=0))
    print(f"dice{dice}")
    print(f'\r   [VAL] mDice:{mDice:0.2f}, mIoU:{mIoU:0.2f}, mAcc:{mAcc:0.2f}, mSen:{mSensitivity:0.2f}, mSpec:{mSpecificity:0.2f}')

    return mDice, mIoU, mAcc, mSensitivity, mSpecificity



def test(model, device, args, num_fold=0):

    if os.path.exists(args.val_result_file):
        with open(args.val_result_file, "r") as f:
            reader = csv.reader(f)
            val_result = list(reader)
        best_epoch = utils.best_model_in_fold(val_result, num_fold, k=4)
    else:
        best_epoch = args.num_epochs

    model_dir = os.path.join(args.checkpoint_dir[num_fold], f'CP_epoch{best_epoch}.pth')
    model.load_state_dict(torch.load(model_dir, map_location=device)["state_dict"])
    print(f'\rtest model loaded: [fold:{num_fold}] [best_epoch:{best_epoch}]')

    dataset_test = myDataset(args.data_root, args.target_root, args.crop_size, "test",
                                k_fold=args.k_fold, imagefile_csv=args.dataset_file_list, num_fold=num_fold)
    dataloader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    all_dice = []
    all_iou = []
    all_acc = []
    all_sen = []
    all_spe = []
    model.eval()
    with torch.no_grad():
        with tqdm(total=len(dataset_test), desc=f'TEST fold {num_fold}/{args.k_fold}', unit='img') as pbar:
            for batch in dataloader:
                image = batch["image"]
                label = batch["label"]

                file = batch["file"]
                org_size = batch["size_org"]
                tp = batch["transpose"]
                assert len(image.size()) == 4
                assert len(label.size()) == 3
                image = image.to(device, dtype=torch.float32)
                label = label.to(device, dtype=torch.long)

                outputs = model(image)
                pred = outputs["main_out"]
                pred = torch.exp(pred).max(dim=1)[1] 
                for b in range(image.size()[0]):

                    hist = utils.fast_hist(label[b,:,:], pred[b,:,:], args.n_class)
                    dice, iou, acc, Sensitivity, Specificity = utils.cal_scores(hist.cpu().numpy(), smooth=0.01)
                    test_result = [file[b], dice.mean()]+list(dice)+[iou.mean()]+list(iou)+[acc] + \
                        [Sensitivity.mean()]+list(Sensitivity)+[Specificity.mean()]+list(Specificity)
                    with open(args.test_result_file, "a") as f:
                        w = csv.writer(f)
                        w.writerow(test_result)
                   
                    all_dice.append(list(dice))
                    all_iou.append(list(iou))
                    all_acc.append([acc])
                    all_sen.append(list(Sensitivity))
                    all_spe.append(list(Specificity))
                    if args.plot:
                        file_name, _ = os.path.splitext(file[b])
                        pred1 = myDataset.recover_size(pred=pred[b], org_size=org_size[b], tp=tp[b])
                        #save_image(pred1.cpu().float().unsqueeze(0), os.path.join(args.plot_save_dir, file_name + f"24.png"), normalize=True)
                        myDataset.saveImg(img=pred1.cpu().float().unsqueeze(0), save_dir=os.path.join(args.plot_save_dir, file_name + f".png"),  Gray=True)
                        #label1 = myDataset.recover_size(pred=label[b], org_size=org_size[b])
                        #myDataset.saveImg(img=label1.cpu().float().unsqueeze(0), save_dir=os.path.join(args.plot_save_dir, file_name + f"label.png"),  Gray=True)



                pbar.update(image.size()[0])

    print(f"\r---------Fold {num_fold} Test Result---------")
    print(f'mDice: {np.array(all_dice).mean()}')
    print(f'mIoU:  {np.array(all_iou).mean()}')
    print(f'mAcc:  {np.array(all_acc).mean()}')
    print(f'mSens: {np.array(all_sen).mean()}')
    print(f'mSpec: {np.array(all_spe).mean()}')

    if num_fold == 0:
        utils.save_print_score(all_dice, all_iou, all_acc, all_sen, all_spe, args.test_result_file, args.label_names)
        return

    return all_dice, all_iou, all_acc, all_sen, all_spe



if __name__ == "__main__":

    seed = 12345
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.empty_cache()
    cudnn.benchmark = True

    args = basic_setting()
    assert args.k_fold != 1
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_id

    if (not os.path.exists(args.dataset_file_list)) and (args.k_fold is not None):
        utils.get_dataset_filelist(args.data_root, args.dataset_file_list)

    mode = args.mode
    if args.k_fold is None:
        print("k_fold is None")
        if mode == "train_test":
            args.mode = "train"
            print("###################### Train Start & " +  args.network  + " ######################")
            main(args)
            args.mode = "test"
            print("###################### Test Start & " +  args.network  + " ######################")
            main(args)
        else:
            main(args)
    else:
        if mode == "train_test":
            print("###################### Train & Test Start & "+  args.network  + " ######################")

        if mode == "train" or mode == "train_test":
            args.mode = "train"
            print("###################### Train Start & " +  args.network  + " ######################")
            for i in range(args.start_fold, args.end_fold):
                torch.cuda.empty_cache()
                main(args, num_fold=i + 1)

        if mode == "test" or mode == "train_test":
            args.mode = "test"
            print("###################### Test Start & " + args.network + " ######################")
            all_dice = []
            all_iou = []
            all_acc = []
            all_sen = []
            all_spe = []
            for i in range(args.start_fold, args.end_fold):
                Dice, IoU, Acc, Sensitivity, Specificity = main(args, num_fold=i + 1)
                all_dice += Dice
                all_iou += IoU
                all_acc += Acc
                all_sen += Sensitivity
                all_spe += Specificity
            utils.save_print_score(all_dice, all_iou, all_acc, all_sen, all_spe, args.test_result_file, args.label_names)







