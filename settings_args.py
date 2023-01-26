import os
import time
import csv
import argparse
from dataset.dataset_3M import *
import ml_collections
import shutil

class basic_setting():

    mode = "train_test"     
    k_fold = None        
    start_fold = None   
    end_fold = None     

    dataset_file_list = "utils/OCTA_3M.csv"  
    data_root = './OCTA_3M/Projection Maps/OCTA(ILM_OPL)'
    data_root_aux = './OCTA_3M/Projection Maps/OCT(ILM_OPL)'
    target_root = './OCTA-500_ground_truth/OCTA_3M/GroundTruth'  # 标签
    test_path = r''

    crop_size = (304, 304)
    depths=[1, 1, 1]
    patch_size=[3, 3, 3]
    over_lap = [2, 2, 2]
    vit_dims=[64, 128, 256]

    token_dim = [128, 128, 128, 64, 64]   #128最好
    #token_dim = [400, 300, 200, 100]

    run_dir = "3M"                 
    val_step = 1                

    in_channel = 1
    n_class = 2
    network = "OCT2Former" 

    note = "careful about data root" 
    
    batch_size = 2

    class_weight = [0.5, 0.5]
    OHEM = False
    num_epochs = 100
    num_workers = 4
    aux = False
    aux_weight = 0.4
    dice_weight = 1
    lr = 5e-4
    #lr = 1e-3
    momentum = 0.9
    weight_decay = 1e-4
    cuda_id = "2"

    img_aug = False

    test_run_file = "test package" 
    label_names = []
    plot = True


    def __init__(self):
        if not os.path.exists("./runs"):
            os.mkdir("./runs")
        self.run_dir = "./runs/"+self.run_dir
        if not os.path.exists(self.run_dir):
            os.mkdir(self.run_dir)

        if self.mode == "train" or self.mode == "train_test":
            time_now = time.strftime("%Y-%m%d-%H%M_%S", time.localtime(time.time()))

            self.dir = os.path.join(self.run_dir, time_now+"_"+self.network+"_"+str(self.num_epochs)+"epoch"+"_"+self.note + f'_fold_{self.k_fold}')
            os.mkdir(self.dir)

            self.val_result_file = os.path.join(self.dir, "val_result.csv")
            with open(self.val_result_file, "a") as f:
                w = csv.writer(f)
                w.writerow(['fold', 'epoch', 'mDice', 'mIoU', 'mAcc', 'mAuc', 'mSens', 'mSpec', 'mBAcc'])

            self.log_dir = os.path.join(self.dir, "log")
            os.mkdir(self.log_dir)
            self.log_dir = [self.log_dir]

            self.checkpoint_dir = os.path.join(self.dir, "checkpoints")
            os.mkdir(self.checkpoint_dir)
            self.checkpoint_dir = [self.checkpoint_dir]

            if self.k_fold is not None:
                for i in range(self.k_fold):
                    cp_i_dir = os.path.join(self.checkpoint_dir[0], f"fold_{i+1}")
                    log_i_dir = os.path.join(self.log_dir[0], f"fold_{i+1}")
                    os.mkdir(cp_i_dir)
                    os.mkdir(log_i_dir)
                    self.checkpoint_dir.append(cp_i_dir)
                    self.log_dir.append(log_i_dir)
            self.logger(self.dir+"/train_log")

            shutil.copytree('./model', os.path.join(self.dir+ '/code', "model"), shutil.ignore_patterns(['.git', '__pycache__']))
            shutil.copytree('./utils', os.path.join(self.dir+ '/code', "utils"), shutil.ignore_patterns(['.git', '__pycache__']))
            shutil.copy('./train3M.py', os.path.join(self.dir + '/code', "train3M.py"), follow_symlinks=False)
            shutil.copy('./settings_3M.py', os.path.join(self.dir + '/code', "settings_3M.py"), follow_symlinks=False)           
            shutil.copy('./dataset/dataset_3M.py', os.path.join(self.dir + '/code', "dataset_3M.py"), follow_symlinks=False)
            shutil.copy('./dataset/transform.py', os.path.join(self.dir + '/code', "transform.py"), follow_symlinks=False)
        if self.mode == "test" or self.mode == "train_test":
            if self.mode == "test":
                self.dir = os.path.join(self.run_dir, self.test_run_file)

                self.val_result_file = os.path.join(self.dir, "val_result.csv")

                self.checkpoint_dir = [os.path.join(self.dir, "checkpoints")]
                if self.k_fold is not None:
                    for i in range(self.k_fold):
                        fold_i_dir = os.path.join(self.checkpoint_dir[0], f"fold_{i+1}")
                        self.checkpoint_dir.append(fold_i_dir)

            self.test_result_file = os.path.join(self.dir, "test_result.csv")
            with open(self.test_result_file, "w") as f:
                w = csv.writer(f)
                title = ['file', 'mDice'] + [name+"_dice" for name in self.label_names[1:]] + \
                        ['mIoU'] + [name + "_iou" for name in self.label_names[1:]] + \
                        ['mAcc'] + \
                        ['mSens'] + [name + "_sens" for name in self.label_names[1:]] + \
                        ['mSpec'] + [name + "_spec" for name in self.label_names[1:]]
                w.writerow(title)

            if self.plot:
                self.plot_save_dir = os.path.join(self.dir, "test_images")
                if not os.path.exists(self.plot_save_dir):
                    os.mkdir(self.plot_save_dir)

    def logger(self, file):
        with open(file, "a") as f:
            attrs = dir(self)
            for att in attrs:
                if ("__"or "test_" or "val_" or "root" or "logger" or "dir") not in att:
                    f.write(f'{str(att)}:    {str(getattr(self, att))}\n\n')
            f.close()