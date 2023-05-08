import os
import time
import csv
import argparse
import ml_collections
import shutil

class basic_setting():
    parser = argparse.ArgumentParser()
    # add arguments to the parser
    parser.add_argument("--dataset", type=str, default="OCTA-3M")
    parser.add_argument("--mode", type=str, default="train_test")
    parser.add_argument("--k_fold", type=int, default=None)
    parser.add_argument("--start_fold", type=int, default=0)
    parser.add_argument("--end_fold", type=int, default=1)
    parser.add_argument("--dataset_file_list", type=str, default="utils/OCTA_3M.csv")
    parser.add_argument("--data_root", type=str, default="./OCTA-500_ground_truth/OCTA_3M/Projection Maps/OCTA(ILM_OPL)")
    parser.add_argument("--data_root_aux", type=str, default="")
    parser.add_argument("--target_root", type=str, default="./OCTA-500_ground_truth/OCTA_3M/GroundTruth")
    parser.add_argument("--test_path", type=str, default="")
    parser.add_argument("--crop_size", type=tuple, default=(304, 304))
    parser.add_argument("--depths", type=list, default=[1, 2, 3])
    parser.add_argument("--patch_size", type=list, default=[2, 2, 2])
    parser.add_argument("--vit_dims", type=list, default=[64, 128, 256, 512])
    parser.add_argument("--token_dim", type=list, default=[128, 128, 128])
    parser.add_argument("--run_dir", type=str, default="3M")
    parser.add_argument("--val_step", type=int, default=1)
    parser.add_argument("--in_channel", type=int, default=1)
    parser.add_argument("--n_class", type=int, default=2)
    parser.add_argument("--network", type=str, default="OCT2Former")
    parser.add_argument("--note", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--class_weight", type=list, default=[0.5, 0.5])
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--aux", action="store_true")
    parser.add_argument("--aux_weight", type=float, default=0.4)
    parser.add_argument("--spec_interpolation", action="store_true")
    parser.add_argument("--dice_weight", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--cuda_id", type=str, default="0")
    parser.add_argument("--img_aug", action="store_true")
    parser.add_argument("--tt_aug", type=bool, default=True)
    parser.add_argument("--test_run_file", type=str, default="")
    parser.add_argument("--label_names", type=list, default=[])
    parser.add_argument("--plot", action="store_true")

    args = parser.parse_args()
    params = vars(args)

    def __init__(self):
        for k, v in self.params.items():
            setattr(self, k, v)

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

if __name__ == '__main__':
    args = basic_setting()