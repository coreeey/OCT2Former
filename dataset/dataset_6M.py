from torch.utils.data import Dataset
import csv
from .transform import *
import torchvision
import numpy as np

class myDataset(Dataset):  # 定义自己的数据类myDataset，继承的抽象类Dataset
    def __init__(self, data_root, target_root, crop_size, data_mode, k_fold=None, imagefile_csv=None, num_fold=None, data_root_aux=None, img_aug=False):
        self.crop_size = crop_size  # h, w
        self.data_root = data_root
        self.data_root_aux = data_root_aux
        self.target_root = target_root
        self.data_mode = data_mode
        self.img_aug = img_aug

        # if data_mode=='train':
        self.transforms = get_transforms_train_6M()
        # else:
        #     self.transforms = get_transforms_valid()

        # 若不交叉验证，直接读取data_root下文件列表
        if k_fold == None:
            with open(imagefile_csv, "r") as f:
                reader = csv.reader(f)
                image_files = list(reader)[0]
            if data_mode == "train":
                self.image_files = image_files[0:180] 
            elif data_mode == "val" :
                # self.image_files = image_files[200: 300]
                self.image_files = image_files[180: 200]
            else:
                self.image_files = image_files[200: 300]

            print(f"{data_mode} dataset: {len(self.image_files)}")

    def __len__(self):  # 定义自己的数据类，必须重写这个方法（函数）
        # 返回的数据的长度
        return len(self.image_files)

    def __getitem__(self, idx):  # 定义自己的数据类
        file = self.image_files[idx]
        file_name = os.path.splitext(file)[0]

        image_path = os.path.join(self.data_root, file)
#        image_path_aux = os.path.join(self.data_root_aux, file)
        label_path = os.path.join(self.target_root, file)

        image, label = fetch(image_path, label_path)  # 读取图片与mask，返回4D张量
        # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        # image_aux, _ = fetch(image_path_aux) 
        # image = np.dstack([image, image_aux])
        #print(image.shape)
        # if self.img_aug == True:  # 数据增强
        if self.data_mode == "train" and self.img_aug == True:  # 数据增强
            image_aug = self.transforms(image=image.astype(np.uint8),  mask=label.astype(np.uint8) )
            image = image_aug['image']
            label = image_aug['mask']
        image, label = convert_to_tensor(image, label)
        
        # -------标签处理-------
        label = (label == 255).float()  # 区分黑白
        # -------标签处理-------

        label = label.squeeze()

        return {
            "image": image,
            "label": label,
            "file": file}
    # ###
    @classmethod 
    def recover_size(self,pred,org_size, tp=False):
        #print(org_size)
        pred = pred.unsqueeze(0).float()
        pred = pred.unsqueeze(0)
        assert len(pred.size())==4
        if tp == True:
            #print("1",pred.size())
            pred = torchvision.transforms.functional.rotate(pred, -90, expand=True)
            #print("2",pred.size())

        pred = F.interpolate(pred, size=(org_size[0], org_size[1]))
        #print(pred.size())
        pred = pred.squeeze(0).squeeze(0)
        return pred

    @classmethod 
    def saveImg(self, img, save_dir,Gray=False):

        imgPath = save_dir
        grid = torchvision.utils.make_grid(img, nrow=8, padding=2, pad_value=0,
                                        normalize=False, range=None, scale_each=False)
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        im = Image.fromarray(ndarr)

        if Gray:
            im.convert('L').save(imgPath)  # Gray = 0.29900 * R + 0.58700 * G + 0.11400 * B
        else:
            im.save(imgPath)


class predict_Dataset(Dataset):
    def __init__(self, data_root, crop_size):
        super(predict_Dataset, self).__init__()
        self.data_root = data_root
        self.crop_size = crop_size

        self.files = os.listdir(data_root)
        print(f"pred dataset:{len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name, _ = os.path.splitext(self.files[idx])
        image_path = os.path.join(self.data_root, self.files[idx])

        image, _ = fetch(image_path)
        #image = image.convert("RGB")
        image_size = image.size  # w,h
        image, _ = convert_to_tensor(image)
        image, _ = resize(self.crop_size, image)

        return {
            "image": image,
            "file_name": file_name,
            "image_size": torch.tensor([image_size[1], image_size[0]])}