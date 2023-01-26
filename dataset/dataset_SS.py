from torch.utils.data import Dataset
import csv
from .transform import *
from utils.loss import  class2one_hot, one_hot2dist

class myDataset(Dataset):  # 定义自己的数据类myDataset，继承的抽象类Dataset
    def __init__(self, data_root, target_root, crop_size, data_mode, k_fold=None, imagefile_csv=None, num_fold=None, img_aug=False):
        self.crop_size = crop_size  # h, w
        self.data_root = data_root
        self.target_root = target_root
        self.data_mode = data_mode
        self.img_aug = img_aug

        if data_mode=='train':
            self.transforms = get_transforms_train_OCTA_RSS()
        else:
            self.transforms = get_transforms_valid()

        # 若不交叉验证，直接读取data_root下文件列表
        if k_fold == None:
            with open(imagefile_csv, "r") as f:
                reader = csv.reader(f)
                image_files = list(reader)[0]
            if data_mode == "train":
                self.image_files = image_files[0: 27]
            elif data_mode == "val" :
                 self.image_files = image_files[27: 30]
            else:
                self.image_files = image_files[30: 55]

            print(f"{data_mode} dataset: {len(self.image_files)}")

    def __len__(self):  # 定义自己的数据类，必须重写这个方法（函数）
        # 返回的数据的长度
        return len(self.image_files)

    def __getitem__(self, idx):  # 定义自己的数据类
        file = self.image_files[idx]
        file_name = os.path.splitext(file)[0]

        #image_path = os.path.join(self.data_root, 'images/original_images/'+ file_name + '.tif')
        image_path = os.path.join(self.data_root, 'images/original_images/'+ str(file))

        #image_path_aux = os.path.join(self.data_root ,file)
        label_path = os.path.join(self.target_root, file_name) + '.png'
        #print(image_path, label_path)
        image, label = fetch(image_path, label_path)  # 读取图片与mask，返回4D张量 
        #image_aux, _ = fetch(image_path_aux)

        if self.data_mode == "train" and self.img_aug == True:  # 数据增强
            image_aug = self.transforms(image=image.astype(np.uint8), mask=label.astype(np.uint8))
            image = image_aug['image']
            label = image_aug['mask']#.unsqueeze(0)
        #image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        #image_aux = cv2.cvtColor(image_aux, cv2.COLOR_RGB2GRAY)

        image, label = convert_to_tensor(image, label)
        
        #image_aux, _ = convert_to_tensor(image_aux)
        #print(image_aux.size(), image.size())
        #print(image.size(),label.size())
        #image = torch.cat((image, image_aux), dim=0)
        # -------标签处理-------
        label = (label == 255).float()  # 区分黑白
        # -------标签处理-------

        #image, label = resize(self.crop_size, image, label)


        # if self.data_mode == "train" and self.img_aug == True:  # 数据增强
        #     image, label = random_Top_Bottom_filp(image, label)
        #     image, label = random_Left_Right_filp(image, label)
        ##########
        ##########

        label = label.squeeze()
        # data2 = class2one_hot(label, 2)
        # data2 = data2[0].numpy()
        # data3 = one_hot2dist(data2)   #bcwh
        return {
            "image": image,
            "label": label,
            "file": file}
    # ###


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
