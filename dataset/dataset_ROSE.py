from torch.utils.data import Dataset
import csv
from .transform import *
import pandas as pd

class myDataset(Dataset): 
    def __init__(self, data_root, target_root, crop_size, data_mode, k_fold=None, imagefile_csv=None, num_fold=None, data_root_aux=None, img_aug=False):
        self.crop_size = crop_size 
        self.data_root = data_root
        self.target_root = target_root
        self.data_mode = data_mode
        self.img_aug = img_aug

        if data_mode=='train':
            self.transforms = get_transforms_train_ROSE()
        else:
            self.transforms = get_transforms_valid()

        self.image_files = pd.read_csv(imagefile_csv, header=None).squeeze().tolist()
 
        if data_mode=='train':
            self.image_files = self.image_files[:27]
        elif data_mode=='val':
            self.image_files = self.image_files[27:]
        elif data_mode=='test':
            self.image_files = os.listdir(data_root.replace('train', 'test'))
            self.data_root = data_root.replace('train', 'test')
            self.target_root = target_root.replace('train', 'test')

        print(f"{data_mode} dataset: {len(self.image_files)}")

    def __len__(self):  
        return len(self.image_files)

    def __getitem__(self, idx): 
        file = self.image_files[idx]
        file_name = os.path.splitext(file)[0]

        image_path = os.path.join(self.data_root, file)
        label_path = os.path.join(self.target_root, file)

        image, label = fetch(image_path, label_path) 

        if self.data_mode == "train" and self.img_aug == True: 
            image_aug = self.transforms(image=image.astype(np.uint8), mask=label.astype(np.uint8))
            image = image_aug['image']
            label = image_aug['mask']

        image, label = convert_to_tensor(image, label)
        label = (label == 255).float() 

        label = label.squeeze()

        return {
            "image": image,
            "label": label,
            "file": file}
 