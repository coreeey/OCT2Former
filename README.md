![Python 3.9](https://img.shields.io/badge/Python-3.9-3776AB)
<span id="jump1"></span>
# OCT2Former
The official code of OCT2Former: A retinal OCT-angiography vessel segmentation transformer

### Prerequisites
* python3
* numpy
* pillow
* opencv-python
* scikit-learn
* tensorboardX
* visdom
* pytorch
* torchvision
* pandas

<span id="jump2"></span>
### FOR OCTA-SS dataset
 > python train.py  --dataset='OCTA-SS' \
 --num_epochs=100 \
 --dataset_file_list='utils/OCTA-SS.csv' \
 --data_root=$OCTA-SS-DATA-PATH \
 --target_root=$OCTA-SS-LABEL-PATH  \
 --run_dir='OCTA-SS' \
 --in_channel=1 \
 --batch_size=2 \
 --lr=5e-4 \
 --spec_interpolation \
 --img_aug 
 
 OR
 
 > sh trainSS.sh


### FOR ROSE1 dataset
 > python train.py --dataset='ROSE' \
 --num_epochs=100 \
 --dataset_file_list='utils/ROSE-1.csv' \
 --data_root=$ROSE-1-SS-DATA-PATH \
 --target_root=$ROSE-1-THICK-LABEL-PATH \
 --run_dir='ROSE-1' \
 --in_channel=1 \
 --batch_size=2 \
 --lr=5e-4 \
 --img_aug 
 
 OR
 
 > sh trainROSE.sh
 
 
### FOR OCTA-6M dataset
 > python train.py  
 --dataset='OCTA-6M' \
 --num_epochs=100 \
 --dataset_file_list='utils/OCTA_6M.csv' \
 --data_root=$OCTA-6M-OCTA-DATA-PATH  \
 --data_root_aux=$OCTA-6M-OCT-DATA-PATH \
 --target_root=$OCTA-6M-LABEL-PATH \
 --run_dir='6M' \
 --in_channel=2 \
 --batch_size=2 \
 --lr=5e-4 \
 --img_aug \
 --cuda_id=6
 
 OR
 
 > sh train6M.sh

### FOR OCTA-3M dataset
 > python train.py  
 --dataset='OCTA-3M' \
 --num_epochs=100 \
 --dataset_file_list='utils/OCTA_3M.csv' \
 --data_root=$OCTA-3M-OCTA-DATA-PATH  \
 --data_root_aux=$OCTA-3M-OCT-DATA-PATH \
 --target_root=$OCTA-3M-LABEL-PATH \
 --run_dir='3M' \
 --in_channel=2 \
 --batch_size=2 \
 --lr=5e-4 \
 --img_aug \
 
 OR
 
 > sh train3M.sh
