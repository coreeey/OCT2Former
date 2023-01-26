![Python 3.9](https://img.shields.io/badge/python-3.6-green.svg)
<span id="jump1"></span>
# OCT2Former
The official code of OCT2Former: A retinal OCT-angiography vessel segmentation transformer
## The Code is being organized.

Prerequisites
* python3
* numpy
* pillow
* opencv-python
* scikit-learn
* tensorboardX
* visdom
* pytorch
* torchvision

<span id="jump2"></span>
### FOR OCTA-SS dataset
python train.py --mode "train_test" --dataset "SS" --data_root = "package to save OCTA-SS"

### FOR ROSE1 dataset
python train.py --mode "train_test" --dataset "ROSE1" --data_root = "package to save ROSE"

### FOR OCTA-6M dataset
python train.py --mode "train_test" --dataset "6M" --data_root = "package to save OCTA-6M"

### FOR OCTA-3M dataset
python train.py --mode "train_test" --dataset "3M" --data_root = "package to save OCTA-3M"
