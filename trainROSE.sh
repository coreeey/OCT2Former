 python train.py  --dataset='ROSE' \
 --num_epochs=100 \
 --dataset_file_list='utils/ROSE-1.csv' \
 --data_root='ROSE-1/SVC/train/img' \
 --target_root='ROSE-1/SVC/train/thick_gt' \
 --run_dir='ROSE-1' \
 --in_channel=1 \
 --batch_size=2 \
 --lr=5e-4 \
 --img_aug \
 --cuda_id=6 