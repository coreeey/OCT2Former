 python train.py  --dataset='OCTA-3M' \
 --num_epochs=100 \
 --dataset_file_list='utils/OCTA_3M.csv' \
 --data_root='OCTA_3M/Projection Maps/OCTA(ILM_OPL)' \
 --data_root_aux='OCTA_3M/Projection Maps/OCT(ILM_OPL)' \
 --target_root='OCTA_3M/GroundTruth' \
 --run_dir='3M' \
 --in_channel=2 \
 --batch_size=2 \
 --lr=5e-4 \
 --img_aug \
 --cuda_id=6