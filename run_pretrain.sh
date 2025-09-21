CUDA_VISIBLE_DEVICES=7 python3 main_pretrain.py --arch swin_t --dataset tinyimagenet --data_dir '~/data/tiny-imagenet-200' \
    --seed 0 --lr 1e-4 --epochs 20 --optim adamw --weight_decay 0.05 --batch_size 128 \
    --save_dir './timgnet_swint_pretrain' \