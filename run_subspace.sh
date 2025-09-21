# multi-class unlearning
CUDA_VISIBLE_DEVICES=0 python3 main_subspace.py --arch swin_t --dataset tinyimagenet --data_dir ~/data/tiny-imagenet-200 \
    --save_dir ./timgnet_swint_subspace  --model_path ./timgnet_swint_pretrain/ckpt_pretrain.pth.tar \
    --class_forget --label_to_forget 44 65 150 168 --seed 0 \
    --unlearn subspace --subspace_s 250 --subspace_steps 50 --unlearn_lr 1 --weight_decay 0.05 

CUDA_VISIBLE_DEVICES=0 python3 main_subspace.py --arch swin_t --dataset tinyimagenet --data_dir ~/data/tiny-imagenet-200 \
    --save_dir ./timgnet_swint_subspace  --model_path ./timgnet_swint_pretrain/ckpt_pretrain.pth.tar \
    --class_forget --label_to_forget 53 57 108 179 --seed 0 \
    --unlearn subspace --subspace_s 250 --subspace_steps 50 --unlearn_lr 1 --weight_decay 0.05 

CUDA_VISIBLE_DEVICES=0 python3 main_subspace.py --arch swin_t --dataset tinyimagenet --data_dir ~/data/tiny-imagenet-200 \
    --save_dir ./timgnet_swint_subspace  --model_path ./timgnet_swint_pretrain/ckpt_pretrain.pth.tar \
    --class_forget --label_to_forget 11 83 115 153 --seed 0 \
    --unlearn subspace --subspace_s 250 --subspace_steps 50 --unlearn_lr 1 --weight_decay 0.05 

# single-class unlearning
CUDA_VISIBLE_DEVICES=0 python3 main_subspace.py --arch resnet50 --dataset imagenet --data_dir ~/data/imagenet \
    --save_dir ./imgnet_r50_subspace  \
    --class_forget --label_to_forget 97 --seed 0 \
    --unlearn subspace --subspace_s 500 --subspace_steps 100 --unlearn_lr 10 --weight_decay 0.05

CUDA_VISIBLE_DEVICES=0 python3 main_subspace.py --arch resnet50 --dataset imagenet --data_dir ~/data/imagenet \
    --save_dir ./imgnet_r50_subspace  \
    --class_forget --label_to_forget 316 --seed 0 \
    --unlearn subspace --subspace_s 500 --subspace_steps 100 --unlearn_lr 10 --weight_decay 0.05

CUDA_VISIBLE_DEVICES=0 python3 main_subspace.py --arch resnet50 --dataset imagenet --data_dir ~/data/imagenet \
    --save_dir ./imgnet_r50_subspace  \
    --class_forget --label_to_forget 852 --seed 0 \
    --unlearn subspace --subspace_s 500 --subspace_steps 100 --unlearn_lr 10 --weight_decay 0.05

# instance unlearning
CUDA_VISIBLE_DEVICES=0 python3 main_subspace.py --arch swin_t --dataset tinyimagenet --data_dir ~/data/tiny-imagenet-200 \
    --save_dir ./timgnet_swint_subspace  --model_path ./timgnet_swint_pretrain/ckpt_pretrain.pth.tar \
    --random_forget --num_to_forget 1000 --seed 0 \
    --unlearn subspace --subspace_s 300 --subspace_steps 100 --unlearn_lr 1 --weight_decay 0.05 

CUDA_VISIBLE_DEVICES=0 python3 main_subspace.py --arch swin_t --dataset tinyimagenet --data_dir ~/data/tiny-imagenet-200 \
    --save_dir ./timgnet_swint_subspace  --model_path ./timgnet_swint_pretrain/ckpt_pretrain.pth.tar \
    --random_forget --num_to_forget 1000 --seed 1 \
    --unlearn subspace --subspace_s 300 --subspace_steps 100 --unlearn_lr 1 --weight_decay 0.05 

CUDA_VISIBLE_DEVICES=0 python3 main_subspace.py --arch swin_t --dataset tinyimagenet --data_dir ~/data/tiny-imagenet-200 \
    --save_dir ./timgnet_swint_subspace  --model_path ./timgnet_swint_pretrain/ckpt_pretrain.pth.tar \
    --random_forget --num_to_forget 1000 --seed 2 \
    --unlearn subspace --subspace_s 300 --subspace_steps 100 --unlearn_lr 1 --weight_decay 0.05 

# extreme unlearning
CUDA_VISIBLE_DEVICES=0 python3 main_subspace.py --arch swin_t --dataset tinyimagenet --data_dir ~/data/tiny-imagenet-200 \
    --save_dir ./timgnet_swint_subspace  --model_path ./timgnet_swint_pretrain/ckpt_pretrain.pth.tar \
    --class_forget --num_label_to_forget 180 --seed 0 \
    --unlearn subspace --subspace_s 18 --subspace_steps 50 --unlearn_lr 1 --weight_decay 0.05 

CUDA_VISIBLE_DEVICES=0 python3 main_subspace.py --arch swin_t --dataset tinyimagenet --data_dir ~/data/tiny-imagenet-200 \
    --save_dir ./timgnet_swint_subspace  --model_path ./timgnet_swint_pretrain/ckpt_pretrain.pth.tar \
    --class_forget --num_label_to_forget 180 --seed 1 \
    --unlearn subspace --subspace_s 18 --subspace_steps 50 --unlearn_lr 1 --weight_decay 0.05 

CUDA_VISIBLE_DEVICES=0 python3 main_subspace.py --arch swin_t --dataset tinyimagenet --data_dir ~/data/tiny-imagenet-200 \
    --save_dir ./timgnet_swint_subspace  --model_path ./timgnet_swint_pretrain/ckpt_pretrain.pth.tar \
    --class_forget --num_label_to_forget 180 --seed 2 \
    --unlearn subspace --subspace_s 18 --subspace_steps 50 --unlearn_lr 1 --weight_decay 0.05 

