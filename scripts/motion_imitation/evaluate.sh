#! /bin/bash
name="exp_ReLU_base_iPER"
data_dir="/home/ronak1997_gmail_com/impersonator/train_data"  # need to be replaced!!!!!
output_dir="./outputs/results/evalOutput/${name}"  # need to be replaced!!!!!

##
gpu=0
gen_name="impersonator"
checkpoints_dir="./trained_models/${name}/"
## if use ImPer dataset trained model
#load_path="./outputs/checkpoints/lwb_imper/net_epoch_30_id_G.pth"

## if use ImPer and Place datasets trained model
#load_path="./outputs/checkpoints/lwb_imper_place/net_epoch_30_id_G.pth"

## if use ImPer, DeepFashion, and Place datasets trained model
load_path="./trained_models/${name}/net_epoch_20_id_G.pth"

## if use DeepFillv2 trained background inpainting network,
bg_model="./outputs/checkpoints/deepfillv2/net_epoch_50_id_G.pth"
## otherwise, it will use the BGNet in the original LiquidWarping GAN
#bg_model="ORIGINAL"

python evaluate.py --gpu_ids ${gpu} \
    --model swapper \
    --gen_name impersonator \
    --image_size 256 \
    --name ${name}  \
    --data_dir  ${data_dir}  \
    --checkpoints_dir ${checkpoints_dir} \
    --bg_model ${bg_model}      \
    --load_path ${load_path}    \
    --output_dir ${output_dir}  \
    --bg_ks 11 --ft_ks 3        \
    --has_detector  --post_tune

# --front_warp

