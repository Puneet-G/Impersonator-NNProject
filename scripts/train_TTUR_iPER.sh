#! /bin/bash

# basic configs
gpu_ids=2,3     # if using multi-gpus, increasing the batch_size
#gpu_ids=0

# dataset configs
dataset_model=iPER  # use iPER dataset
data_dir=/home/ronak1997_gmail_com/impersonator/train_data  # need to be replaced!!!!!
images_folder=images
smpls_folder=smpls
train_ids_file=train.txt
test_ids_file=val.txt

# saving configs
checkpoints_dir=/home/ronak1997_gmail_com/impersonator/trained_models   # directory to save models, need to be replaced!!!!!
name=expC2_LRec15_g4-4_d1-4_dec15_1-5_iPER   # the directory is ${checkpoints_dir}/name, which is used to save the checkpoints.
loss_path="trained_models/${name}/loss_log2.txt"
trained_models_path="trained_models/${name}/"
first_gpu=${gpu_ids:0:1}
inf_out_path="./outputs/results/runOutput/plaid_shirt/${name}"
inf_out_zip_path="${inf_out_path}.zip"
graph_output="./plots/${name}"

# model configs
model=impersonator_trainer
gen_name=impersonator
image_size=256

# training configs
load_path="None"
batch_size=24
lambda_rec=15.0
lambda_tsf=10.0
lambda_face=5.0
lambda_style=0.0
lambda_mask=1.0
#lambda_mask=2.5
lambda_mask_smooth=1.0
nepochs_no_decay=15  # fixing learning rate when epoch ranges in [0, 5]
nepochs_decay=15    # decreasing the learning rate when epoch ranges in [6, 25+5]
lr_G=0.0004
lr_D=0.0001
final_lr=0.00001

python train.py --gpu_ids ${gpu_ids}        \
    --data_dir  ${data_dir}                 \
    --images_folder    ${images_folder}     \
    --smpls_folder     ${smpls_folder}      \
    --checkpoints_dir  ${checkpoints_dir}   \
    --train_ids_file   ${train_ids_file}    \
    --test_ids_file    ${test_ids_file}     \
    --load_path        ${load_path}         \
    --model            ${model}             \
    --gen_name         ${gen_name}          \
    --name             ${name}              \
    --dataset_mode     ${dataset_model}     \
    --image_size       ${image_size}        \
    --batch_size       ${batch_size}        \
    --lambda_face      ${lambda_face}       \
    --lambda_tsf       ${lambda_tsf}        \
    --lambda_style     ${lambda_style}      \
    --lambda_rec       ${lambda_rec}         \
    --lambda_mask      ${lambda_mask}       \
    --lambda_mask_smooth  ${lambda_mask_smooth} \
    --lr_G             ${lr_G}              \
    --lr_D             ${lr_D}              \
    --final_lr         ${final_lr}          \
    --nepochs_no_decay ${nepochs_no_decay}  --nepochs_decay ${nepochs_decay}  \
    --mask_bce     --use_vgg       --use_face  # --spectral_norm     # --label_smooth   # --gradient_penalty

python plot_graph.py --path ${loss_path}
sudo chmod -R 777 ${graph_output}
zip -r plots.zip plots
sudo chmod -R 777 plots.zip

./scripts/run_inf2.sh ${name} 5 30 5 ${first_gpu}
sudo chmod -R 777 ${inf_out_path}
zip -r ${inf_out_zip_path}  ${inf_out_path}

echo Inference run is complete. The results are at: $inf_out_zip_path
