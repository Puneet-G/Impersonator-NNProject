exp_name=$1
gpu_id=$5

echo $gpu_id

if [ -z "$2" ] || [ -z "$3" ]
then
    min_epoch=5
    max_epoch=30
else
    min_epoch=$2
    max_epoch=$3
fi

if [ -z "$4" ] 
then
    step=5
else
    step=$4
fi

echo $exp_name
echo $min_epoch
echo $max_epoch

load_path="./trained_models/${exp_name}/"
output_dir="./outputs/results/runOutput/imitator/${exp_name}/"
echo $load_path
echo $output_dir

i=$min_epoch
while [ $i -le $max_epoch ]
    do
        full_path="${load_path}net_epoch_${i}_id_G.pth"
        epoch_output="${output_dir}epoch_${i}/"
        python run_imitator.py --gpu_ids ${gpu_id}                                     \
            --load_path ${full_path} --model imitator                                  \
            --output_dir ${epoch_output}                                               \
            --src_path ./assets/src_imgs/beachshorts.jpg                               \
            --tgt_path ./assets/samples/refs/iPER/024_8_2                              \
            --bg_ks 7 --ft_ks 3 --has_detector --post_tune --front_warp --save_res     
        i=$(( i+step ))
    done

zip_name="${exp_name}.zip"
sudo chmod -R 777 ${output_dir}
