load_path=$1

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

echo $load_path
echo $min_epoch
echo $max_epoch


i=$min_epoch
while [ $i -le $max_epoch ]
    do
        full_path="${load_path}net_epoch_${i}_id_G.pth"
        python demo_imitator.py --gpu_ids 0 --load_path ${full_path}
        i=$(( i+step ))
    done
