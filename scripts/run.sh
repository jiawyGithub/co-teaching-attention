# export noise_type="symmetric"
export noise_type="pairflip"
#export noise_type="asymmetric"
#export noise_type="tridiagonal"

export dataset="cifar10"
#export dataset="cifar100"
# export dataset="mnist"
#export dataset="F-MNIST"
#export dataset="SVHN"
#export dataset="clothing1M"

export gpu=0
export noise_rate=0.45
export attentionType="coord_external"
export date="0515"
export layer="3_1"

CUDA_VISIBLE_DEVICES=${gpu} nohup python -u main.py --dataset ${dataset} --noise_type ${noise_type} --noise_rate ${noise_rate} --attentionType ${attentionType} --gpu ${gpu} > ./${dataset}_${noise_type}_${noise_rate}_${attentionType}_${layer}_${date}.log 2>&1 & 
# CUDA_VISIBLE_DEVICES=${gpu} python -u main.py --dataset ${dataset} --noise_type ${noise_type} --noise_rate ${noise_rate} --gpu ${gpu} 