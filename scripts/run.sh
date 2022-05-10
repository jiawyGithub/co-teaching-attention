export noise_type="symmetric"
#export noise_type="pairflip"
#export noise_type="asymmetric"
#export noise_type="tridiagonal"

export dataset="cifar10"
#export dataset="cifar100"
# export dataset="mnist"
#export dataset="F-MNIST"
#export dataset="SVHN"
#export dataset="clothing1M"

export gpu=2
export noise_rate=0.5
export model="hamnet"
export date="0510"

CUDA_VISIBLE_DEVICES=${gpu} nohup python -u main.py --dataset ${dataset} --noise_type ${noise_type} --noise_rate ${noise_rate} --gpu ${gpu} > ./${dataset}_${noise_type}_${noise_rate}_${model}_${date}.log 2>&1 & 
# CUDA_VISIBLE_DEVICES=${gpu} python -u main.py --dataset ${dataset} --noise_type ${noise_type} --noise_rate ${noise_rate} --gpu ${gpu} 