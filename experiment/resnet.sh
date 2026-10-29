#! /bin/bash

eval "$(conda shell.bash hook)"
conda activate npzoo

COMMON_ARGS="seed=1,2,3,4,5 dataset=CIFAR10 suffix='resnet/' lr=1e-3 epochs=300 \
zoo.max_dim=498 zoo.num_vectors=200 zoo.diff=forward \
zoo.pre.gamma=0.01 zoo.pre.alpha=1e-5 zoo.pre.tau=0.5 zoo.pre.Tud=500 "
python main.py -m cuda_id=4 zoo.vectors=fromI zoo.pre.warm_up=0    $COMMON_ARGS &
python main.py -m cuda_id=5 zoo.vectors=H2    zoo.pre.warm_up=0    $COMMON_ARGS &
python main.py -m cuda_id=6 zoo.vectors=F2    zoo.pre.warm_up=0    $COMMON_ARGS &
python main.py -m cuda_id=7 zoo.vectors=F2    zoo.pre.warm_up=1000 $COMMON_ARGS &
