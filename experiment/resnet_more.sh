#! /bin/bash

eval "$(conda shell.bash hook)"
conda activate npzoo

COMMON_ARGS="dataset=CIFAR10r suffix='resnet_more/' lr=1e-3 epochs=3000 \
zoo.max_dim=498 zoo.num_vectors=200 zoo.pre.initFscale=1 zoo.diff=forward zoo.pre.report_error=false \
zoo.pre.alpha=1e-5 zoo.pre.gamma=0.01 zoo.pre.tau=0.5 zoo.pre.Tud=500 "
python main.py -m cuda_id=0 seed=1 save_file=fromI.pt zoo.vectors=fromI zoo.pre.warm_up=0    $COMMON_ARGS &
python main.py -m cuda_id=1 seed=1 save_file=H2.pt    zoo.vectors=H2    zoo.pre.warm_up=0    $COMMON_ARGS &
python main.py -m cuda_id=2 seed=1 save_file=F2.pt    zoo.vectors=F2    zoo.pre.warm_up=0    $COMMON_ARGS &
python main.py -m cuda_id=3 seed=1 save_file=F2w.pt   zoo.vectors=F2    zoo.pre.warm_up=1000 $COMMON_ARGS &
python main.py -m cuda_id=4 seed=2 save_file=fromI.pt zoo.vectors=fromI zoo.pre.warm_up=0    $COMMON_ARGS &
python main.py -m cuda_id=5 seed=2 save_file=H2.pt    zoo.vectors=H2    zoo.pre.warm_up=0    $COMMON_ARGS &
python main.py -m cuda_id=6 seed=2 save_file=F2.pt    zoo.vectors=F2    zoo.pre.warm_up=0    $COMMON_ARGS &
python main.py -m cuda_id=7 seed=2 save_file=F2w.pt   zoo.vectors=F2    zoo.pre.warm_up=1000 $COMMON_ARGS &
