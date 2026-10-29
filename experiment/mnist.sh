#! /bin/bash

eval "$(conda shell.bash hook)"
conda activate npzoo

COMMON_ARGS="seed=1,2,3,4,5 dataset=MNIST suffix='mnist/' optimizer=SGD lr=1e-3 epochs=200 \
zoo.max_dim=800 zoo.num_vectors=10 zoo.diff=central zoo.pre.gamma=0.01 zoo.pre.tau=0.5 zoo.pre.report_error=true "
python main.py -m cuda_id=0 zoo.vectors=fromI num_features=32,48,64,80 zoo.pre.alpha=1e-4 $COMMON_ARGS &
python main.py -m cuda_id=1 zoo.vectors=H1    num_features=32,48,64,80 zoo.pre.alpha=1e-4 $COMMON_ARGS &
python main.py -m cuda_id=2 zoo.vectors=H2    num_features=32,48,64,80 zoo.pre.alpha=1e-4 $COMMON_ARGS &
python main.py -m cuda_id=3 zoo.vectors=F2    num_features=32,48,64,80 zoo.pre.alpha=1e-4 $COMMON_ARGS &
python main.py -m cuda_id=4 zoo.vectors=fromI num_features=80 zoo.pre.alpha=1e-4,2e-4,5e-4,1e-3 $COMMON_ARGS &
python main.py -m cuda_id=5 zoo.vectors=H1    num_features=80 zoo.pre.alpha=1e-4,2e-4,5e-4,1e-3 $COMMON_ARGS &
python main.py -m cuda_id=6 zoo.vectors=H2    num_features=80 zoo.pre.alpha=1e-4,2e-4,5e-4,1e-3 $COMMON_ARGS &
python main.py -m cuda_id=7 zoo.vectors=F2    num_features=80 zoo.pre.alpha=1e-4,2e-4,5e-4,1e-3 $COMMON_ARGS &
