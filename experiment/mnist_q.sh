#! /bin/bash

eval "$(conda shell.bash hook)"
conda activate zopre

COMMON_ARGS="seed=1,2,3,4,5 dataset=MNIST suffix='mnist_q/' optimizer=SGD lr=1e-3 epochs=200 \
num_features=80 zoo.max_dim=800 zoo.num_vectors=2,4,6,8,10 zoo.diff=central \
zoo.pre.gamma=0.01 zoo.pre.tau=0.5 zoo.pre.report_error=true "
python main.py -m cuda_id=6 zoo.vectors=H2 zoo.pre.alpha=2e-4 $COMMON_ARGS &
python main.py -m cuda_id=7 zoo.vectors=F2 zoo.pre.alpha=2e-4 $COMMON_ARGS &
