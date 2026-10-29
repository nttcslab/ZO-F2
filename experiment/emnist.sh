#! /bin/bash

eval "$(conda shell.bash hook)"
conda activate npzoo

COMMON_ARGS="seed=1,2,3,4,5 dataset=EMNIST suffix='emnist/' lr=2e-4 epochs=300 \
zoo.max_dim=497 zoo.num_vectors=40 zoo.diff=central zoo.pre.alpha=3e-6 zoo.pre.tau=0.5 "
python main.py -m cuda_id=4 zoo.vectors=fromI $COMMON_ARGS &
python main.py -m cuda_id=5 zoo.vectors=H1    $COMMON_ARGS &
python main.py -m cuda_id=6 zoo.vectors=H2    $COMMON_ARGS &
python main.py -m cuda_id=7 zoo.vectors=F2    $COMMON_ARGS &
