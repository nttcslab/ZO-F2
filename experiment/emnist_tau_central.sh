#! /bin/bash

eval "$(conda shell.bash hook)"
conda activate npzoo

COMMON_ARGS="seed=1,2,3 dataset=EMNIST suffix='emnist_tau_central/' lr=2e-4 epochs=300 \
zoo.max_dim=497 zoo.num_vectors=40 zoo.diff=central zoo.vectors=F2 zoo.pre.alpha=1e-5,2e-5,3e-5 "
python main.py -m cuda_id=0 zoo.pre.tau=1e-3 $COMMON_ARGS &
python main.py -m cuda_id=1 zoo.pre.tau=1e-2 $COMMON_ARGS &
python main.py -m cuda_id=2 zoo.pre.tau=1e-1 $COMMON_ARGS &
python main.py -m cuda_id=3 zoo.pre.tau=2e-1 $COMMON_ARGS &
python main.py -m cuda_id=4 zoo.pre.tau=5e-1 $COMMON_ARGS &
python main.py -m cuda_id=5 zoo.pre.tau=7e-1 $COMMON_ARGS &
python main.py -m cuda_id=6 zoo.pre.tau=8e-1 $COMMON_ARGS &
python main.py -m cuda_id=7 zoo.pre.tau=9e-1 $COMMON_ARGS &
