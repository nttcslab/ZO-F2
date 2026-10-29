#! /bin/bash

eval "$(conda shell.bash hook)"
conda activate npzoo

COMMON_ARGS="seed=1,2,3,4,5 dataset=copying suffix='copying_cond/' lr=1e-4 epochs=100 \
zoo.max_dim=453 zoo.num_vectors=20 zoo.diff=central \
zoo.pre.gamma=0.1 zoo.pre.alpha=1e-5,3e-5 zoo.pre.tau=0.5 "
python main.py -m cuda_id=0 zoo.vectors=F2 zoo.pre.update_type=whiten     $COMMON_ARGS &
python main.py -m cuda_id=1 zoo.vectors=F2 zoo.pre.update_type=solve_clip $COMMON_ARGS &
