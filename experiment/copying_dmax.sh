#! /bin/bash

eval "$(conda shell.bash hook)"
conda activate zopre

COMMON_ARGS="dataset=copying suffix='copying_dmax/' lr=1e-4 \
epochs=100 zoo.num_vectors=20 zoo.diff=central \
zoo.pre.gamma=0.1 zoo.pre.alpha=1e-5 zoo.pre.tau=0.5 zoo.vectors=F2 "
# python main.py -m cuda_id=0 zoo.max_dim=1    $COMMON_ARGS &
# python main.py -m cuda_id=1 zoo.max_dim=4    $COMMON_ARGS &
# python main.py -m cuda_id=2 zoo.max_dim=13   $COMMON_ARGS &
# python main.py -m cuda_id=3 zoo.max_dim=39   $COMMON_ARGS &
python main.py -m cuda_id=4 seed=5 zoo.max_dim=528  $COMMON_ARGS &
python main.py -m cuda_id=5 seed=5 zoo.max_dim=1584 $COMMON_ARGS &
python main.py -m cuda_id=6 seed=5 zoo.max_dim=2112 $COMMON_ARGS &
python main.py -m cuda_id=7 seed=5 zoo.max_dim=2640 $COMMON_ARGS &
