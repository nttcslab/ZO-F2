#! /bin/bash

eval "$(conda shell.bash hook)"
conda activate npzoo

COMMON_ARGS="seed=1,2,3 dataset=EMNIST suffix='emnist_lr_q/' epochs=300 \
zoo.max_dim=497 zoo.diff=central zoo.pre.alpha=3e-6 zoo.pre.tau=0.5 "
python main.py -m cuda_id=0 zoo.vectors=fromI lr=1e-4,2e-4,5e-4,1e-3 zoo.num_vectors=40 $COMMON_ARGS &
python main.py -m cuda_id=1 zoo.vectors=H1    lr=1e-4,2e-4,5e-4,1e-3 zoo.num_vectors=40 $COMMON_ARGS &
python main.py -m cuda_id=2 zoo.vectors=H2    lr=1e-4,2e-4,5e-4,1e-3 zoo.num_vectors=40 $COMMON_ARGS &
python main.py -m cuda_id=3 zoo.vectors=F2    lr=1e-4,2e-4,5e-4,1e-3 zoo.num_vectors=40 $COMMON_ARGS &
python main.py -m cuda_id=4 zoo.num_vectors=10 zoo.vectors=fromI,H1,H2,F2 lr=2e-4 $COMMON_ARGS &
python main.py -m cuda_id=5 zoo.num_vectors=20 zoo.vectors=fromI,H1,H2,F2 lr=2e-4 $COMMON_ARGS &
python main.py -m cuda_id=6 zoo.num_vectors=40 zoo.vectors=fromI,H1,H2,F2 lr=2e-4 $COMMON_ARGS &
python main.py -m cuda_id=7 zoo.num_vectors=80 zoo.vectors=fromI,H1,H2,F2 lr=2e-4 $COMMON_ARGS &
