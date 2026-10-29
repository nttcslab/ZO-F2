#! /bin/bash

eval "$(conda shell.bash hook)"
conda activate zopre

COMMON_ARGS="seed=1,2,3 zoo.vectors=F2 dataset=EMNIST suffix='emnist_empirical/' lr=2e-4 epochs=300 \
zoo.max_dim=497 zoo.num_vectors=40 zoo.diff=central  zoo.pre.tau=0.5 "
python main.py -m cuda_id=2 zoo.pre.empiricalFIM=false zoo.pre.alpha=2e-5 $COMMON_ARGS &
python main.py -m cuda_id=3 zoo.pre.empiricalFIM=true  zoo.pre.alpha=2e-5 $COMMON_ARGS &
python main.py -m cuda_id=6 zoo.pre.empiricalFIM=false zoo.pre.alpha=3e-6 $COMMON_ARGS &
python main.py -m cuda_id=7 zoo.pre.empiricalFIM=true  zoo.pre.alpha=3e-6 $COMMON_ARGS &
