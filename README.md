# ZO-F2: Low-Variance Fisher Preconditioner via Bilinear Estimation for Zeroth-Order Optimization

All source code necessary to perform the experiments described in [the above titled manuscript presented at NeurIPS 2026](https://neurips.cc/virtual/2026/loc/sydney/poster/153380).

## Directories
```
supplementary_code
├─ compile
├─ experiment
│  ├─ task
│  └─ train
├─ log
│  ├─ copying
│  ├─ emnist
│  └─ ...
└─ plot
```
- experiment: python code and shell scripts for the experiments
- experiment/task: python code for the models and datasets
- experiment/train: python code for zeroth-order optimization
- log: to store execution log files in subdirectories, e.g., copying, emnist.
- module_chip_core: custom CUDA kernels and C++ code for computationally efficient simulation of optical neural networks
- plot: python code for generating figures and tables

## Quick procedure to generate figures and table 
There are already [log files](./log) of the experiments produced by the authors.
From the log files, one can quickly generate figures and table

1. Prepare a python environment, e.g., Miniconda
   ```bash
   conda create -n zopre python=3.11.12
   conda activate zopre
   ```
2. Install the required packages
   ```bash
   pip install -r requirements_plot.txt
   ```
3. To generate figures and table, please go to the directory `plot`, and run the corresponding python script. An example is:
   ```bash
   cd plot; python mnist.py
   ```
   When the environment does not have tex or latex installed, please refer to [tex](#tex).

### Files in plot
- [table3.py](./plot/table3.py): Table 3, overall results on test accuracies
- [mnist.py](./plot/mnist.py): Figure 2(a) (mnist.pdf)
- [emnist.py](./plot/emnist.py): Figure 2(b) (emnist_central.pdf), Figure 4(b) (emnist_forward.pdf)
- [copying.py](./plot/copying.py): Figure 2(c) (copying.pdf)
- [resnet.py](./plot/resnet.py): Figure 2(d) (resnet.pdf)
- [mnist_q.py](./plot/mnist_q.py): Figure 3 (mnist_q.pdf)
- [emnist_lr_q.py](./plot/emnist_lr_q.py): Figure 4(a) (emnist_lr_q.pdf)
- [emnist_empirical.py](./plot/emnist_empirical.py): Figure 4(c) (emnist_empirical.pdf)
- [copying_dmax.py](./plot/copying_dmax.py): Table 5

## Procedure
One can reproduce the experimental results by the following procedure.

1. Prepare a python environment, e.g., Miniconda
   ```bash
   conda create -n zopre python=3.11.12
   conda activate zopre
   ```
2. Install the required packages
   ```bash
   pip install -r requirements_experiment.txt
   ``` 
3. Install pytorch by the instructions in https://pytorch.org/ or https://pytorch.org/get-started/previous-versions/ (the following command is an install command of pytorch=2.6.0 for an environment CUDA 11.8 installed).
   ```bash
   pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
   ```
4. (Optional, for the Copying memory tasks) Compile CUDA and C++ codes in [Compile](./compile/):
   ```bash
   cd compile; pip install --no-build-isolation --editable .
   ```
   The built package `mzi_onn_sim` is imported by the python files [experiment/task/chip.py](./experiment/task/chip.py).
5. Run the proposed method (for the MNIST task, by the setting of [config.yaml](./experiment/config.yaml)):
   ```bash
   cd experiment; python main.py
   ```
6. To reproduce all the experimental results, one needs to run all the shell scripts in the directory `experiment`. An example of running a shell script is:
   ```bash
   cd experiment; bash mnist.sh
   ```
   This example shell script [mnist.sh](./experiment/mnist.sh) assumes that the environment has 8 GPUs, `cuda_id` from 0 to 7, and takes advantage of parallel execution.
   When the environment does not have that amount of GPUs, please refer to [none or less GPUs](#none-or-less-gpus).

### Files in experiment

#### Shell script for Experiments
- [mnist.sh](./experiment/mnist.sh): Table 3 (Linear, MNIST), Figure 2(a)
- [emnist.sh](./experiment/emnist.sh): Table 3 (CNN, EMNIST)
- [copying.sh](./experiment/copying.sh): Table 3 (RNN, Copying)
- [resnet.sh](./experiment/resnet.sh): Table 3 (ResNet, CIFAR-10)
- [emnist_tau_central.sh](./experiment/emnist_tau_central.sh): Figure 2(b)
- [copying_cond.sh](./experiment/copying_cond.sh): Figure 2(c)
- [resnet_more.sh](./experiment/resnet_more.sh): Figure 2(d)
- [mnist_q.sh](./experiment/mnist_q.sh): Figure 3 (Linear, MNIST)
- [emnist_lr_q.sh](./experiment/emnist_lr_q.sh): Figure 4(a)
- [emnist_tau_forward.sh](./experiment/emnist_tau_forward.sh): Figure 4(b)
- [emnist_empirical.sh](./experiment/emnist_empirical.sh): Figure 4(c)
- [copying_dmax.sh](./experiment/copying_dmax.sh): Table 5

#### Python code for the main program
- [main.py](./experiment/main.py): running ZO-I, ZO-H1, ZO-H2, ZO-F2 depending on [config.yaml](./experiment/config.yaml) and the arguments passed to main.py

#### Python code related to the proposed method in `experiment/train`
- [train_zo.py](./experiment/train/train_zo.py): Zeroth-Order (ZO) optimization 
- [precon.py](./experiment/train/precon.py): Perturbation preconditioner 

## Trouble shooting
### none or less GPUs
If the environment does not have any GPU, i.e., without CUDA, please run a python program as
```bash
python main.py device=cpu
```
If the environment does not have 8 GPUs, please modify a shell script to accommodate the number of GPUs.
For example, if there are 2 GPUs, [emnist.sh](./experiment/emnist.sh) should be modified to
```bash
python main.py -m cuda_id=0 zoo.vectors=fromI,H2 $COMMON_ARGS &
python main.py -m cuda_id=1 zoo.vectors=H1,F2    $COMMON_ARGS &
```
If a shell script is modified and therefore the mapping from `cuda_id` to the experimental condition changes, one should also modify the corresponding plot file in [Files in plot](#files-in-plot).
### tex
For plots with fancy equations by tex, we turn on `plt.rcParams['text.usetex']`. If tex or latex is not installed in the environment, please turn this off by, for example
```python
    plt.rcParams['text.usetex'] = False
```

## Citation

```
@inproceedings{sawada2026zof2,
  title = {{ZO-F2}: Low-Variance {Fisher} Preconditioner via Bilinear Estimation for Zeroth-Order Optimization},
  author = {Hiroshi Sawada and Yuya Hikima and Kenta Niwa},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year = {2026}
}
```

## License

[SoftwareLicenseAgreement.pdf](./SoftwareLicenseAgreement.pdf)
