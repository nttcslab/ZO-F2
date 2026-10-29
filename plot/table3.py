import numpy as np
from scipy.stats import mannwhitneyu
import pandas as pd
import mnist, copying, emnist, resnet
from omegaconf import DictConfig


def print_line(name: str, header: str, cfg_list: list[DictConfig], df_list: list[pd.DataFrame], methods: list[str]) -> None:
    cfg = cfg_list[-1]
    num_vectors = cfg.zoo.num_vectors
    lr = cfg.lr
    cfg_exec = cfg.exec
    diff = 'cen.' if cfg.zoo.diff == 'central' else 'for.'
    print(f'{header} & {name}', end='')
    print(f' & {cfg_exec.parameters:,} & {num_vectors} & {diff} & ', end='')
    data_frame = pd.concat([df.iloc[[-1]] for df in df_list])
    pvalues, data = get_pvalues(data_frame, methods)
    for pv, d in zip(pvalues, data):
        d *= 100  # percentage
        mean = d.mean()
        std = d.std(ddof=1)
        if pv is None:
            print(f'\\textbf{{{mean:2.1f}}}$\\pm {std:1.1f}$ & ', end='')
        elif pv <= 0.05:
            print(f'\\textit{{{mean:2.1f}}}$\\pm {std:1.1f}$ & ', end='')
        else:
            print(f'{mean:2.1f}$\\pm {std:1.1f}$ & ', end='')
    print('\b\b\\\\')
    print('pvalues =', pvalues)


def get_pvalues(data_frame: pd.DataFrame, methods: list[str]):
    data = np.array([
        data_frame[(data_frame.method == meth)]['te_acc']
        for meth in methods
    ])
    # identify best mean values
    best_idx = data.mean(axis=1).argmax()
    # p-values by Mann-Whitney U test
    pvalues = [
        # None: best mean value, otherwise p-value
        None if i == best_idx else mannwhitneyu(data[best_idx], data1, alternative='greater')[1]
        for i, data1 in enumerate(data)
    ]
    return pvalues, data


def main():
    # read results
    seeds = [1, 2, 3, 4, 5]
    vectors_list = ['fromI', 'H1', 'H2', 'F2']
    features_mnist = [[80,],] * 4
    # alpha_mnist = [1e-4, ]
    cfg_mnist, df_mnist = mnist.read_dir_ndim('mnist', [0, 1, 2, 3], vectors_list, seeds, 1e-3, features_mnist, 200)
    # cfg_mnist, df_mnist = mnist.read_dir_alpha('mnist', [4, 5, 6, 7], vectors_list, seeds, 1e-3, alpha_mnist, 200)
    alpha_list = [3e-6, ] * 4
    cfg_emnist, df_emnist = emnist.read_dir('emnist', [4, 5, 6, 7], vectors_list, seeds, 2e-4, alpha_list, 300)
    alpha_copying = [1e-5,]
    cfg_copying, df_copying = copying.read_dir('copying', [0, 1, 2, 3], vectors_list, seeds, 1e-4, alpha_copying, 100)
    vectors_resnet = ['fromI', 'H2', 'F2', 'F2']
    seeds_list_resnet = [[1,2,3,4,5],] * 4
    cuda_ids_resnet = [4,5,6,7]
    cfg_resnet, df_resnet = resnet.read_dir('resnet', cuda_ids_resnet, vectors_resnet, seeds_list_resnet, 1e-3, 300)

    # print results
    methods = ['ZO-I', 'ZO-H1', 'ZO-H2', 'ZO-F2']
    print_line('MNIST', 'Linear', cfg_mnist, df_mnist, methods)
    print_line('EMNIST', 'CNN', cfg_emnist, df_emnist, methods)
    print_line('Copying', 'RNN', cfg_copying, df_copying, methods)
    methods = ['ZO-I', 'ZO-H2', 'ZO-F2']
    print_line('CIFAR10', 'ResNet', cfg_resnet, df_resnet, methods)

if __name__ == '__main__':
    main()
