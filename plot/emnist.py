import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from omegaconf import DictConfig
import shutil
import read_log


def read_dir(dir: str, cuda_ids: list[int], vectors_list, seeds: list[int], lr: float,
             alpha_list: list, epochs: int) -> tuple[list[DictConfig], list[pd.DataFrame]]:
    cfg_list = []
    df_list = []
    for ci, vectors in zip(cuda_ids, vectors_list):
        for alpha in alpha_list:
            for s in seeds:
                logfile = f'../log/{dir}/{vectors}_lr{lr}_alpha{alpha}_tau0.5_{ci}_{s}'
                logfile += '.log'
                cfg, df = read_log.read_log(logfile, epochs)
                match vectors:
                    case 'fromI':
                        method = 'ZO-I'
                    case 'F1' | 'F2' | 'H1' | 'H2':
                        method = f'ZO-{vectors}'
                df = df.assign(alpha=cfg.zoo.pre.alpha)
                df = df.assign(method=method)
                df = df.assign(seed=s)
                df = df.assign(tau=cfg.zoo.pre.tau)
                df = df.assign(nq=cfg.zoo.num_vectors)
                cfg_list.append(cfg)
                df_list.append(df)
    return cfg_list, df_list


def read_dir_tau(dir: str, cuda_ids: list[int], vectors_list, seeds: list[int], lr: float,
             tau_list: list, alpha_list: list, epochs: int) -> tuple[list[DictConfig], list[pd.DataFrame]]:
    cfg_list = []
    df_list = []
    for ci, vectors, tau in zip(cuda_ids, vectors_list, tau_list):
        for alpha in alpha_list:
            for s in seeds:
                logfile = f'../log/{dir}/{vectors}_lr{lr}_alpha{alpha}_tau{tau}_{ci}_{s}'
                logfile += '.log'
                cfg, df = read_log.read_log(logfile, epochs)
                df = df.assign(alpha=alpha)
                df = df.assign(seed=s)
                df = df.assign(tau=tau)
                df = df.assign(nq=cfg.zoo.num_vectors)
                cfg_list.append(cfg)
                df_list.append(df)
    return cfg_list, df_list


def main():
    vectors_list = ['F2'] * 7
    alpha_list = [1e-5, 2e-5, 3e-5,]

    pdflatex_available = shutil.which("pdflatex") is not None
    plt.rcParams['text.usetex'] = pdflatex_available
    plt.rcParams['pdf.use14corefonts'] = pdflatex_available
    plt.rcParams['font.size'] = 15
    sns.set_style('whitegrid')
    sns.color_palette("colorblind")

    dashes = {np.float64(1e-05): '', np.float64(2e-05): (2, 2), np.float64(3e-05): (4, 2)}
    markers = {np.float64(1e-05): 'o', np.float64(2e-05): 'o', np.float64(3e-05): 'o'}
    custom_palette = {np.float64(1e-05): '#c5b0d5', np.float64(2e-05): '#9467bd', np.float64(3e-05): '#7b4173'}
    labels = [r'$\alpha=10^{-5}$', r'$\alpha=2\!\cdot\! 10^{-5}$', r'$\alpha=3\!\cdot\! 10^{-5}$']

    # central difference
    seeds = [1,2,3]
    cuda_ids = [1, 2, 3, 4, 5, 6, 7]
    tau_list = [1e-2, 1e-1, 2e-1, 5e-1, 7e-1, 8e-1, 9e-1]
    cfg, df_list = read_dir_tau('emnist_tau_central', cuda_ids, vectors_list, seeds, 2e-4, tau_list, alpha_list, 300)
    df = pd.concat([d.iloc[[-1]] for d in df_list])
    df['te_acc'] = df['te_acc'] * 100

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3))
    g = sns.lineplot(ax=ax, data=df, x='tau', y='te_acc', hue='alpha', style='alpha', dashes=dashes, markers=markers, palette=custom_palette)
    g.set_xlabel(r'Clipping threshold $\tau$', fontsize=15)
    g.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    g.set_ylabel('Test accuracy', fontsize=15)
    g.set_ylim([82.2, 84.5])
    handles, _ = plt.gca().get_legend_handles_labels()
    g.legend(handles=handles, labels=labels, loc='lower center')
    fig.tight_layout()
    plt.savefig('emnist_central.png', bbox_inches="tight", dpi=500)
    plt.savefig('emnist_central.pdf', bbox_inches="tight", pad_inches=0.01)
    plt.show()

    # forward difference
    seeds = [1,2,3]
    cuda_ids = [0, 1, 2, 4, 5, 6, 7]
    tau_list = [1e-2, 1e-1, 2e-1, 5e-1, 7e-1, 8e-1, 9e-1]
    cfg, df_list = read_dir_tau('emnist_tau_forward', cuda_ids, vectors_list, seeds, 2e-4, tau_list, alpha_list, 300)
    df = pd.concat([d.iloc[[-1]] for d in df_list])
    df['te_acc'] = df['te_acc'] * 100

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3))
    g = sns.lineplot(ax=ax, data=df, x='tau', y='te_acc', hue='alpha', style='alpha', dashes=dashes, markers=markers, palette=custom_palette)
    g.set_xlabel(r'Clipping threshold $\tau$', fontsize=15)
    g.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    g.set_ylabel('Test accuracy', fontsize=15)
    g.set_ylim([82.2, 84.5])
    handles, _ = plt.gca().get_legend_handles_labels()
    g.legend(handles=handles, labels=labels, loc='lower right')
    fig.tight_layout()
    plt.savefig('emnist_forward.png', bbox_inches="tight", dpi=500)
    plt.savefig('emnist_forward.pdf', bbox_inches="tight", pad_inches=0.01)
    plt.show()

if __name__ == '__main__':
    main()
