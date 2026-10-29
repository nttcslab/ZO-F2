import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import NullLocator
import seaborn as sns
from omegaconf import DictConfig
import shutil
import read_log


def read_dir_lr(dir: str, cuda_ids: list[int], vectors_list, seeds: list[int],
             lr_list: list, epochs: int) -> tuple[list[DictConfig], list[pd.DataFrame]]:
    cfg_list = []
    df_list = []
    alpha = 3e-6
    for ci, vectors in zip(cuda_ids, vectors_list):
        for lr in lr_list:
            for s in seeds:
                logfile = f'../log/{dir}/{vectors}_lr{lr}_alpha{alpha}_tau0.5_{ci}_{s}'
                logfile += '.log'
                cfg, df = read_log.read_log(logfile, epochs)
                match vectors:
                    case 'fromI':
                        method = 'ZO-I'
                    case 'F1' | 'F2' | 'H1' | 'H2':
                        method = f'ZO-{vectors}'
                df = df.assign(method=method)
                df = df.assign(seed=s)
                df = df.assign(lr=cfg.lr)
                cfg_list.append(cfg)
                df_list.append(df)
    return cfg_list, df_list


def read_dir_nq(dir: str, cuda_ids: list[int], vectors_list, seeds: list[int],
             nq_list: list, epochs: int) -> tuple[list[DictConfig], list[pd.DataFrame]]:
    cfg_list = []
    df_list = []
    alpha = 3e-6
    lr = 2e-4
    for ci, nq in zip(cuda_ids, nq_list):
        for vectors in vectors_list:
            for s in seeds:
                logfile = f'../log/{dir}/{vectors}_lr{lr}_alpha{alpha}_tau0.5_{ci}_{s}'
                logfile += '.log'
                cfg, df = read_log.read_log(logfile, epochs)
                match vectors:
                    case 'fromI':
                        method = 'ZO-I'
                    case 'F1' | 'F2' | 'H1' | 'H2':
                        method = f'ZO-{vectors}'
                df = df.assign(method=method)
                df = df.assign(seed=s)
                df = df.assign(nq=cfg.zoo.num_vectors)
                cfg_list.append(cfg)
                df_list.append(df)
    return cfg_list, df_list


def main():
    # read results
    vectors_list = ['fromI', 'H1', 'H2', 'F2']
    seeds = [1,2,3]
    lr_list = [1e-4,2e-4,5e-4,1e-3]
    cfg, df_lr_list = read_dir_lr('emnist_lr_q', [0,1,2,3], vectors_list, seeds, lr_list, 300)
    nq_list = [10, 20, 40, 80]
    cfg, df_nq_list = read_dir_nq('emnist_lr_q', [4,5,6,7], vectors_list, seeds, nq_list, 300)

    pdflatex_available = shutil.which("pdflatex") is not None
    plt.rcParams['text.usetex'] = pdflatex_available
    plt.rcParams['pdf.use14corefonts'] = pdflatex_available
    plt.rcParams['font.size'] = 15
    sns.set_style('whitegrid')
    sns.color_palette("colorblind")

    df_lr = pd.concat([d.iloc[[-1]] for d in df_lr_list])
    df_lr['te_acc'] = df_lr['te_acc'] * 100
    df_nq = pd.concat([d.iloc[[-1]] for d in df_nq_list])
    df_nq['te_acc'] = df_nq['te_acc'] * 100

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7.6, 3))
    ax[0].xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax[0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    custom_palette = {'ZO-I': '#228833', 'ZO-H1': '#CCBB44', 'ZO-H2': '#EE6677', 'ZO-F2': '#4477AA', 'ZO-F2(warm)': '#66CCEE'}
    dashes = {'ZO-I': (1, 1), 'ZO-H1': (2, 2), 'ZO-H2': '', 'ZO-F2': (4, 2), 'ZO-F2(warm)': (4, 2, 1, 2)}
    g = sns.lineplot(ax=ax[0], data=df_lr, x='lr', y='te_acc', hue='method', marker='o', errorbar='sd', palette=custom_palette, style='method', dashes=dashes)
    g.set_xlabel(r'Learning rate $\eta$', fontsize=15)
    g.set_xscale('log')
    g.xaxis.set_major_locator(plt.FixedLocator(lr_list))
    g.xaxis.set_major_formatter(plt.FixedFormatter([r'$10^{-4}$', r'$2\cdot 10^{-4}$', r'$5\cdot 10^{-4}$', r'$10^{-3}$']))
    g.xaxis.set_minor_locator(NullLocator())
    g.set_ylabel('Test accuracy', fontsize=15)
    g.set_ylim(66, 84.5)
    g.legend([],[], frameon=False)
    g = sns.lineplot(ax=ax[1], data=df_nq, x='nq', y='te_acc', hue='method', marker='o', errorbar='sd', palette=custom_palette, style='method', dashes=dashes)
    g.set_xlabel(r'Number of perturbations $q$', fontsize=15)
    g.set_ylabel('', labelpad=0)
    g.set_ylim(66, 84.5)
    g.legend(bbox_to_anchor=(0.7, 0.7), loc='upper left')
    fig.tight_layout(w_pad=0.3)
    plt.savefig('emnist_lr_q.png', bbox_inches="tight", dpi=500)
    plt.savefig('emnist_lr_q.pdf', bbox_inches="tight", pad_inches=0.01)
    plt.show()

if __name__ == '__main__':
    main()
