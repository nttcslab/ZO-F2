import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
                if cfg.zoo.pre.update_type == 'solve_clip':
                    method += '_d'
                df = df.assign(alpha=cfg.zoo.pre.alpha)
                df = df.assign(method=method)
                df = df.assign(seed=s)
                cfg_list.append(cfg)
                df_list.append(df)
    return cfg_list, df_list


def main():
    epoch = 50

    vectors_list = ['F2', 'F2']
    alpha_list = [3e-5]
    seeds = [1,2,4]
    _, df_list = read_dir('copying_cond', [0, 1], vectors_list, seeds, 1e-4, alpha_list, epoch)

    df = pd.concat(df_list)

    pdflatex_available = shutil.which("pdflatex") is not None
    plt.rcParams['text.usetex'] = pdflatex_available
    plt.rcParams['pdf.use14corefonts'] = pdflatex_available
    plt.rcParams['font.size'] = 15
    sns.set_style('whitegrid')

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7.7, 3))
    ax[0].xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax[0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax[1].xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax[1].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    dashes = {'ZO-F2': '', 'ZO-F2_d': (2, 0.8)}
    custom_palette = {'ZO-F2': '#bcaaa4', 'ZO-F2_d': '#5d4037'}

    g = sns.lineplot(ax=ax[0], data=df, x='step', y='m_cond', hue='method', style='method', dashes=dashes, palette=custom_palette, units='seed', estimator=None)
    handles, labels = ax[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax[0].legend(by_label.values(), ['whitening', 'direct solving'], fontsize=15)
    g.set_yscale('log')
    g.set(xlabel='Steps')
    g.set(ylabel='Condition number')
    g.set_xlim([0, 3e+4])
    g.set_ylim([0.9, 7e+6])
    g = sns.lineplot(ax=ax[1], data=df, x='step', y='tr_loss', hue='method', style='method', dashes=dashes, palette=custom_palette, units='seed', estimator=None)
    handles, labels = ax[1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax[1].legend(by_label.values(), ['whitening', 'direct solving'], fontsize=14)
    g.set(xlabel='Steps')
    g.set_yscale('log')
    g.set(ylabel='Training loss')
    g.set_xlim([0, 3e+4])
    g.set_ylim([0.06, 1.2])
    fig.tight_layout()
    plt.savefig('copying.png', bbox_inches="tight", dpi=500)
    plt.savefig('copying.pdf', bbox_inches="tight", pad_inches=0.02)
    plt.show()


if __name__ == '__main__':
    main()
