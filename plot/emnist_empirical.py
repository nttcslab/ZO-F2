import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from omegaconf import DictConfig
import shutil
import read_log


def read_dir_empirical(dir: str, cuda_ids: list[int], seeds: list[int], lr: float,
             empirical_list: list, alpha_list: list, epochs: int) -> tuple[list[DictConfig], list[pd.DataFrame]]:
    cfg_list = []
    df_list = []
    for ci, empirical, alpha in zip(cuda_ids, empirical_list, alpha_list):
        for s in seeds:
            logfile = f'../log/{dir}/F2_lr{lr}_q40_d497_alpha{alpha}_tau0.5_{ci}_{s}'
            logfile += '.log'
            cfg, df = read_log.read_log(logfile, epochs)
            df = df.assign(seed=s)
            if empirical is True:
                method = 'Empirical Fisher'
            else:
                method = '(true) FIM'
            df = df.assign(method=method)
            df = df.assign(alpha=alpha)
            cfg_list.append(cfg)
            df_list.append(df)
    return cfg_list, df_list


def main():
    cuda_ids = [2, 3, 6, 7]
    empirical_list = [False, True, False, True]
    alpha_list = [2e-5, 2e-5, 3e-6, 3e-6]

    # central difference
    seeds = [1,2,3]
    cfg, df_list = read_dir_empirical('emnist_empirical', cuda_ids, seeds, 2e-4, empirical_list, alpha_list, 300)
    df = pd.concat(df_list)
    df['te_acc'] = df['te_acc'] * 100

    pdflatex_available = shutil.which("pdflatex") is not None
    plt.rcParams['text.usetex'] = pdflatex_available
    plt.rcParams['pdf.use14corefonts'] = pdflatex_available
    plt.rcParams['font.size'] = 15
    sns.set_style('whitegrid')
    sns.color_palette("colorblind")

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))
    ax[0].xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax[0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax[1].xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax[1].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    custom_palette = {'(true) FIM': '#4477AA', 'Empirical Fisher': '#ff7f0e'}
    g = sns.lineplot(ax=ax[0], data=df[df['alpha']==3e-6], x='step', y='te_acc', hue='method', errorbar='sd', palette=custom_palette)
    g.set_title(r'$\alpha=3\cdot 10^{-6}$')
    g.set(xlabel='Steps', ylabel='Test accuracy')
    g.set_xlim([0, 3.384e+5])
    g.set_ylim(70, 85)
    g.legend([],[], frameon=False)
    g = sns.lineplot(ax=ax[1], data=df[df['alpha']==2e-5], x='step', y='te_acc', hue='method', errorbar='sd')
    g.set_title(r'$\alpha=2\cdot 10^{-5}$')
    g.set(xlabel='Steps')
    g.set_ylabel('', labelpad=0)
    g.set_ylim(70, 85)
    g.set_xlim([0, 3.384e+5])
    g.legend(bbox_to_anchor=(0.28, 0.5), loc='upper left')
    fig.tight_layout()
    plt.savefig('emnist_empirical.png', bbox_inches="tight", dpi=500)
    plt.savefig('emnist_empirical.pdf', bbox_inches="tight", pad_inches=0.01)
    plt.show()


if __name__ == '__main__':
    main()
