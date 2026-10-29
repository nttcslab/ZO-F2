import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from omegaconf import DictConfig
import shutil
import read_log


def read_dir(dir: str, cuda_ids: list[int], vectors_list, seeds_list: list[list[int]], lr: float,
             epochs: int) -> tuple[list[DictConfig], list[pd.DataFrame]]:
    cfg_list = []
    df_list = []
    alpha = 1e-5
    for ci, vectors, seeds in zip(cuda_ids, vectors_list, seeds_list):
        for s in seeds:
            logfile = f'../log/{dir}/{vectors}_lr{lr}_alpha{alpha}_tau0.5_{ci}_{s}'
            logfile += '.log'
            cfg, df = read_log.read_log(logfile, epochs)
            match vectors:
                case 'fromI':
                    method = 'ZO-I'
                case 'F1' | 'F2' | 'H1' | 'H2':
                    method = f'ZO-{vectors}'
            if cfg.zoo.pre.warm_up > 0:
                method += '(warm)'
            df = df.assign(method=method)
            df = df.assign(seed=s)
            cfg_list.append(cfg)
            df_list.append(df)
    return cfg_list, df_list


def main():
    epochs = 3000
    time_limit = 4e+5

    # read results
    cuda_ids = [0,1,2,3,4,5,6,7]
    vectors_list = ['fromI', 'H2', 'F2', 'F2', 'fromI', 'H2', 'F2', 'F2']
    seeds_list = [[1,],] * 4 + [[2,],] * 4
    cfg, df_list = read_dir('resnet_more', cuda_ids, vectors_list, seeds_list, 1e-3, epochs)
    df = pd.concat(df_list)
    df['time_mean'] = df.groupby(['method', 'epoch'])['time'].transform('mean')
    df['te_acc'] = df['te_acc'] * 100

    pdflatex_available = shutil.which("pdflatex") is not None
    plt.rcParams['text.usetex'] = pdflatex_available
    plt.rcParams['pdf.use14corefonts'] = pdflatex_available
    plt.rcParams['font.size'] = 15
    sns.set_style('whitegrid')
    sns.color_palette("colorblind")

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    custom_palette = {'ZO-I': '#228833', 'ZO-H1': '#CCBB44', 'ZO-H2': '#EE6677', 'ZO-F2': '#4477AA', 'ZO-F2(warm)': '#66CCEE'}
    g = sns.lineplot(ax=ax, data=df[df['time']<=time_limit], x='step', y='te_acc', hue='method', errorbar='sd', palette=custom_palette)
    g.set(xlabel='Steps', ylabel='Test accuracy')
    g.set_ylim(32, 80)
    g.legend(fontsize=13)
    fig.tight_layout()
    plt.savefig('resnet.png', bbox_inches="tight", dpi=500)
    plt.savefig('resnet.pdf', bbox_inches="tight", pad_inches=0.01)
    plt.show()

if __name__ == '__main__':
    main()
