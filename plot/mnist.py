import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import seaborn as sns
from omegaconf import DictConfig
import shutil
import read_log


def read_dir_ndim(dir: str, cuda_ids: list[int], vectors_list, seeds: list[int], lr: float,
             features_list: list, epochs: int) -> tuple[list[DictConfig], list[pd.DataFrame]]:
    cfg_list = []
    df_list = []
    alpha = 1e-4
    tau = 0.5
    for ci, vectors, f_list in zip(cuda_ids, vectors_list, features_list):
        for num_features in f_list:
            for s in seeds:
                logfile = f'../log/{dir}/{vectors}_lr{lr}_alpha{alpha}_tau{tau}_f{num_features}_{ci}_{s}'
                logfile += '.log'
                cfg, df = read_log.read_log(logfile, epochs)
                match vectors:
                    case 'fromI':
                        method = 'ZO-I'
                    case 'F1' | 'F2' | 'H1' | 'H2':
                        method = f'ZO-{vectors}'
                df = df.assign(method=method)
                df = df.assign(seed=s)
                df = df.assign(ndim=10*cfg.num_features)
                cfg_list.append(cfg)
                df_list.append(df)
    return cfg_list, df_list


def read_dir_alpha(dir: str, cuda_ids: list[int], vectors_list, seeds: list[int], lr: float,
             alpha_list: list, epochs: int) -> tuple[list[DictConfig], list[pd.DataFrame]]:
    cfg_list = []
    df_list = []
    num_features = 80
    tau = 0.5
    for ci, vectors in zip(cuda_ids, vectors_list):
        for alpha in alpha_list:
            for s in seeds:
                logfile = f'../log/{dir}/{vectors}_lr{lr}_alpha{alpha}_tau{tau}_f{num_features}_{ci}_{s}'
                logfile += '.log'
                cfg, df = read_log.read_log(logfile, epochs)
                match vectors:
                    case 'fromI':
                        method = 'ZO-I'
                    case 'F1' | 'F2' | 'H1' | 'H2':
                        method = f'ZO-{vectors}'
                df = df.assign(method=method)
                df = df.assign(seed=s)
                df = df.assign(alpha=cfg.zoo.pre.alpha)
                cfg_list.append(cfg)
                df_list.append(df)
    return cfg_list, df_list


def compute_diff_from_baseline(df):
    metrics = ['te_acc', 'tr_loss']    
    df_base = df[df['method'] == 'ZO-I'][['ndim', 'seed'] + metrics].rename(
        columns={m: f'{m}_base' for m in metrics}
    )
    df_merged = df.merge(df_base, on=['ndim', 'seed'])
    for m in metrics:
        df_merged[f'{m}_diff'] = (df_merged[m] - df_merged[f'{m}_base'])
    return df_merged


def main():
    epochs = 200
    custom_palette = {'ZO-I': '#228833', 'ZO-H1': '#CCBB44', 'ZO-H2': '#EE6677', 'ZO-F2': '#4477AA', 'ZO-F2(warm)': '#66CCEE'}
    vectors_list = ['fromI', 'H1', 'H2', 'F2']
    lr = 1e-3
    seeds = [1, 2, 3, 4, 5]

    cudaid_list = [0, 1, 2, 3]
    features_list = [[32,48,64,80]] * 4
    _, df_list = read_dir_ndim('mnist', cudaid_list, vectors_list, seeds, lr, features_list, epochs)
    dflast = pd.concat([d.iloc[[-1]] for d in df_list])
    dflast = dflast.reset_index(drop=True)
    dflast = compute_diff_from_baseline(dflast)
    dflast_ndim = dflast[~dflast.method.str.contains('ZO-I')]

    cudaid_list = [4, 5, 6, 7]
    alpha_list = [1e-4, 2e-4, 5e-4, 1e-3]
    _, df_list = read_dir_alpha('mnist', cudaid_list, vectors_list, seeds, lr, alpha_list, epochs)
    dflast = pd.concat([d.iloc[[-1]] for d in df_list])
    dflast = dflast.reset_index(drop=True)
    dflast_alpha = dflast[dflast.method.str.contains('2')]

    pdflatex_available = shutil.which("pdflatex") is not None
    plt.rcParams['text.usetex'] = pdflatex_available
    plt.rcParams['pdf.use14corefonts'] = pdflatex_available
    plt.rcParams['font.size'] = 15
    sns.set_style('whitegrid')
    # sns.color_palette("colorblind")

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7.6, 3))
    g = sns.lineplot(ax=ax[0], data=dflast_ndim, palette=custom_palette, x='ndim', y='error', hue='method', marker='o', errorbar='sd')
    g.set_yscale('log')
    g.set(xlabel=r'Number of parameters $d$', ylabel='Estimation error\n(log scale)')
    g.set_xticks([320, 480, 640, 800])
    # g.set_ylim([5e-2, 3e+7])
    g.legend(loc='upper left')
    g.set_title(r'$\alpha=10^{-4}$')
    # 
    g = sns.lineplot(ax=ax[1], data=dflast_alpha, palette=custom_palette, x='alpha', y='error', hue='method', marker='o', errorbar='sd')
    g.set_xscale('log')
    g.xaxis.set_major_locator(plt.FixedLocator(alpha_list))
    g.xaxis.set_major_formatter(plt.FixedFormatter([r'$10^{-4}$', r'$2\cdot 10^{-4}$', r'$5\cdot 10^{-4}$', '$10^{-3}$']))
    g.xaxis.set_minor_locator(NullLocator())
    g.set(xlabel=r'EMA coefficient $\alpha$', ylabel='Estimation error\n(linear scale)')
    g.set_title(r'$d=800$')
    g.legend(loc='upper left')
    fig.tight_layout()
    plt.savefig('mnist.png', bbox_inches="tight", dpi=500)
    plt.savefig('mnist.pdf', bbox_inches="tight", pad_inches=0.03)
    plt.show()

if __name__ == '__main__':
    main()
