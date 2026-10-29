import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from omegaconf import DictConfig
import shutil
import read_log


def read_dir_q(dir: str, cuda_ids: list[int], vectors_list, alpha_list, seeds: list[int], lr: float,
             q_list: list, epochs: int) -> tuple[list[DictConfig], list[pd.DataFrame]]:
    cfg_list = []
    df_list = []
    tau = 0.5
    for ci, vectors in zip(cuda_ids, vectors_list):
        for alpha in alpha_list:
            for q in q_list:
                for s in seeds:
                    logfile = f'../log/{dir}/{vectors}_lr{lr}_q{q}_d800_alpha{alpha}_tau{tau}_f80_{ci}_{s}'
                    logfile += '.log'
                    cfg, df = read_log.read_log(logfile, epochs)
                    method = f'ZO-{vectors}'
                    df = df.assign(method=method)
                    df = df.assign(seed=s)
                    df = df.assign(q=q)
                    df = df.assign(alpha=alpha)
                    cfg_list.append(cfg)
                    df_list.append(df)
    return cfg_list, df_list


def main():
    epochs = 200
    custom_palette = {'ZO-I': '#228833', 'ZO-H1': '#CCBB44', 'ZO-H2': '#EE6677', 'ZO-F2': '#4477AA', 'ZO-F2(warm)': '#66CCEE'}
    vectors_list = ['H2', 'F2']
    lr = 1e-3
    seeds = [1, 2, 3, 4, 5]
    cudaid_list = [6, 7]
    q_list = [2,4,6,8,10]
    alpha_list = [1e-4, 2e-4]
    _, df_list = read_dir_q('mnist_q', cudaid_list, vectors_list, alpha_list, seeds, lr, q_list, epochs)
    df = pd.concat([d.iloc[[-1]] for d in df_list])

    pdflatex_available = shutil.which("pdflatex") is not None
    plt.rcParams['text.usetex'] = pdflatex_available
    plt.rcParams['pdf.use14corefonts'] = pdflatex_available
    plt.rcParams['font.size'] = 15
    sns.set_style('whitegrid')

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7.2, 3))
    g = sns.lineplot(ax=ax[0], data=df[df['alpha']==1e-4], palette=custom_palette, x='q', y='error', hue='method', marker='o', errorbar='sd')
    g.set_yscale('log')
    g.set(xlabel=r'Number of perturbations $q$', ylabel='Estimation error\n(log scale)')
    g.set_xticks(q_list)
    g.set_ylim([2e-1, 2e+2])
    g.legend(loc='upper right')
    g.set_title(r'$\alpha=10^{-4}$')
    g = sns.lineplot(ax=ax[1], data=df[df['alpha']==2e-4], palette=custom_palette, x='q', y='error', hue='method', marker='o', errorbar='sd')
    g.set_yscale('log')
    g.set(xlabel=r'Number of perturbations $q$', ylabel='')
    g.set_xticks(q_list)
    g.set_ylim([2e-1, 2e+2])
    g.legend(loc='upper right')
    g.set_title(r'$\alpha=2\cdot 10^{-4}$')
    fig.tight_layout()
    plt.savefig('mnist_q.png', bbox_inches="tight", dpi=500)
    plt.savefig('mnist_q.pdf', bbox_inches="tight", pad_inches=0.03)
    plt.show()


if __name__ == '__main__':
    main()
