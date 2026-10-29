import pandas as pd
from omegaconf import DictConfig
import read_log


def read_dir(dir: str, cuda_ids: list[int], dmax_list, seeds: list[int], lr: float,
             alpha_list: list, epochs: int) -> tuple[list[DictConfig], list[pd.DataFrame]]:
    cfg_list = []
    df_list = []
    for ci, dmax in zip(cuda_ids, dmax_list):
        for alpha in alpha_list:
            for s in seeds:
                logfile = f'../log/{dir}/F2_lr{lr}_q20_d{dmax}_alpha{alpha}_tau0.5_{ci}_{s}'
                logfile += '.log'
                cfg, df = read_log.read_log(logfile, epochs)
                method = 'ZO-F2'
                if cfg.zoo.pre.update_type == 'solve_clip':
                    method += '_d'
                df = df.assign(dmax=dmax)
                df = df.assign(alpha=cfg.zoo.pre.alpha)
                df = df.assign(method=method)
                df = df.assign(seed=s)
                cfg_list.append(cfg)
                df_list.append(df)
    return cfg_list, df_list


def main():
    epoch = 100
    cuda_ids =  [0, 1, 2,  3,  4,   5,   4,   6,    5,    6,    7,    7]
    dmax_list = [1, 4, 13, 39, 117, 352, 528, 1056, 1584, 2112, 2640, 3168]
    alpha_list = [1e-5]
    seeds = [1,2,3,4,5]
    _, df_list = read_dir('copying_dmax', cuda_ids, dmax_list, seeds, 1e-4, alpha_list, epoch)
    df = pd.concat([df.iloc[[-1]] for df in df_list])

    mean = df.groupby('dmax')['te_acc'].mean() * 100
    std = df.groupby('dmax')['te_acc'].std() * 100
    time = df.groupby('dmax')['time'].mean() / epoch
    memory = df.groupby('dmax')['memory'].mean()
    print('$d_\mathrm{max}$ ', end='')
    for dmax in dmax_list:
        print(f'& {dmax} ', end='')
    print('\\\\')
    print('Test accuracy ', end='')
    for me, st in zip(mean, std):
        print(f'& {me:2.1f} ', end='')
    print('\\\\')
    print('', end='')
    for me, st in zip(mean, std):
        print(f'& $\\pm {st:1.1f}$ ', end='')
    print('\\\\')
    print('Seconds/epoch ', end='')
    for te in time:
        print(f'& {te:3.1f} ', end='')
    print('\\\\')
    print('Memory footprint ', end='')
    for mem in memory:
        print(f'& {mem:1.2f} ', end='')
    print('\\\\')


if __name__ == '__main__':
    main()
