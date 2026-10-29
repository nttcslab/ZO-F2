import pandas as pd
import numpy as np
import io
import ast
from omegaconf import DictConfig, OmegaConf
import re


def read_log(logfile: str, epochs: int) -> tuple[DictConfig, pd.DataFrame]:
    with open(logfile, 'r') as f:
        lines = f.readlines()
    cfg_list = []
    index_list = []
    for i, line in enumerate(lines):
        if line[0] == '{':
            cfg_dict = ast.literal_eval(line)
            cfg: DictConfig = OmegaConf.create(cfg_dict)
            cfg_list.append(cfg)
        if line[0] == '#':
            index_list.append(i)
            # params_match = re.search(r'#parameters\s*=\s*(\d+)', line)
            # blocks_match = re.search(r'#blocks\s*=\s*(\d+)', line)
            # cfg.num_parameters = int(params_match.group(1)) if params_match else None
            # cfg.num_blocks = int(blocks_match.group(1)) if blocks_match else None
    region_start = index_list[-1] + 1
    region_end = region_start + epochs + 1
    cfg = cfg_list[-1]
    df = pd.read_table(io.StringIO(''.join(lines[region_start:region_end])))
    exec_cfg_pairs = re.findall(r'#?(\w+)\s*=\s*(\d+)', lines[region_start-1])
    cfg.update({'exec': {k: int(v) for k, v in exec_cfg_pairs}})
    return cfg, df


def read_dir(dir: str, dataset: str, max_dim: int, cuda_ids: list[int], vectors_list, seeds: list[int], lr: float,
             alpha_list: list, epochs: int, budget:int, npW_detail=False) -> tuple[list[DictConfig], list[pd.DataFrame]]:
    cfg_list = []
    df_list = []
    for ci, vectors, alpha in zip(cuda_ids, vectors_list, alpha_list):
        for s in seeds:
            logfile = f'../log/{dir}/{vectors}_{max_dim}_lr{lr}_alpha{alpha}_{ci}_{s}'
            logfile += '.log'
            cfg, df = read_log(logfile, epochs)
            match vectors:
                case 'fromI':
                    method = 'ZO-I'
                case 'np':
                    method = 'NP'
                case 'F1' | 'F2' | 'H1' | 'H2':
                    method = f'ZO-{vectors}'
            if cfg.zoo.diff == 'central':
                method += 'c'
            else:  # 'forward
                method += 'f'
            if cfg.zoo.np.warm_up > 0:
                method += f'w{cfg.zoo.np.warm_up}'
            df = df.assign(alpha=cfg.zoo.np.alpha)
            df = df.assign(method=method)
            df = df.assign(seed=s)
            df = df.assign(tau=cfg.zoo.np.tau)
            cfg_list.append(cfg)
            df_list.append(df)
    return cfg_list, df_list
