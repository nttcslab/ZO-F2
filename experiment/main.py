import os
import hydra
import numpy as np
import torch
import task, train


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.cuda_id)
    print('CUDA_VISIBLE_DEVICES = ', os.environ['CUDA_VISIBLE_DEVICES'])
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # dataset and network
    train_loader, test_loader, net = task.get_dataset_model(cfg)
    if cfg.load_file is not None:
        net.load_state_dict(torch.load(cfg.load_file))
    net.eval()

    # report setting
    outfile = f'/{cfg.suffix}{cfg.zoo.vectors}_lr{cfg.lr}_q{cfg.zoo.num_vectors}_d{cfg.zoo.max_dim}_alpha{cfg.zoo.pre.alpha}_tau{cfg.zoo.pre.tau}'
    if cfg.dataset == 'MNIST':
        outfile += f'_f{cfg.num_features}'
    outfile += f'_{cfg.cuda_id}_{cfg.seed}.log'
    progr = train.ReportProgress(cfg, outfile)

    # training net by ZO optimization
    tr_zoo = train.TrainZO(cfg, net, progr, train_loader, test_loader)

    try:
        tr_zoo.train(cfg)
    except KeyboardInterrupt:
        print('KeyboardInterrupt')
    if cfg.save_file is not None:
        save_file = f'../log/{cfg.suffix}_{cfg.cuda_id}_{cfg.seed}_{cfg.save_file}'
        torch.save(net.state_dict(), save_file)


if __name__ == '__main__':
    main()
