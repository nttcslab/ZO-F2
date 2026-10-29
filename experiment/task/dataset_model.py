from omegaconf import DictConfig
import torch
from torch import Tensor
from torch.nn import Module
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, dataset


class TensorsLoader:
    ''' tensors: a list of tensors '''
    def __init__(self, tensors: list[Tensor], batch_size: int=1, shuffle: bool=False) -> None:
        self.tensors = tensors
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_size = tensors[0].shape[0]

    def __iter__(self):
        self._i = 0
        if self.shuffle:
            index_shuffle = torch.randperm(self.data_size)
            self.tensors = [tensor[index_shuffle] for tensor in self.tensors]
        return self

    def __next__(self) -> list[Tensor]:
        i1 = self.batch_size * self._i
        i2 = min(self.batch_size * (self._i + 1), self.data_size)
        if i1 >= self.data_size:
            raise StopIteration()
        self._i += 1
        return [tensor[i1:i2] for tensor in self.tensors]


def get_loader(data: dataset.Dataset, device: str, batch_size: int, shuffle: bool) -> TensorsLoader:
    loader = DataLoader(data, batch_size=batch_size)
    input_list = []
    target_list = []
    for inp, tar in loader:
        input_list.append(inp)
        target_list.append(tar)
    input = torch.cat(input_list, dim=0)
    input = input.to(device)
    target = torch.cat(target_list, dim=0)
    target = target.to(device)
    return TensorsLoader([input, target], batch_size=batch_size, shuffle=shuffle)


def get_dataset_model(cfg: DictConfig) -> tuple[TensorsLoader, TensorsLoader, Module]:   
    batch_size = cfg.batch_size
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ))
    ])
    transform_cifar = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    match cfg.dataset:
        case 'MNIST' | 'FMNIST':
            from . import mnist
            if cfg.dataset == 'MNIST':
                train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
                test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)
            elif cfg.dataset == 'FMNIST':
                train_data = datasets.FashionMNIST(root='data', train=True, download=True, transform=transform)
                test_data = datasets.FashionMNIST(root='data', train=False, download=True, transform=transform)
            train_loader = get_loader(train_data, cfg.device, batch_size, shuffle=True)
            test_loader = get_loader(test_data, cfg.device, batch_size, shuffle=False)
            model: Module = mnist.MNIST_random_projection_linear(cfg.num_features)
        case 'EMNIST':
            from . import cnn
            split = 'balanced'
            num_class = 47
            train_data = datasets.EMNIST(root='data', split=split, train=True, download=True, transform=transform)
            test_data = datasets.EMNIST(root='data', split=split, train=False, download=True, transform=transform)
            train_loader = get_loader(train_data, cfg.device, batch_size, shuffle=True)
            test_loader = get_loader(test_data, cfg.device, batch_size, shuffle=False)
            model: Module = cnn.CNN_MNIST(cfg.cnn.ch, cfg.cnn.linear_width, num_class)
        case 'copying':
            from . import copying
            nSamples_train, nSamples_test = 60000, 10000
            len_symbol, len_blank = 10, 100
            input, target = copying.generate_data(len_symbol, len_blank, nSamples_train, cfg.device)
            train_loader = TensorsLoader([input, target], batch_size=batch_size, shuffle=True)
            input, target = copying.generate_data(len_symbol, len_blank, nSamples_test, cfg.device)
            test_loader = TensorsLoader([input, target], batch_size=batch_size, shuffle=True)
            model = copying.CopyingRNN(cfg)
        case 'CIFAR10':
            from . import resnet
            train_data = datasets.CIFAR10(root='data', train=True, download=True, transform=transform_cifar)
            train_loader = get_loader(train_data, cfg.device, batch_size, shuffle=True)
            test_data = datasets.CIFAR10(root='data', train=False, download=True, transform=transform_cifar)
            test_loader = get_loader(test_data, cfg.device, batch_size, shuffle=False)
            model = resnet.resnet20(num_classes=10)

    model = model.to(cfg.device)
    return train_loader, test_loader, model
