from collections.abc import Callable
import torch
from torch import Tensor
from . import eval_output, nn_vmap


class CNN_MNIST(torch.nn.Module):
    def __init__(self, inner_ch: int, linear_width: int, output_size):
        super().__init__()
        num_ch = 1
        wh = 28
        w1 = wh // 2
        w2 = w1 // 2
        kernel_size = 3
        km1 = kernel_size - 1
        self.last_wh = (w2 + 2*1 - km1 - 1) // 2
        self.cnn = torch.nn.ModuleList([
            nn_vmap.Conv2d(num_ch, inner_ch, kernel_size, padding=1),
            nn_vmap.ReLU(),
            nn_vmap.MaxPool2d(2),
            nn_vmap.Conv2d(inner_ch, 2*inner_ch, kernel_size, padding=1),
            nn_vmap.ReLU(),
            nn_vmap.MaxPool2d(2),
            nn_vmap.Conv2d(2*inner_ch, 2*inner_ch, kernel_size, stride=2),
            nn_vmap.ReLU()
        ])
        self.fully = torch.nn.ModuleList([
            nn_vmap.Linear((self.last_wh**2)*2*inner_ch, linear_width),
            nn_vmap.ReLU(),
            nn_vmap.Linear(linear_width, output_size),
        ])
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.accuracy: Callable[[Tensor, Tensor], float] = eval_output.accuracy
        self.calc_output_fisher: Callable[[Tensor], Tensor] = eval_output.fisher_multinomial
        self.calc_output_fisher_empirical: Callable[[Tensor], Tensor] = eval_output.fisher_multinomial_empirical

    def forward(self, x: Tensor) -> Tensor:
        for f in self.cnn:
            x = f(x)
        xsh = x.shape
        x = x.view(*xsh[:-3], -1)
        for f in self.fully:
            x = f(x)
        return x
