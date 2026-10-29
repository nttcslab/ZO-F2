from collections.abc import Callable
import torch
from torch import Tensor
from . import eval_output, nn_vmap


class MNIST_random_projection_linear(torch.nn.Module):
    def __init__(self, mid_width):
        super().__init__()
        self.fixed_linear = torch.nn.Linear(28*28, mid_width, bias=False)
        for param in self.fixed_linear.parameters():
            param.requires_grad = False
        self.linear = nn_vmap.Linear(mid_width, 10, bias=False)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.accuracy: Callable[[Tensor, Tensor], float] = eval_output.accuracy
        self.calc_output_fisher: Callable[[Tensor], Tensor] = eval_output.fisher_multinomial

    def forward(self, x: Tensor) -> Tensor:
        xshape = x.shape
        x = x.view((xshape[0], xshape[1], -1))
        self.random_projection = torch.einsum('oi,vbi->vbo', self.fixed_linear.weight, x)
        return self.linear(self.random_projection)

    def get_fim(self, train_loader):
        count = 0
        cov_sum = None
        for input, label in train_loader:
            batch_size = input.shape[0]
            output = self.forward(input[None])
            Fy = self.calc_output_fisher(output[0])
            rand_proj = self.random_projection[0]
            cov = torch.einsum('bi,bj,bmn->minj', rand_proj, rand_proj, Fy) / batch_size
            sh = cov.shape
            cov = cov.reshape(sh[0]*sh[1], sh[2]*sh[3])
            if cov_sum is None:
                cov_sum = cov
            else:
                cov_sum += cov
            count += 1
        assert cov_sum is not None
        fim = cov_sum / count
        return fim
