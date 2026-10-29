# from sklearn import metrics
import torch
from torch import Tensor


def accuracy(output: Tensor, label: Tensor) -> float:
    return torch.eq(torch.argmax(output.detach(), dim=1), label).float().mean().item()


# def r2_score(output: Tensor, target: Tensor) -> float:
#     r2score = metrics.r2_score(target.detach().cpu().numpy(), output.detach().cpu().numpy())
#     return float(r2score)   


def fisher_multinomial(logits: Tensor) -> Tensor:
    probability = torch.softmax(logits.detach(), dim=1)
    return torch.diag_embed(probability) - torch.einsum('bm,bn->bmn', probability, probability)

def fisher_multinomial_empirical(logits: Tensor, target: Tensor) -> Tensor:
    probability = torch.softmax(logits.detach(), dim=1)
    onehot = torch.nn.functional.one_hot(target, num_classes=logits.size(-1)).float()
    grad = onehot - probability
    return torch.einsum('bm,bn->bmn', grad, grad)

def fisher_multinomial_RNN(logits: Tensor) -> Tensor:
    probability = torch.softmax(logits.detach(), dim=1)
    num_classes = probability.shape[1]
    log_derivative = (-probability[None]).repeat(num_classes, 1, 1, 1)
    for m in range(num_classes):
        log_derivative[m, :, m] += 1
    return torch.einsum('bos,obms,obns->bmns', probability, log_derivative, log_derivative)


def fisher_normal(output: Tensor) -> Tensor:
    num_sample, dim_output = output.shape
    eye = torch.eye(dim_output, device=output.device, dtype=output.dtype)
    return eye[None].repeat(num_sample, 1, 1)
