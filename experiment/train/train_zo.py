''' Zeroth-order Optimization '''
import math
import torch
from torch import Tensor
from omegaconf import DictConfig
from . import util, precon
from task import TensorsLoader


def split_number_last_only_minmax(N, T):
    k = -(-N // T)   # ceil(N / T)
    m = -(-N // k)   # ceil(N / k)
    last = N - m * (k - 1)
    return [m] * (k - 1) + [last]


class TrainZO():
    def __init__(self, cfg: DictConfig, net: torch.nn.Module, progr: util.ReportProgress,
                 train_loader: TensorsLoader, test_loader: TensorsLoader):
        self.num_vectors = cfg.zoo.num_vectors
        if cfg.optimizer == 'SGD':
            optimizer = torch.optim.SGD(net.parameters(), lr=cfg.lr)
        elif cfg.optimizer == 'Adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr)
        else:
            raise ValueError('Unknown cfg.optimizer')
        if cfg.zoo.vectors == 'F1' or cfg.zoo.vectors == 'F2' or cfg.zoo.vectors == 'H1' or cfg.zoo.vectors == 'H2':
            self.zoo = ZOoptimEstimPCP(cfg, net, optimizer)
        else:  # 'fromI'
            self.zoo = ZOoptim(cfg, net, optimizer)
        progr.report_msg(f'# ZO optimization, #parameters = {self.zoo.nDim}, '
                         f'#blocks = {len(self.zoo.num_params)}, resulting_max = {max(self.zoo.num_params)}, '
                         f'resulting_min = {min(self.zoo.num_params)}')
        msg_append = '\tloss\tgrad_n\terror\tnum_u_p\tnum_sam\tm_cond\ttr_loss\tte_acc\tn_pertu\tstep\tfeval\tmemory'
        fmt_append = ('{:.5f}', '{:.1e}', '{:.5f}', '{:d}', '{:d}', '{:.1e}', '{:.5f}', '{:.5f}', '{:d}', '{:d}', '{:d}', '{:.2f}')
        self.progr = progr.set_header(msg_append, fmt_append)
        self.train_loader = train_loader
        self.test_loader = test_loader

    def train(self, cfg) -> None:
        total_iterations = 0
        for epoch in range(cfg.epochs):
            torch.cuda.reset_peak_memory_stats()
            self.progr.start_epoch(epoch, ['loss', 'grad_norm'])
            for i, (input, label) in enumerate(self.train_loader):  # line 4, Algorithm 1
                total_iterations += 1
                loss = self.zoo.initialize_batch_and_forward(input, label)
                if total_iterations < cfg.zoo.pre.warm_up:
                    grad = self.zoo.perform_I(self.num_vectors)
                elif total_iterations == cfg.zoo.pre.warm_up and cfg.zoo.vectors != 'fromI':
                    self.zoo.initFIMonehot()
                else:
                    match cfg.zoo.vectors:
                        case 'fromI':
                            grad = self.zoo.perform_I(self.num_vectors)
                        case 'F1' | 'F2':
                            grad = self.zoo.perform_F(self.num_vectors, cfg.zoo.vectors)  # line 5-9, Algorithm 1
                        case 'H1':
                            grad = self.zoo.perform_H1(self.num_vectors)
                        case 'H2':
                            grad = self.zoo.perform_H2(self.num_vectors, cfg.zoo.diff)
                        case _:
                            raise ValueError('Unexpected configuration: zoo.vectors')
                self.progr.append_results([loss, grad.norm()])
                if cfg.zoo.vectors == 'H2' and cfg.zoo.diff == 'central':
                    self.zoo.cost_func_eval -= 1
            if cfg.zoo.vectors == 'F1' or cfg.zoo.vectors == 'F2' or cfg.zoo.vectors == 'H1'  or cfg.zoo.vectors == 'H2': 
                num_update, samples_estim, max_cond_num = self.zoo.get_info()
                if cfg.zoo.pre.report_error is True:
                    fim = self.zoo.net.get_fim(self.train_loader)
                    estim_error = self.zoo.precon.compute_error(fim)
                else:
                    estim_error = 0
            else:
                num_update = samples_estim = max_cond_num = estim_error = 0
            _, te_acc = util.calc_loss_accuracy_zo(self.zoo.net, self.test_loader)
            tr_loss, tr_acc = util.calc_loss_accuracy_zo(self.zoo.net, self.train_loader)
            memory_GB = torch.cuda.max_memory_allocated() / 1024**3
            self.progr.progress_report(estim_error, num_update, samples_estim, max_cond_num, tr_loss, te_acc, self.zoo.dWeights.shape[1], total_iterations, self.zoo.cost_func_eval, memory_GB)


class ZOoptim():
    def __init__(self, cfg: DictConfig, net, optimizer):
        self.cfg = cfg.zoo
        self.net = net
        self.optimizer = optimizer
        self.device = list(self.net.parameters())[0].device
        self.cost_func_eval = int(0)
        self.target_params = []
        self.nDim = 0
        for p in self.net.parameters():
            if p.requires_grad is True:
                self.target_params.append(p)
                self.nDim += int(math.prod(p.shape))
        self.num_params = split_number_last_only_minmax(self.nDim, self.cfg.max_dim)
        self.gen_multi = torch.Generator().manual_seed(cfg.seed)

    def initialize_batch_and_forward(self, input: Tensor, label: Tensor) -> Tensor:
        self.input = input
        self.label = label
        self.parameters_vector = torch.cat([p.flatten() for p in self.target_params])
        self.output0 = self._chip_forward(self.parameters_vector[:,None])[0]
        self.loss0 = self.net.criterion(self.output0, self.label).mean()
        if self.cfg.pre.empiricalFIM is True:
            self.output_fisher = self.net.calc_output_fisher_empirical(self.output0, label)
        else:
            self.output_fisher = self.net.calc_output_fisher(self.output0)
        return self.loss0

    def perform_I(self, num_vectors):
        self.perturb_normal(num_vectors)
        self.evaluate()
        grad = self.calc_grad()
        self.grad_and_step(grad)
        return grad

    def perturb_normal(self, num_vectors: int) -> list:
        total_num_params = sum(self.num_params)
        self.dWeights = torch.randn((total_num_params, num_vectors), device=self.device)
   
    def calc_cov_fsd(self) -> Tensor:
        dOutputs = self.dOutputs
        sha = dOutputs.shape
        if len(sha) == 4:  # RNNs
            cov_fsd = torch.einsum('pbms,bmks,qbks->pq', dOutputs, self.output_fisher, dOutputs) / sha[1] / sha[3]
        else:  # other NNs
            cov_fsd = torch.einsum('pbm,bmk,qbk->pq', dOutputs, self.output_fisher, dOutputs) / sha[1]
        return cov_fsd

    def evaluate(self, for_fim: bool = False) -> None:
        parameters_tensor = self.parameters_vector[:,None] + self.cfg.mu * self.dWeights
        outputs_p = self._chip_forward(parameters_tensor)
        if self.cfg.diff == 'forward' or for_fim is True:
            self.dOutputs = (outputs_p - self.output0[None]) / self.cfg.mu
        elif self.cfg.diff == 'central':
            parameters_tensor = self.parameters_vector[:,None] - self.cfg.mu * self.dWeights
            outputs_m = self._chip_forward(parameters_tensor)
            self.dOutputs = 0.5 * (outputs_p - outputs_m) / self.cfg.mu
        else:
            raise ValueError('Unknown self.cfg.diff')
        if for_fim is False:
            n_pert = self.dWeights.shape[1]
            if outputs_p.ndim == 4:  # RNNs
                outputs_p = outputs_p.permute(1,2,3,0)
            else:  # other NNs
                outputs_p = outputs_p.permute(1,2,0)
            lab = self.label
            loss_p = self.net.criterion(outputs_p, lab[...,None].expand(*lab.shape, n_pert)).mean(dim=tuple(range(lab.ndim)))
            if self.cfg.diff == 'forward':
                self.dLosses = (loss_p - self.loss0) / self.cfg.mu
            elif self.cfg.diff == 'central':
                if outputs_m.ndim == 4:  # RNNs
                    outputs_m = outputs_m.permute(1,2,3,0)
                else:  # other NNs
                    outputs_m = outputs_m.permute(1,2,0)
                loss_m = self.net.criterion(outputs_m, lab[...,None].expand(*lab.shape, n_pert)).mean(dim=tuple(range(lab.ndim)))
                self.dLosses = 0.5 * (loss_p - loss_m) / self.cfg.mu

    def _chip_forward(self, param: Tensor) -> Tensor:
        nParams, nVariation = param.shape
        self.cost_func_eval += int(nVariation)
        # expand parameters for nVariation
        params_backup = []
        idx_start = 0
        for p in self.target_params:
            params_backup.append(p.clone())
            psh = p[0].shape
            length = math.prod(psh)
            extracted = param[idx_start:idx_start+length]
            extracted = extracted.T.reshape((nVariation,) + psh)
            p.data = extracted.data
            idx_start += length
        expanded_inputs = torch.stack((self.input, ) * nVariation)
        with torch.no_grad():
            outputs = self.net(expanded_inputs).detach()
        # revert to the original parameters
        for p, b in zip(self.target_params, params_backup):
            p.data = b.data
        return outputs

    def grad_and_step(self, grad):
        idx_start = 0
        for p in self.target_params:
            p_shape = p.shape
            length = math.prod(p_shape)
            g = grad[idx_start:idx_start+length]
            p.grad = g.reshape(p_shape)
            idx_start += length
        self.optimizer.step()
        self.optimizer.zero_grad()

    def calc_grad(self, average=True) -> Tensor:
        grad = torch.matmul(self.dWeights, self.dLosses)
        if average is True:
            grad /= self.dLosses.shape[0]
        return grad

    def is_onehot_perturbations0(self, total_iterations):
        if total_iterations % self.cfg.pre.Tud == 0:
            coordinate = True
        else:
            coordinate = False
        return coordinate


class ZOoptimEstimPCP(ZOoptim):
    def __init__(self, cfg: DictConfig, net, optimizer):
        super().__init__(cfg, net, optimizer)
        if cfg.zoo.vectors == 'F1' or cfg.zoo.vectors == 'F2':
            absolute_when_clip = False
        elif cfg.zoo.vectors == 'H1' or cfg.zoo.vectors == 'H2':
            absolute_when_clip = True
        else:
            raise ValueError('Unknown cfg.zoo.vectors')
        if cfg.zoo.vectors == 'H1':
            assert cfg.zoo.diff == 'central'
        self.precon = precon.Preconditioner(self.num_params, self.device, self.cfg.pre, absolute_when_clip)

    def initFIMonehot(self):
        FIM_list = []
        start = 0
        for i, d_blk in enumerate(self.num_params):
            self.dWeights = torch.zeros(self.nDim, d_blk, device=self.device)
            self.dWeights[start:start+d_blk, :] = torch.eye(d_blk, device=self.device)
            self.evaluate(for_fim=True)
            FIM = self.calc_cov_fsd()
            FIM_list.append(FIM)
            start += d_blk
        self.precon.initFIM(FIM_list)

    def perform_F(self, num_vectors, vectors):  # line 5-9, Algorithm 1
        perturbI, self.dWeights = self.precon.get_perturbations(num_vectors)
        self.evaluate()
        cov_fsd = self.calc_cov_fsd()
        if vectors == 'F1':
            self.precon.estimator_F1(perturbI, cov_fsd)
        else:  # 'F2'
            self.precon.estimator_F2(perturbI, cov_fsd)
        grad = self.calc_grad()
        self.grad_and_step(grad)
        return grad

    def perform_H1(self, num_vectors):
        perturbI, self.dWeights = self.precon.get_perturbations(num_vectors)
        self.evaluate_H1()
        self.precon.estimator_H1(perturbI, self.Hquads)
        grad = self.calc_grad()
        self.grad_and_step(grad)
        return grad

    def perform_H2(self, num_vectors, diff):
        pertIdouble, pertWdouble = self.precon.get_perturbations(num_vectors)
        if diff == 'forward':
            self.evaluate_H2forward(pertWdouble)
        else:  # 'central'
            self.evaluate_H2central(pertWdouble)
        self.precon.estimator_H2(pertIdouble, self.Hbilinear)
        grad = self.calc_grad()
        self.grad_and_step(grad)
        return grad

    def evaluate_H1(self):
        loss_p = self._get_perturbed_loss(self.dWeights)
        loss_m = self._get_perturbed_loss(-self.dWeights)
        self.dLosses = (loss_p - loss_m) / (2 * self.cfg.mu)
        self.Hquads = (loss_p + loss_m - 2 * self.loss0) / (self.cfg.mu**2)

    def evaluate_H2forward(self, pert_double):
        pert1 = pert_double[:,::2]
        pert2 = pert_double[:,1::2]
        assert pert1.shape[1] == pert2.shape[1]
        loss_12 = self._get_perturbed_loss(pert1+pert2)
        loss_1   = self._get_perturbed_loss(pert1)
        loss_2 = self._get_perturbed_loss(pert2)
        dLoss1 = (loss_1 - self.loss0) / self.cfg.mu
        dLoss2 = (loss_2 - self.loss0) / self.cfg.mu
        self.dLosses = torch.cat((dLoss1, dLoss2))
        self.dWeights = torch.cat((pert1, pert2), dim=1)
        self.Hbilinear = (loss_12 - loss_1 - loss_2 + self.loss0) / (self.cfg.mu**2)

    def evaluate_H2central(self, pert_double):
        pert1 = pert_double[:,::2]
        pert2 = pert_double[:,1::2]
        assert pert1.shape[1] == pert2.shape[1]
        loss_1p2p = self._get_perturbed_loss( pert1 +pert2)
        loss_1p   = self._get_perturbed_loss( pert1)
        loss_1m2p = self._get_perturbed_loss(-pert1 +pert2)
        loss_1m   = self._get_perturbed_loss(-pert1)
        dLoss1 = (loss_1p - loss_1m) / (2 * self.cfg.mu)
        self.dLosses = dLoss1
        self.dWeights = pert1
        self.Hbilinear = (loss_1p2p - loss_1p - loss_1m2p + loss_1m) / (2 * self.cfg.mu**2)

    def _get_perturbed_loss(self, perturb):
        n_pert = perturb.shape[1]
        parameters_tensor = self.parameters_vector[:,None] + self.cfg.mu * perturb
        outputs = self._chip_forward(parameters_tensor)
        if outputs.ndim == 3:
            outputs = outputs.permute(1,2,0)
        elif outputs.ndim == 4:
            outputs = outputs.permute(1,2,3,0)
        lab = self.label
        loss = self.net.criterion(outputs, lab[...,None].expand(*lab.shape, n_pert)).mean(dim=tuple(range(lab.ndim)))
        return loss

    def get_info(self):
        if len(self.precon.num_samples_list) >= 1:
            num_estim_samples = self.precon.num_samples_list[-1]
        else:
            num_estim_samples = 0
        return self.precon.num_update_by_perturb, num_estim_samples, self.precon.max_cond_number
