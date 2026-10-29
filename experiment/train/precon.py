''' Perturbation preconditioner '''
import torch


class Preconditioner:
    def __init__(self, num_params, device, cfg, absolute_when_clip):
        d_blk = num_params[0]
        n_blk = len(num_params)
        self.total_dim = sum(num_params)
        self.d_last_blk = num_params[-1]
        self.W = torch.eye(d_blk, device=device)[None].repeat(n_blk, 1, 1)
        self.Ghat = cfg.initFscale * torch.eye(d_blk, device=device)[None].repeat(n_blk, 1, 1)
        self.alpha = cfg.alpha
        self.Tud = cfg.Tud
        self.call_since_last_updateW = 0
        self.update_type = cfg.update_type
        self.gamma = cfg.gamma
        self.tau = cfg.tau
        self.estim_samples_since_last_updateW = 0
        self.num_update_by_perturb = 0
        self.error_list = []
        self.num_samples_list = []
        self.max_cond_number = 1
        self.absolute_when_clip = absolute_when_clip

    def get_perturbations(self, num_vectors):
        n_blk, d_blk, _ = self.W.shape
        pert = torch.randn((n_blk, d_blk, num_vectors), device=self.W.device)
        Wpert = torch.bmm(self.W, pert)
        Wpert = Wpert.reshape((-1, num_vectors))[:self.total_dim]
        pert[-1,self.d_last_blk:] = 0
        return pert, Wpert

    def initFIM(self, FIM_list):
        n_blk = len(FIM_list)
        for i in range(n_blk):
            if i == n_blk - 1:
                self.Ghat[i][:self.d_last_blk,:self.d_last_blk] = FIM_list[i]
            else:
                self.Ghat[i][:] = FIM_list[i]
        self._update_operator_initFIM()

    def _update_operator_initFIM(self):
        n_blk, d_blk, _ = self.W.shape
        self.W = torch.eye(d_blk, device=self.W.device)[None].repeat(n_blk, 1, 1)
        tau = 1e-2
        clipG = self._force_positive_semidefinite(self.Ghat, tau)
        Ghat = clipG + self.gamma * torch.eye(d_blk, device=self.W.device)[None].repeat(n_blk, 1, 1)
        L = torch.linalg.cholesky(Ghat)
        self.W = torch.linalg.solve_triangular(L.mT, self.W, upper=True, left=False)
        C = torch.linalg.cholesky(clipG)
        LinvC = torch.linalg.solve_triangular(L, C, upper=False, left=True)
        self.Ghat = LinvC @ LinvC.mT

    def estimator_F1(self, U, C):
        diagC = C.diag()
        n_blk, d_blk, num_samples = U.shape
        identity = torch.eye(d_blk, device=U.device)[None].expand(n_blk, d_blk, d_blk)
        Gtilde = torch.einsum('bmq,q,bnq->bmn', U, diagC, U) - diagC.sum() * identity
        Gtilde /= (2*num_samples)
        self._update_estimator(Gtilde, num_samples)

    def estimator_F2(self, U, C):
        sizeC = C.shape[-1]
        num_samples = int(sizeC*(sizeC-1) / 2)
        idx = torch.arange(sizeC)
        C[idx,idx] = 0
        n_blk, d_blk, n_pert = U.shape
        C_expanded = C[None].expand(n_blk, sizeC, sizeC)
        Gtilde = (U @ C_expanded @ U.mT) / 2 / num_samples
        self._update_estimator(Gtilde, num_samples)

    def estimator_H1(self, U, Hquads):
        n_blk, d_blk, num_samples = U.shape
        identity = torch.eye(d_blk, device=U.device)[None].expand(n_blk, d_blk, d_blk)
        Gtilde = torch.einsum('bmq,q,bnq->bmn', U, Hquads, U) - Hquads.sum() * identity
        Gtilde /= (2*num_samples)
        self._update_estimator(Gtilde, num_samples)

    def estimator_H2(self, Udouble, Hbilinear):
        U1 = Udouble[:,:,::2]
        U2 = Udouble[:,:,1::2]
        n_blk, d_blk, num_samples = U1.shape
        Gtilde = torch.einsum('bmq,q,bnq->bmn', U1, Hbilinear, U2) / num_samples
        Gtilde = 0.5 * (Gtilde + Gtilde.mT)
        self._update_estimator(Gtilde, num_samples)

    def _update_estimator(self, Gtilde, num_samples):
        self.num_samples_list.append(num_samples)
        self.call_since_last_updateW += 1
        if self.call_since_last_updateW >= self.Tud:
            alpha_t = self.alpha
            self.Ghat.lerp_(Gtilde, alpha_t)
            self._update_operator()
            self.num_update_by_perturb += 1
            self.call_since_last_updateW = 0
        else:
            alpha_t = self.alpha
            self.Ghat.lerp_(Gtilde, alpha_t)

    def _force_positive_semidefinite(self, G, floor=0.0):
        G = 0.5 * (G + G.mT)
        D, Q = torch.linalg.eigh(G)
        if self.absolute_when_clip is True:
            D_clamped = torch.clamp(abs(D), min=floor)
        else:
            D_clamped = torch.clamp(D, min=floor)
        forcedG = (Q * D_clamped.unsqueeze(-2)) @ Q.mT
        return forcedG

    def _update_operator(self):
        n_blk, d_blk, _ = self.W.shape
        identity = torch.eye(d_blk, device=self.W.device)[None].expand(n_blk, d_blk, d_blk)
        if self.update_type == 'whiten':
            clipG = self._force_positive_semidefinite(self.Ghat, self.tau)
            R = self.gamma * (self.W.mT @ self.W) +  clipG
            self.max_cond_number = max(torch.linalg.cond(R))
            L = torch.linalg.cholesky(R)
            Wupdate = torch.linalg.solve_triangular(L.mT, identity, upper=True, left=False)
            self.W = self.W @ Wupdate
            self.Ghat = Wupdate.mT @ clipG @ Wupdate
        elif self.update_type == 'solve_clip' or self.update_type == 'clip_solve':
            invW = torch.linalg.solve_triangular(self.W, identity, upper=True)
            if self.update_type == 'solve_clip':
                Fhat = invW.mT @ self.Ghat @ invW
                Fhat = self._force_positive_semidefinite(Fhat, self.tau)
            elif self.update_type == 'clip_solve':  # will fail in torch.linalg.cholesky(Pinv)
                clipG = self._force_positive_semidefinite(self.Ghat, self.tau)
                Fhat = invW.mT @ clipG @ invW
            else:
                raise ValueError('Unknown cfg.zoo.pre.update_type')
            Pinv = self.gamma * identity + Fhat
            self.max_cond_number = max(torch.linalg.cond(Pinv))
            L = torch.linalg.cholesky(Pinv)
            self.W = torch.linalg.solve_triangular(L.mT, identity, upper=True)
            self.Ghat = self.W.mT @ Fhat @ self.W
        else:
            assert self.update_type == 'none'

    def compute_error(self, fim):
        n_blk, d_blk, _ = self.W.shape
        I = torch.eye(d_blk, device=self.W.device)
        invW = torch.linalg.solve(self.W, I)
        Fhat = invW.mT @ self.Ghat @ invW
        # error = (fim - Fhat).norm() / fim.norm()
        error = (fim - Fhat).norm()
    
        self.error_list.append(error.item())
        return error.item()
