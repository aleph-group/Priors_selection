from deepinv.optim.prior import WaveletPrior as _DeepInvWaveletPrior
from deepinv.optim.prior import TVPrior as dinv_tv

import torch
from utils import device


class ParametrizedPrior:
    def __init__(self, param):
        self.param = torch.clone(param).detach().to(device)

    def forward(self, x, *args, **kwargs):
        pass

    def __call__(self, x, *args, **kwargs):
        return self.forward(x, **kwargs)

    def grad(self, x, lam_reg=None):
        pass

    def grad_param(self, x):
        pass

    def lipsch_bound(self, lam_reg=None):
        if lam_reg is not None:
            return 1 / lam_reg
        else:
            return -1

    def update_param(self, new_param):
        self.param = torch.clone(new_param)


class Likelihood:
    def __init__(self):
        pass

    def __call__(self, x, y, dim=None):
        return self.f(x, y, dim)

    def f(self, x, y, dim=None):
        pass

    def grad(self, x, y):
        pass


class L2(Likelihood):
    def __init__(self, sigma, p):
        super().__init__()
        self.sigma = sigma
        self.p = p

    def f(self, x, y, dim=None):
        return 0.5 * torch.sum(torch.square(y - self.p.A(x)), dim=dim) / self.sigma**2

    def grad(self, x, y):
        return self.p.A_adjoint(self.p.A(x) - y) / self.sigma**2


class GSDPrior(ParametrizedPrior):
    r"""s
    Gradient-Step Denoiser prior.
    """

    def __init__(self, param, denoiser, sigma_denoiser=0.1):
        super().__init__(param)
        self.denoiser = denoiser
        self.sigma_denoiser = sigma_denoiser

    def forward(self, x):
        return self.param * self.denoiser.potential(x, self.sigma_denoiser)

    def grad(self, x, lam_reg=None):
        return self.param * (x - self.denoiser(x, self.sigma_denoiser))

    def grad_param(self, x):
        return self.denoiser.potential(x, self.sigma_denoiser)

    def lipsch_bound(self, lam_reg=None):
        return self.param * 5


class REDPrior(ParametrizedPrior):
    r"""Regularization by Denoising (RED) prior.
    Uses gradient: \lambda (x - D(x, \sigma)) where D is any denoiser.
    Works with denoisers that lack an explicit potential (e.g. DRUNet, DnCNN).
    """

    def __init__(self, param, denoiser, sigma_denoiser=None):
        super().__init__(param)
        self.denoiser = denoiser
        self.sigma_denoiser = sigma_denoiser  # None for noise-blind denoisers (DnCNN)

    def forward(self, x):
        Dx = self.denoiser(x, self.sigma_denoiser)
        return self.param * 0.5 * torch.sum(x * (x - Dx))

    def grad(self, x, lam_reg=None):
        return self.param * (x - self.denoiser(x, self.sigma_denoiser))

    def grad_param(self, x):
        Dx = self.denoiser(x, self.sigma_denoiser)
        return 0.5 * torch.sum(x * (x - Dx))

    def lipsch_bound(self, lam_reg=None):
        return self.param * 2  # conservative: ||I - D||_Lip <= 2 for non-expansive D


class WaveletPrior(ParametrizedPrior):
    def __init__(self, param, start=8, wv=None):
        super().__init__(param)
        if wv is None:
            wv = ["db{}".format(i) for i in range(start, 9)]
        self.dinv_tv = _DeepInvWaveletPrior(level=3, wv=wv, p=1, device=device)

    def grad(self, x, lam_reg):
        # Moreau gradient: (x - prox_{lam_reg * g}(x)) / lam_reg
        # where g(x) = self.param * ||Psi x||_1
        # deepinv prox uses keyword-only args: prox(x, ths=..., gamma=...)
        # effective threshold = ths * gamma = self.param * lam_reg
        return (x - self.dinv_tv.prox(x, ths=self.param, gamma=lam_reg)) / lam_reg

    def forward(self, x):
        return self.dinv_tv(x) * self.param

    def grad_param(self, x):
        return self.dinv_tv(x)


class TikhonovPrior(ParametrizedPrior):
    def __init__(self, param):
        super().__init__(param)

    def grad(self, x, lam_reg):
        return x * self.param

    def forward(self, x):
        return torch.norm(x) ** 2 * self.param / 2.0

    def grad_param(self, x):
        return torch.norm(x) ** 2 / 2.0

    def lipsch_bound(self, lam_reg):
        return self.param


class DiagonalWeightedTikhonovPrior(ParametrizedPrior):  # coded for flat vectors
    def __init__(self, param, mat):
        super().__init__(param)
        self.weights = (
            torch.reshape(mat, (1, -1)).clone().detach().to(device)
        )  # d, d matrix
        self.axis = tuple(range(1, 3))

    def grad(self, x, lam_reg):
        return self.weights[None, :, :] * x * self.param

    def forward(self, x):
        return (
            torch.sum(torch.square(x) * self.weights[None, :, :], self.axis)
            * self.param
            / 2.0
        )

    def grad_param(self, x):
        return torch.sum(torch.square(x) * self.weights[None, :, :], self.axis) / 2.0


class L1Prior(ParametrizedPrior):
    def __init__(self, param):
        super().__init__(param)

    def grad(self, x, lam_reg):
        return (
            x
            - torch.sign(x)
            * torch.max(torch.abs(x) - lam_reg * self.param, torch.zeros_like(x))
        ) / lam_reg

    def forward(self, x):
        return torch.sum(torch.abs(x)) * self.param

    def grad_param(self, x):
        return torch.sum(torch.abs(x))


class TVPrior(ParametrizedPrior):
    def __init__(self, param, n_it_max=1000):
        super().__init__(param)
        self.dinv_tv = dinv_tv(n_it_max=n_it_max)

    def grad(self, x, lam_reg):
        return (x - self.dinv_tv.prox(x, gamma=self.param*lam_reg)) / lam_reg

    def forward(self, x):
        return self.dinv_tv(x) * self.param 

    def grad_param(self, x):
        return self.dinv_tv(x)