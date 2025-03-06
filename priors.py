import deepinv as dinv
import torch
from utils import device


class ParametrizedPrior:
    def __init__(self, param):
        self.param = torch.clone(param).to(device)

    def forward(self, x, *args, **kwargs):
        pass

    def __call__(self, x, *args, **kwargs):
        return self.forward(x, **kwargs)
        
    def grad(self, x, lam_reg=None):
        pass

    def grad_param(self, x):
        pass
        
    def update_param(self, new_param):
        self.param = torch.clone(new_param)


class Likelihood:
    def __init__(self):
        pass

    def __call__(self, x, y):
        return self.f(x,y)
    
    def f(self, x, y):
        pass

    def grad(self, x, y):
        pass


class L2(Likelihood):
    def __init__(self, sigma, p):
        super().__init__()
        self.sigma = sigma
        self.p = p
        
    def f(self, x, y):
        return 0.5 * torch.norm(y - self.p.A(x)) / 2 / self.sigma**2

    def grad(self, x, y):
        return self.p.A_adjoint(self.p.A(x) - y) / self.sigma**2


class CombinedPrior(ParametrizedPrior):
    def __init__(self, param, prior1, prior2):
        super().__init__(param)
        self.prior1 = prior1
        self.prior2 = prior2

    def grad(self, x, lam_reg):
        return self.param * self.prior1.grad(x, lam_reg) + (1 - self.param) * self.prior2.grad(x, lam_reg)

    def forward(self, x):
        return self.param * self.prior1(x) + (1 - self.param) * self.prior2(x)

    def grad_param(self, x):
        return  self.prior1.grad_param(x) - self.prior2.grad_param(x)


class GSPnP(dinv.optim.prior.RED):
    r"""s
    Gradient-Step Denoiser prior.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.explicit_prior = True

    def forward(self, x, *args, **kwargs):
        r"""
        Computes the prior :math:`g(x)`.

        :param torch.tensor x: Variable :math:`x` at which the prior is computed.
        :return: (torch.tensor) prior :math:`g(x)`.
        """
        return self.denoiser.potential(x, *args, **kwargs)


class WaveletPrior(ParametrizedPrior):
    def __init__(self, param):
        super().__init__(param)
        self.dinv_tv = dinv.optim.prior.WaveletPrior(level=3, wv=["db{}".format(i) for i in range(8, 9)], p=1, device=device)

    def grad(self, x, lam_reg):
        return (x - self.dinv_tv.prox(x, self.param*lam_reg)) / lam_reg  

    def forward(self, x):
        return self.dinv_tv(x)*self.param

    def grad_param(self, x):
        return self.dinv_tv(x)

class TikhonovPrior(ParametrizedPrior):
    def __init__(self, param):
        super().__init__(param)

    def grad(self, x, lam_reg):
        return x * self.param

    def forward(self, x):
        return torch.norm(x)**2 * self.param / 2.

    def grad_param(self, x):
        return torch.norm(x)**2 / 2.

class L1Prior(ParametrizedPrior):
    def __init__(self, param):
        super().__init__(param)

    def grad(self, x, lam_reg):
        return (x - torch.sign(x) * 
                torch.max(torch.abs(x) - lam_reg * self.param, 
                          torch.zeros_like(x))) / lam_reg

    def forward(self, x):
        return torch.sum(torch.abs(x)) * self.param 

    def grad_param(self, x):
        return torch.sum(torch.abs(x))

class RedPrior(ParametrizedPrior):
    def __init__(self, param, dinv_red_prior, sigma_denoiser=0.1):
        super().__init__(param)
        self.dinv_red_prior = dinv_red_prior
        self.sigma_denoiser = sigma_denoiser
        
    def grad(self, x, lam_reg):
        return self.param * self.dinv_red_prior.grad(x, sigma_denoiser=self.sigma_denoiser)

    def forward(self, x):
        return self.param * self.dinv_red_prior(x, sigma=self.sigma_denoiser) 

    def grad_param(self, x):
        return  self.dinv_red_prior(x, sigma=self.sigma_denoiser) 