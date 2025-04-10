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

    def lipsch_bound(self, lam_reg=None):
        if lam_reg is not None:
            return 1 / lam_reg
        else:
            return - 1
        
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
        return 0.5 * torch.sum(torch.square(y - self.p.A(x))) / self.sigma**2

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

    def lipsch_bound(self, lam_reg=None):
        return self.param*self.prior1.lipsch_bound(lam_reg) + (1-self.param)*self.prior2.lipsch_bound(lam_reg)

        
class GSDPrior(ParametrizedPrior):
    r"""s
    Gradient-Step Denoiser prior.
    """

    def __init__(self, param, denoiser, sigma_denoiser=0.1):
        super().__init__(param)
        self.denoiser = denoiser
        self.sigma_denoiser = sigma_denoiser

    def forward(self, x):
        return self.param*self.denoiser.potential(x, self.sigma_denoiser)

    def grad(self, x, lam_reg=None):
        return self.param*(x - self.denoiser(x, self.sigma_denoiser))

    def grad_param(self, x):
        return self.denoiser.potential(x, self.sigma_denoiser)

    def lipsch_bound(self, lam_reg=None):
        return self.param*5
        
class WaveletPrior(ParametrizedPrior):
    def __init__(self, param, start=8):
        super().__init__(param)
        self.dinv_tv = dinv.optim.prior.WaveletPrior(level=3, wv=["db{}".format(i) for i in range(start, 9)], p=1, device=device)

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

    def lipsch_bound(self, lam_reg):
        return self.param


class WeightedTikhonovPrior(ParametrizedPrior):  # coded for flat vectors
    def __init__(self, param, mat):
        super().__init__(param)
        self.weights = mat  # d, d matrix
        
    def grad(self, x, lam_reg):
        return torch.reshape(self.weights @ torch.reshape(x, (-1, 1)), (1, 1, -1)) * self.param

    def forward(self, x): 
        return torch.sum(torch.reshape(x, (-1, 1))* self.weights@torch.reshape(x, (-1, 1))) * self.param / 2.

    def grad_param(self, x):
        return torch.sum(torch.reshape(x, (-1, 1))* self.weights@torch.reshape(x, (-1, 1))) / 2.



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


class CRRPrior(ParametrizedPrior):
    def __init__(self, param, crr_model):
        super().__init__(param)  # param = lambda, mu
        self.crr_model = crr_model

    def forward(self, x):
        return self.param[0] * self.crr_model.cost(self.param[1]*x) / self.param[1]

    def grad(self, x, lam_reg):
        return self.param[0] * self.crr_model(self.param[1]*x)
        
    def grad_param(self, x):
        return torch.tensor([self.crr_model.cost(self.param[1]*x) / self.param[1], 
                             self.param[0] * (torch.sum(x*self.crr_model(self.param[1]*x)) / self.param[1] -
                                              self.crr_model.cost(self.param[1]*x) / self.param[1]**2)], 
                            device=device)

    def lipsch_bound(self, lam_reg=None):
        return self.crr_model.precise_lipschitz_bound(n_iter=500)*self.param[0]*self.param[1]
        
        
class CRRPriorStaticMu(ParametrizedPrior):
    def __init__(self, param, crr_model):
        super().__init__(param[0])  # param = lambda, mu
        self.crr_model = crr_model
        self.mu = param[1]

    def grad_param(self, x):
        return self.crr_model.cost(self.mu*x) / self.mu

    def forward(self, x):
        return self.param * self.crr_model.cost(self.mu*x) / self.mu

    def grad(self, x, lam_reg):
        return self.param * self.crr_model(self.mu*x)

    def lipsch_bound(self, lam_reg=None):
        return self.crr_model.precise_lipschitz_bound(n_iter=500)*self.param*self.mu