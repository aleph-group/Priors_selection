import torch 
from utils import tcheby, tcheby_der, device
from deepinv.sampling import DiffPIR  as dinv_DiffPIR 
from deepinv.optim.data_fidelity import L2


class Sampler:
    def __init__(self, gradU, gamma, X_init, proj):
        self.gradU = gradU
        self.gamma = gamma
        self.proj = proj
        self.X = X_init.clone().to(device)

    def __call__(self, *args, **kwargs):
        return self.proj(self.X)

    def update_state(self, X):
        self.X = X.clone().to(device)


class ULA(Sampler):
    def __init__(self, gradU, gamma, X_init, proj):
        super().__init__(gradU, gamma, X_init, proj)

    def __call__(self, *args, **kwargs):
        with torch.no_grad():
            self.X = self.proj(self.X - self.gamma * self.gradU(self.X, *args, **kwargs) + 
                               torch.sqrt(2*self.gamma) * torch.randn_like(self.X))
        return self.X


class SKROCK(Sampler):
    def __init__(self, gradU, gamma, X_init, proj, s=3, eta=0.05):
        super().__init__(gradU, gamma, X_init, proj)
        self.s, self.eta = torch.tensor(s), torch.tensor(eta)
        w0 = 1 + self.eta / (self.s ** 2)
        w1 = tcheby(w0, s) / tcheby_der(w0, s)
        self.muj, self.nuj, self.kj = torch.zeros(s+1, device=device), torch.zeros(s+1 , device=device), torch.zeros(s+1, device=device)
        self.muj[1] = w1 / w0
        self.nuj[1] = self.s * w1 / 2.
        self.kj[1] = self.s * w1 / w0
        for j in range(2, s + 1):
            self.muj[j] = 2 * w1 * tcheby(w0, j-1) / tcheby(w0, j)
            self.nuj[j] = 2 * w0 * tcheby(w0, j-1) / tcheby(w0, j)
        self.kj[2:] = 1 - self.nuj[2:]

    def __call__(self, *args, **kwargs):
        with torch.no_grad():
            K = self.X
            Z = torch.sqrt(2*self.gamma)*torch.randn_like(self.X)
            Kp = (self.X - self.muj[1]*self.gamma*self.gradU(self.X + self.nuj[1]*Z, *args, **kwargs) +
                  self.kj[1]*Z)
            for j in range(2, self.s + 1):
                Kpp = - self.muj[j]*self.gamma*self.gradU(Kp, *args, **kwargs) + self.nuj[j]*Kp + self.kj[j]*K
                K, Kp = Kp, Kpp
            self.X = self.proj(Kpp.clone())
        return self.X


class Gaussian(Sampler):
    def __init__(self, gradU, gamma, X_init, proj, Q, D):
        super().__init__(None, None, X_init, proj)
        self.Q, self.D = Q, D
        self.d = self.Q.shape[0]
    def __call__(self, mean):
        with torch.no_grad():
            self.X = self.proj((self.Q @ torch.sqrt(torch.diag(self.D))) @ torch.randn(self.d, device=device) + torch.reshape(mean, (-1,)))
        return self.X
        

class GaussianDiag(Sampler):
    def __init__(self, gradU, gamma, X_init, proj, d, sigma):
        super().__init__(None, None, X_init, proj)
        self.sigma, self.d = sigma, d
        self.d = d
    def __call__(self, mean):
        with torch.no_grad():
            self.X = self.proj(torch.randn((self.X.shape[0], 1, self.d), device=device)*self.sigma + mean)
        return self.X


class DiffPIR(Sampler):
    def __init__(self, gradU, gamma, X_init, proj, y,
                 physics, denoiser, verbose=False, batch_size=1, max_iter=500):
        super().__init__(None, None, X_init, proj)
        self.model = dinv_DiffPIR(data_fidelity=L2(), model=denoiser, device=device, 
                                  verbose=verbose, max_iter=max_iter)
        self.p = physics
        self.y = y.repeat(batch_size, 1, 1, 1)
        
    def __call__(self, *args, **kwargs):
        return self.model(self.y, self.p)
        