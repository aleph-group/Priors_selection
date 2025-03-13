import torch 
from priors import L2
import tqdm
from utils import device
from sampling import ULA

class DegradedLikelihood:
    def __init__(self, y, prior, physics, sigma, gamma,
                 X_init=None, lam_reg=None, project='clamp', noise=None):
        """
        y: observations
        prior: model tested
        physics: forward operator implemented through deepinv
        sigma: noise std
        gamma: MC step
        lam_reg: regularization parameter for the prox
        project: 'clamp' to project on [0,1], 'refl' to apply absolute value, no projection otherwise
        noise: initial additional degradation. A new sample is generated if None.
        """
        self.prior = prior
        self.f = L2(sigma, physics)  # likelihood for Ax + N
        self.f_add = L2(2**0.5*sigma, physics)  # likelihood for Ax + N + e

        self.y = y
        self.dimx = y.numel()
        self.gamma = gamma  # MC step
        self.lam_reg = lam_reg  # prior regularization parameter

        if project == 'clamp':
            proj = lambda t: torch.clamp(t, 0., 1.)
        elif project == 'refl':
            proj = lambda t: torch.abs(t)
        else:
            proj = lambda t: t  
            
        if X_init is None:
            X_post = proj(torch.randn_like(y)).to(device)
        else: 
            X_post = proj(X_init.clone())
        self.y_sub, self.y_add = None, None
        if noise is None:
            self._add_noise()
        else:
            self.y_sub, self.y_add = self.y - noise, self.y + noise
        
        gradU = lambda t, y: self.f_add.grad(t, y) + self.prior.grad(t, lam_reg)
        self.sampler = ULA(gradU, gamma, X_post, proj=proj)

    def _add_noise(self):
        noise = torch.randn_like(self.y)*self.f.sigma
        self.y_sub, self.y_add = self.y - noise, self.y + noise 

    def compute_test(self, nb_steps, log_stats=False, burnin_ratio=0.25, thinning=1, tol=1e-4, patience=10):
        
        it_burnin = int(burnin_ratio*nb_steps)
        n_rem = nb_steps - it_burnin
        
        if log_stats:
            post_hist = torch.zeros(n_rem, device=device)
            X_post_trace = torch.zeros([n_rem, self.dimx]).to(device)
            lik_trace = torch.zeros(n_rem, device=device)
        for n in tqdm.tqdm(range(it_burnin)):  # warmup stage
            with torch.no_grad():
                self.sampler(self.y_sub)
        
        lik1_mean = torch.tensor(0., device=device)

        trange = tqdm.tqdm(range(n_rem))
        for n in trange:
            trange.set_description("t={:.4f}".format(lik1_mean.item()))

            with torch.no_grad():
                for _ in range(thinning):
                    self.sampler(self.y_sub)
                    
                lik1 = self.f_add(self.sampler.X, self.y_add).detach()
                lik1_mean = (lik1_mean*n + lik1) / (n + 1)  # p(y+/x)

                if log_stats:
                    X_post_trace[n] =  torch.flatten(self.sampler.X).detach()
                    post_hist[n] = lik1 + self.prior(self.sampler.X)             
                    lik_trace[n] = lik1             

                if n % patience == 0:
                    if n > 0:
                        if torch.abs(old_mean - lik1_mean) < tol:
                            break
                        else:
                            old_mean = lik1_mean.clone()
                    else:
                        old_mean = lik1_mean.clone()
        if log_stats:
            return X_post_trace[:n+1].cpu(), post_hist[:n+1].cpu(), lik_trace[:n+1].cpu()
        else:
            return lik1_mean.item() 
            

    def compute_test2(self, nb_steps, burnin_ratio=0.25, thinning=10, tol=1e-4, patience=10):
        lik1_mean = torch.tensor(0., device=device)

        it_burnin = int(burnin_ratio*nb_steps)
        n_rem = nb_steps - it_burnin

        self._add_noise()
        for _ in tqdm.tqdm(range(it_burnin)):  # warmup stage
            with torch.no_grad():
                self.sampler(self.y_sub)

        trange = tqdm.tqdm(range(n_rem))
        for n in trange:
            trange.set_description("t={:.4f}".format(lik1_mean.item()))
            with torch.no_grad():
                self._add_noise()  # regenerate additional noise
                for _ in range(thinning):
                    self.sampler(self.y_sub)
                
                lik1_mean = (lik1_mean*n + self.f_add(self.sampler.X, self.y_add).detach()) / (n + 1)
                if n % patience == 0:
                    if n > 0:
                        if torch.abs(old_mean - lik1_mean) < tol:
                            break
                        else:
                            old_mean = lik1_mean.clone()
                    else:
                        old_mean = lik1_mean.clone()

        return lik1_mean.item()