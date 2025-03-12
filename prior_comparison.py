import torch 
from priors import L2
import tqdm
from utils import device


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
            self.proj = lambda t: torch.clamp(t, 0., 1.)
        elif project == 'refl':
            self.proj = lambda t: torch.abs(t)
        else:
            self.proj = lambda t: t  
            
        if X_init is None:
            self.X_post = self.proj(torch.randn_like(y)).to(device)
        else: 
            self.X_post = self.proj(X_init.clone())
        self.y_sub, self.y_add = None, None
        if noise is None:
            self._add_noise()
        else:
            self.y_sub, self.y_add = self.y - noise, self.y + noise
    
    def _ULA_it(self, y):
        """
        ULA step given the observations y.
        """
        with torch.no_grad():
            self.X_post =  self.proj(self.X_post - self.gamma * self.f_add.grad(self.X_post, y) - 
                                     self.gamma * self.prior.grad(self.X_post, self.lam_reg) + 
                                     torch.sqrt(2*self.gamma) * torch.randn_like(self.X_post))

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
                self._ULA_it(self.y_sub)
        
        lik1_mean = torch.tensor(0., device=device)

        trange = tqdm.tqdm(range(n_rem))
        for n in trange:
            trange.set_description("t={:.4f}".format(lik1_mean.item()))

            with torch.no_grad():
                for _ in range(thinning):
                    self._ULA_it(self.y_sub)
                    
                lik1 = self.f_add(self.X_post, self.y_add).detach()
                lik1_mean = (lik1_mean*n + lik1) / (n + 1)  # p(y+/x)

                if log_stats:
                    X_post_trace[n] =  torch.flatten(self.X_post).detach()
                    post_hist[n] = lik1 + self.prior(self.X_post)             
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
                self._ULA_it(self.y_sub)

        trange = tqdm.tqdm(range(n_rem))
        for n in trange:
            trange.set_description("t={:.4f}".format(lik1_mean.item()))
            with torch.no_grad():
                self._add_noise()  # regenerate additional noise
                for _ in range(thinning):
                    self._ULA_it(self.y_sub)
                
                lik1_mean = (lik1_mean*n + self.f_add(self.X_post, self.y_add).detach()) / (n + 1)
                if n % patience == 0:
                    if n > 0:
                        if torch.abs(old_mean - lik1_mean) < tol:
                            break
                        else:
                            old_mean = lik1_mean.clone()
                    else:
                        old_mean = lik1_mean.clone()

        return lik1_mean.item()