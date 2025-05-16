import torch 
from priors import L2
import tqdm
from utils import device
from sampling import ULA, GaussianDiag


class DegradedLikelihood:
    def __init__(self, y, prior, physics, sigma, gamma, sampler=ULA, sampler_kwargs={}, batch_size=1,
                 X_init=None, lam_reg=None, project='clamp', noise=None, alpha=0.5, **kwargs):
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
        self.alpha = torch.clone(alpha).detach()
        self.f_add = L2(sigma / torch.sqrt(1-self.alpha), physics)  # likelihood for Ax + N + e*calpha
        self.f_sub = L2(sigma / torch.sqrt(self.alpha), physics)  # likelihood for Ax + N + e/calpha

        self.calpha = torch.sqrt(self.alpha / (1-self.alpha))
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
            X_post = proj(torch.randn((batch_size,) + y.shape[1:])).to(device)
        else: 
            X_post = proj(X_init.clone())
            if X_post.shape[0] != batch_size:
                X_post = X_post.repeat(*[batch_size] + [1 for _ in range(X_post.dim()-1)]) 
            
        self.batch_size = batch_size
            
        self.y_sub, self.y_add = None, None
        
        if noise is None:
            self._add_noise()
        else:
            self.y_sub, self.y_add = self.y - noise/self.calpha, self.y + self.calpha*noise
        
        gradU = lambda t, y: self.f_sub.grad(t, y) + self.prior.grad(t, lam_reg)
        self.sampler = sampler(gradU, gamma, X_post, proj=proj, **sampler_kwargs)
        if sampler == GaussianDiag:   # if x follows a diagonal Gaussian prior
            self.factor = lambda t: self.alpha*t/(self.f.sigma**2+self.alpha*kwargs["sigmax"]**2) 
        else:
            self.factor = lambda t: t

    def _add_noise(self):
        noise = torch.randn_like(self.y, device=device)*self.f.sigma
        self.y_sub, self.y_add = self.y - noise/self.calpha, self.y + noise*self.calpha 

    def compute_test(self, nb_steps, log_stats=False, burnin_ratio=0.25, thinning=1, normalize=False, log_wu=False):
        
        it_burnin = int(burnin_ratio*nb_steps) if burnin_ratio < 1 else burnin_ratio
        n_rem = nb_steps - it_burnin if burnin_ratio < 1 else nb_steps
        
        n_rem = n_rem // self.batch_size
        lik_trace = torch.zeros((self.batch_size, n_rem), device=device)
        axis = tuple(range(1, self.sampler.X.dim()))

        if log_stats:
            post_hist = torch.zeros(n_rem, device=device)
            X_post_trace = torch.zeros([n_rem, self.dimx], device=device)
        if log_wu:
            wu_trace = torch.zeros((it_burnin, self.batch_size) + self.sampler.X.shape[1:], device=device)
        
        with torch.no_grad():
            for n in tqdm.tqdm(range(it_burnin)):  # warmup stage
                self.sampler(self.y_sub)
                if log_wu:
                    wu_trace[n] = self.sampler.X.clone()

        lik1_mean = torch.tensor(0., device=device)

        trange = tqdm.tqdm(range(n_rem), mininterval=1)
        with torch.no_grad():
            for n in trange:
                for _ in range(thinning):
                    self.sampler(self.factor(self.y_sub))
                lik1 = - self.f_add(self.sampler.X, self.y_add, dim=axis)
                lik_trace[:, n] = lik1             
                if log_stats:
                    X_post_trace[n] =  torch.flatten(self.sampler.X).detach()
                    post_hist[n] = lik1 + self.prior(self.sampler.X)             

        n_rem = torch.tensor(n_rem, device=device)
        lik1_mean = torch.logsumexp(lik_trace, (0, 1)) - torch.log(n_rem*self.batch_size)
        if normalize:
            lik1_mean = lik1_mean - 0.5 * self.dimx * torch.log(torch.tensor(2*torch.pi, device=device)) - self.dimx * torch.log(self.f_add.sigma) 

        res = lik_trace.cpu().reshape(-1), lik1_mean.item() 
        if log_stats:
            res =  (X_post_trace.cpu(), post_hist.cpu()) + res 
        if log_wu:
            res = (wu_trace.cpu(),) + res
            
        return res

    def compute_test2(self, nb_steps, log_stats=False, burnin_ratio=0.25, thinning=10, tol=1e-4, patience=10, normalize=False):
        lik1_mean = torch.tensor(0., device=device)

        it_burnin = int(burnin_ratio*nb_steps)
        n_rem = nb_steps - it_burnin
        
        if log_stats:
            post_hist = torch.zeros(n_rem, device=device)
            X_post_trace = torch.zeros([n_rem, self.dimx]).to(device)
            lik_trace = torch.zeros(n_rem, device=device)
            
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
                    self.sampler(self.factor(self.y_sub))
                
                lik1 = - self.f_add(self.sampler.X, self.y_add)
                lik_trace[n] = lik1             
                lik1_mean = torch.logsumexp(lik_trace[:n+1], 0) - torch.log(n + 1) 
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
        if normalize:
            lik1_mean = lik1_mean - 0.5 * self.dimx * torch.log(torch.tensor(2*torch.pi)) - self.dimx * torch.log(self.f_add.sigma) 
            lik_trace = lik_trace - 0.5 * self.dimx * torch.log(torch.tensor(2*torch.pi)) - self.dimx * torch.log(self.f_add.sigma)
        if log_stats:
            return X_post_trace[:n+1].cpu(), post_hist[:n+1].cpu(), lik_trace[:n+1].cpu(), lik1_mean.item()
        else:
            return lik1_mean.item()

def compute_test3(self, nb_steps, log_stats=False, burnin_ratio=0.25, inner_it=1000, normalize=False):
            lik1_mean = torch.tensor(0., device=device)
    
            it_burnin = int(burnin_ratio*nb_steps)
            n_rem = nb_steps - it_burnin
            ntot = inner_it*n_rem
            lik_trace = torch.zeros(ntot, device=device)
    
            if log_stats:
                post_hist = torch.zeros(ntot, device=device)
                X_post_trace = torch.zeros([ntot, self.dimx]).to(device)
                
            self._add_noise()
            for _ in tqdm.tqdm(range(it_burnin)):  # warmup stage
                with torch.no_grad():
                    self.sampler(self.y_sub)
    
            trange = tqdm.tqdm(range(n_rem))
            n = -1
            for _ in trange:
                trange.set_description("t={:.4f}".format(lik1_mean.item()))
                with torch.no_grad():
                    for _ in range(inner_it):
                        n += 1
                        self.sampler(self.factor(self.y_sub))
                        lik1 = - self.f_add(self.sampler.X, self.y_add)
                        lik_trace[n] = lik1             
                        lik1_mean = torch.logsumexp(lik_trace[:n+1], 0) - torch.log(n + 1) 
                        if log_stats:
                            X_post_trace[n] =  torch.flatten(self.sampler.X).detach()
                            post_hist[n] = lik1 + self.prior(self.sampler.X)             
                            lik_trace[n] = lik1    

                    self._add_noise()  # regenerate additional noise
      
            if normalize:
                lik1_mean = lik1_mean - 0.5 * self.dimx * torch.log(torch.tensor(2*torch.pi)) - self.dimx * torch.log(self.f_add.sigma) 
                lik_trace = lik_trace - 0.5 * self.dimx * torch.log(torch.tensor(2*torch.pi)) - self.dimx * torch.log(self.f_add.sigma)
            if log_stats:
                return X_post_trace[:n+1].cpu(), post_hist[:n+1].cpu(), lik_trace[:n+1].cpu(), lik1_mean.item()
            else:
                return lik1_mean.item()
                
def compute_test3(self, nb_steps, log_stats=False, burnin_ratio=0.25, inner_it=1000, normalize=False):
    lik1_mean = torch.tensor(0., device=device)

    it_burnin = int(burnin_ratio*nb_steps)
    n_rem = nb_steps - it_burnin
    ntot = inner_it*n_rem
    lik_trace = torch.zeros(ntot, device=device)

    if log_stats:
        post_hist = torch.zeros(ntot, device=device)
        X_post_trace = torch.zeros([ntot, self.dimx]).to(device)
        
    self._add_noise()
    for _ in tqdm.tqdm(range(it_burnin)):  # warmup stage
        with torch.no_grad():
            self.sampler(self.y_sub)

    trange = tqdm.tqdm(range(n_rem))
    n = -1
    for _ in trange:
        trange.set_description("t={:.4f}".format(lik1_mean.item()))
        with torch.no_grad():
            for _ in range(inner_it):
                n += 1
                self.sampler(self.factor(self.y_sub))
                lik1 = - self.f_add(self.sampler.X, self.y_add)
                lik_trace[n] = lik1             
                lik1_mean = torch.logsumexp(lik_trace[:n+1], 0) - torch.log(n + 1) 
                if log_stats:
                    X_post_trace[n] =  torch.flatten(self.sampler.X).detach()
                    post_hist[n] = lik1 + self.prior(self.sampler.X)             
                    lik_trace[n] = lik1    

            self._add_noise()  # regenerate additional noise

    if normalize:
        lik1_mean = lik1_mean - 0.5 * self.dimx * torch.log(torch.tensor(2*torch.pi)) - self.dimx * torch.log(self.f_add.sigma) 
        lik_trace = lik_trace - 0.5 * self.dimx * torch.log(torch.tensor(2*torch.pi)) - self.dimx * torch.log(self.f_add.sigma)
    if log_stats:
        return X_post_trace[:n+1].cpu(), post_hist[:n+1].cpu(), lik_trace[:n+1].cpu(), lik1_mean.item()
    else:
        return lik1_mean.item()