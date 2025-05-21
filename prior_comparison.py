import torch 
from priors import L2
import tqdm
from utils import device
from sampling import ULA, GaussianDiag, DiffPIR


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
        if sampler == GaussianDiag:   # if x follows a diagonal Gaussian prior
            self.factor = lambda t: self.alpha*t/(self.f.sigma**2+self.alpha*kwargs["sigmax"]**2) 
        else:
            self.factor = lambda t: t

        if sampler == DiffPIR:
            sampler_kwargs['y'] = self.y_sub
            sampler_kwargs['physics'].noise_model.update_parameters(sigma / torch.sqrt(self.alpha))
        self.sampler = sampler(gradU, gamma, X_post, proj=proj, **sampler_kwargs)


    def _add_noise(self):
        noise = torch.randn_like(self.y, device=device)*self.f.sigma
        self.y_sub, self.y_add = self.y - noise/self.calpha, self.y + noise*self.calpha 

    def _get_nit(self, nb_steps, burnin_ratio):
        it_burnin = int(burnin_ratio*nb_steps) if burnin_ratio < 1 else burnin_ratio
        n_rem = nb_steps - it_burnin if burnin_ratio < 1 else nb_steps
        return it_burnin, n_rem

    def compute_test(self, nb_steps, log_stats=False, burnin_ratio=0.25, thinning=1, normalize=False, log_wu=False, 
                     logsum=True):
        """Compute log p(y+/y-) or E_x/y-(log p(y+/x)) using MC for a fixed iteration of noise."""
        it_burnin, n_rem = self._get_nit(nb_steps, burnin_ratio)
        
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
        if logsum:
            lik1_mean = torch.logsumexp(lik_trace, (0, 1)) - torch.log(n_rem*self.batch_size)
        else:
            lik1_mean = torch.mean(lik_trace, (0, 1))

        if normalize:
            lik1_mean = lik1_mean - 0.5 * self.dimx * torch.log(torch.tensor(2*torch.pi, device=device)) - self.dimx * torch.log(self.f_add.sigma) 
            lik_trace = lik_trace - 0.5 * self.dimx * torch.log(torch.tensor(2*torch.pi, device=device)) - self.dimx * torch.log(self.f_add.sigma)
        res = lik_trace.cpu().reshape(-1), lik1_mean.item() 
        if log_stats:
            res =  (X_post_trace.cpu(), post_hist.cpu()) + res 
        if log_wu:
            res = (wu_trace.cpu(),) + res
            
        return res
    

    def compute_test2(self, nb_steps, nb_noise, burnin_ratio=0.25, thinning=10, thinning_noise=10, normalize=False, logsum=True):
        """Compute log E_eps(p(y+/y-)) or E_eps,x/y-(log p(y+/x)) using MC (average over noise and x/y-)."""
        it_burnin, n_rem = self._get_nit(nb_steps, burnin_ratio)
        
        n_rem = n_rem // self.batch_size
        lik_trace = torch.zeros((nb_noise, self.batch_size, n_rem), device=device)
        axis = tuple(range(1, self.sampler.X.dim()))

        with torch.no_grad():
            for n in tqdm.tqdm(range(it_burnin)):  # warmup stage
                self.sampler(self.y_sub)

        with torch.no_grad():
            for t in range(nb_noise):
                for _ in range(thinning_noise):
                    self.sampler(self.y_sub)

                trange = tqdm.tqdm(range(n_rem), mininterval=1)
                for n in trange:
                    for _ in range(thinning):
                        self.sampler(self.factor(self.y_sub))
                    lik1 = - self.f_add(self.sampler.X, self.y_add, dim=axis)
                    lik_trace[t, :, n] = lik1                   

                self._add_noise()
  
        n_rem = torch.tensor(n_rem*nb_noise, device=device)
        if logsum:
            lik1_mean = torch.logsumexp(lik_trace, (0, 1)) - torch.log(n_rem*self.batch_size)
        else:
            lik1_mean = torch.mean(lik_trace, (0, 1))
        if normalize:
            lik1_mean = lik1_mean - 0.5 * self.dimx * torch.log(torch.tensor(2*torch.pi, device=device)) - self.dimx * torch.log(self.f_add.sigma) 
            lik_trace = lik_trace - 0.5 * self.dimx * torch.log(torch.tensor(2*torch.pi, device=device)) - self.dimx * torch.log(self.f_add.sigma)

        res = lik_trace.cpu().reshape(-1), lik1_mean.item() 
            
        return res

