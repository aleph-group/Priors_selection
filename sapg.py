import torch 
import tqdm
from utils import device
from sampling import ULA, SKROCK


class SAPG:
    def __init__(self, y, g, f, gamma, gammap, lam_reg, X_init=None, project='clamp', sampler='ULA', sampler_kwargs=None):
        """
        y: observations
        f: likelihood
        g: prior
        """
        self.y = y.clone().to(device)
        self.dimx = y.numel()
        self.g, self.f = g, f  # - log prior and likelihood
        self.eta = torch.log(g.param).detach().to(device)  # e^eta = theta
        self.gamma, self.gammap = gamma.to(device), gammap.to(device)
        self.lam_reg = lam_reg.to(device)
        if lam_reg is None:  # regularization parameter for prior
            self.lamp_reg = None
        else:
            self.lamp_reg = 0.98*lam_reg

        if project == 'clamp':
            proj = lambda t: torch.clamp(t, 0., 1.)
        elif project == 'refl':
            proj = lambda t: torch.abs(t)
        else:
            proj = lambda t: t
        
        if sampler_kwargs is None:
            sampler_kwargs = {}

        if X_init is None:  # initialize the chains
            X_prior, X_post = proj(torch.randn_like(y).to(device)), proj(torch.randn_like(y).to(device))
        else: X_prior, X_post = proj(X_init), proj(X_init)

        gradU = lambda t: f.grad(t, y) + g.grad(t, lam_reg)
        gradU_prior = lambda t: g.grad(t, self.lamp_reg)
        if sampler == 'ULA':
            self.sampler = ULA(gradU, gamma, X_post, proj=proj)
            self.sampler_prior = ULA(gradU_prior, gammap, X_prior, proj=proj)
        elif sampler == 'SKROCK':
            self.sampler = SKROCK(gradU, gamma, X_post, proj=proj, **sampler_kwargs)
            self.sampler_prior = SKROCK(gradU_prior, gammap, X_prior, proj=proj, **sampler_kwargs)
        else:
            raise ValueError("Unknown sampler")

        self.X_prior_warm, self.X_post_warm = None, None
    
    def sample_post(self, nb_it):
        X_post = self.sampler.X

        hist, post_hist = torch.zeros((nb_it,) + X_post.shape, device=device), torch.zeros(nb_it, device=device)
        for n in tqdm.tqdm(range(nb_it)):
            with torch.no_grad():
                X_post = self.sampler()
                hist[n] = X_post.detach().clone()
                post_hist[n] = self.post(X_post, self.y).detach()
        return hist, post_hist
            
    def warm_up(self, nb_steps, log_stats=False, burnin_ratio=0.8, thinning_prior=20, 
                warm_up_prior=False):
        if log_stats:
            it_burnin, count_stats = int(burnin_ratio*nb_steps), 0
            n_rem = nb_steps - it_burnin
            post_hist = torch.zeros(n_rem, device=device)
            X_post_trace = torch.zeros([n_rem, self.dimx]).to(device)
            if warm_up_prior:
                X_prior_trace = torch.zeros([n_rem, self.dimx]).to(device)
                prior_hist = torch.zeros(n_rem, device=device)

        for n in tqdm.tqdm(range(nb_steps)):
            with torch.no_grad():
                X_post = self.sampler()

                if warm_up_prior:  # more steps for prior
                    for _ in range(thinning_prior):
                        X_prior = self.sampler_prior()
                if log_stats and n >= it_burnin:
                    X_post_trace[count_stats] =  torch.flatten(X_post).detach()
                    post_hist[count_stats] = self.post(X_post, self.y).detach()
                    
                    if warm_up_prior:
                        X_prior_trace[count_stats] = torch.flatten(X_prior).detach()
                        prior_hist[count_stats] = - self.g(X_prior).detach()
                        
                    count_stats += 1

        self.X_post_warm = X_post.clone()
        if warm_up_prior:
            self.X_prior_warm = X_prior.clone()
        if log_stats:
            if warm_up_prior:
                return X_post_trace.cpu(), X_prior_trace.cpu(), post_hist[:count_stats].cpu(), prior_hist.cpu()
            else:
                return X_post_trace.cpu(), post_hist[:count_stats].cpu()
        else:
            return

    def post(self, x, y):  # log posterior 
        return -self.f(x,y) - self.g(x)
        
    def update_param(self, new_param):
        self.eta = torch.log(new_param)
        self.g.update_param(new_param)
        self.sampler.gradU = lambda t: self.f.grad(t, self.y) + self.g.grad(t, self.lam_reg)
        self.sampler.gradU_prior = lambda t: self.g.grad(t, self.lamp_reg)

    def run(self, delta, nb_steps, bounds, init_param=None, thinning_global=1,
            burnin_ratio=0.8, thinning_post=10, thinning_prior=10, tol=1e-4, alpha=None,
            reuse_post=False, verbose=True):
        """
        delta: fun that updates gradient stepsize
        """
        if ((self.X_prior_warm is None) and (alpha is None) and (not reuse_post)) or (self.X_post_warm is None):
            print("Warm up MC first")
            return 
        else:  # reset sampler to post warm up state
            self.sampler.update_state(self.X_post_warm)
            if alpha is None and not reuse_post:
                self.sampler_prior.update_state(self.X_prior_warm)

        if init_param is not None:
            self.update_param(init_param)
        else:
            self.update_param(self.g.param)  # to ensure eta is up to date
        log_bounds = torch.log(torch.tensor(bounds)).to(device)
        
        g = self.g
        nit_param = 0  # num of param updates - 1
        nit_mean = 0  # num of mean estimator updates - 1
        
        if g.param.ndim == 0:  # scalar parameter
            param_hist = torch.zeros(nb_steps // thinning_global, device=device)
            prior_hist = torch.zeros(nb_steps // thinning_global, device=device)
            post_hist = torch.zeros(nb_steps // thinning_global, device=device)
            mean_hist = torch.zeros(nb_steps // thinning_global, device=device)
        else:
            param_hist = torch.zeros([nb_steps // thinning_global, len(g.param)], device=device)
            prior_hist = torch.zeros([nb_steps // thinning_global, len(g.param)], device=device)
            post_hist = torch.zeros([nb_steps // thinning_global, len(g.param)], device=device)
            mean_hist = torch.zeros([nb_steps // thinning_global, len(g.param)], device=device)
        it_burnin = int(burnin_ratio*nb_steps)  # first iteration for updating mean

        trange = range(1, nb_steps + 1)
        if verbose:
            trange = tqdm.tqdm(trange)

        for n in trange:
            if verbose:
                trange.set_description("theta={}, eta={}".format(g.param, self.eta))

            with torch.no_grad():

                for _ in range(thinning_post):
                    self.sampler()
                if alpha is None:
                    if reuse_post:  # restart prior chain from last post sample
                        self.sampler_prior.update_state(self.sampler.X)
                    for _ in range(thinning_prior):
                       self.sampler_prior()
                
                if n % thinning_global == 0:  # update parameters
                    step = delta(n)
                    grad_param_post = g.grad_param(self.sampler.X)
                    if alpha is None:
                        grad_param_prior = g.grad_param(self.sampler_prior.X)
                    else:
                        grad_param_prior = self.dimx / alpha / self.g.param
                    param_hist[nit_param] = g.param.detach().clone()
                    prior_hist[nit_param] = - grad_param_prior
                    post_hist[nit_param] = - grad_param_post
                    
                    # gradient step in logarithmic scale
                    self.eta = torch.clamp(self.eta +   # scaling for logarithmic change of variables
                                           step*self.g.param*(grad_param_prior - grad_param_post), 
                                           *log_bounds)
                    self.update_param(torch.exp(self.eta))
                    
                    nit_param += 1 
                    if nit_mean > 0:  # update mean estimator
                        new_mean = (nit_mean*mean_hist[nit_mean-1] + g.param) / (nit_mean + 1)
                        mean_hist[nit_mean] = new_mean
                        nit_mean += 1 

                        if torch.max(abs(mean_hist[nit_mean - 1] - mean_hist[nit_mean - 2])) < tol:  # stop if mean estimator has converged
                            break

                    elif n >= it_burnin:
                        nit_mean = 1
                        mean_hist[0] = g.param

        return (g.param.cpu().numpy(), param_hist[:nit_param].cpu(), mean_hist[:nit_mean].cpu(), 
                post_hist[:nit_param].cpu(), prior_hist[:nit_param].cpu())
