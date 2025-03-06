import torch 
import tqdm
from utils import device


class SAPG:
    def __init__(self, y, g, f, gamma, gammap, lam_reg, X_init=None, project='clamp'):
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
            self.proj = lambda t: torch.clamp(t, 0., 1.)
        elif project == 'refl':
            self.proj = lambda t: torch.abs(t)
        else:
            self.proj = lambda t: t
        
        if X_init is None:
            self.X_prior, self.X_post = torch.randn_like(y).to(device), torch.randn_like(y).to(device)
        else: self.X_prior, self.X_post = self.proj(X_init.clone()), self.proj(X_init.clone())

        self.X_prior_warm, self.X_post_warm = None, None

    def _ULA_it_post(self):
        with torch.no_grad():
            self.X_post = self.proj(self.X_post - self.gamma * self.f.grad(self.X_post, self.y) - 
                                      self.gamma * self.g.grad(self.X_post, self.lam_reg) + 
                                      torch.sqrt(2*self.gamma) * torch.randn_like(self.X_post))

    def _ULA_it_prior(self):
            self.X_prior = self.proj(self.X_prior - self.gammap * self.g.grad(self.X_prior, self.lamp_reg) + 
                                       torch.sqrt(2*self.gammap) * torch.randn_like(self.X_prior))
            
    def sample_post(self, nb_it, X_init=None):
        if X_init is not None:
            X_post = X_init.clone()
        else:
            X_post = self.X_post.clone()

        hist, post_hist = torch.zeros((nb_it,) + X_post.shape, device=device), torch.zeros(nb_it, device=device)
        for n in tqdm.tqdm(range(nb_it)):
            with torch.no_grad():
                X_post = self.proj(X_post - self.gamma * self.f.grad(X_post, self.y) - 
                          self.gamma * self.g.grad(X_post, self.lam_reg) + torch.sqrt(2*self.gamma) * torch.randn_like(X_post))
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
                self._ULA_it_post()

                if warm_up_prior:
                    for _ in range(thinning_prior):
                        self._ULA_it_prior()
                if log_stats and n >= it_burnin:
                    X_post_trace[count_stats] =  torch.flatten(self.X_post).detach()
                    post_hist[count_stats] = self.post(self.X_post, self.y).detach()
                    
                    if warm_up_prior:
                        X_prior_trace[count_stats] = torch.flatten(self.X_prior).detach()
                        prior_hist[count_stats] = - self.g(self.X_prior).detach()
                        
                    count_stats += 1

        self.X_post_warm = self.X_post.clone()
        if warm_up_prior:
            self.X_prior_warm = self.X_prior.clone()
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
        
    def run(self, delta, nb_steps, bounds, init_param=None, thinning_global=1,
            burnin_ratio=0.8, thinning_post=10, thinning_prior=10, tol=1e-4, alpha=None,
            reuse_post=False):
        """
        delta: fun that updates gradient stepsize
        """
        if ((self.X_prior_warm is None) and (alpha is None) and (not reuse_post)) or (self.X_post_warm is None):
            print("Warm up MC first")
            return 
        else:
            self.X_posterior = self.X_post_warm.clone()
            if alpha is None and not reuse_post:
                self.X_prior = self.X_prior_warm.clone()

        if init_param is not None:
            self.update_param(init_param)
        else:
            self.update_param(self.g.param)  # to ensure eta is up to date
        log_bounds = torch.log(torch.tensor(bounds)).to(device)
        
        g = self.g
        nit_param = 0  # num of param updates - 1
        nit_mean = 0  # num of mean estimator updates - 1

        prior_hist = torch.zeros(nb_steps // thinning_global, device=device)
        post_hist = torch.zeros(nb_steps // thinning_global, device=device)
        mean_hist =  torch.zeros(nb_steps // thinning_global, device=device)
        param_hist = torch.zeros(nb_steps // thinning_global, device=device)
        
        it_burnin = int(burnin_ratio*nb_steps)  # first iteration for updating mean
        
        trange = tqdm.tqdm(range(1, nb_steps + 1))
        for n in trange:
            trange.set_description("theta={:.4f}, eta={:.4f}".format(g.param.item(), self.eta.item()))

            with torch.no_grad():

                for _ in range(thinning_post):
                    self._ULA_it_post()
                if alpha is None:
                    if reuse_post:
                        self.X_prior = self.X_post.clone()
                    for _ in range(thinning_prior):
                        self._ULA_it_prior()
                
                if n % thinning_global == 0:  # update parameters
                    step = delta(n)
                    grad_param_post = g.grad_param(self.X_post)
                    if alpha is None:
                        grad_param_prior = g.grad_param(self.X_prior)
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
                        new_mean = (nit_mean*mean_hist[nit_mean-1] + g.param.item()) / (nit_mean + 1)
                        mean_hist[nit_mean] = new_mean
                        nit_mean += 1 

                        if abs(mean_hist[nit_mean - 1] - mean_hist[nit_mean - 2]) < tol:  # stop if mean estimator has converged
                            break

                    elif n >= it_burnin:
                        nit_mean = 1
                        mean_hist[0] = g.param.item()

        return (g.param.item(), param_hist[:nit_param].cpu(), mean_hist[:nit_mean].cpu(), 
                post_hist[:nit_param].cpu(), prior_hist[:nit_param].cpu())
