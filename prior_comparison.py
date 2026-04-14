import torch
from priors import L2
import tqdm
from utils import device
from deepinv.loss.metric import PSNR
from sampling import ULA, GaussianDiag, DiffPIR
import numpy as np


class DegradedLikelihood:
    def __init__(
        self,
        y,
        prior,
        physics,
        sigma,
        gamma,
        sampler=ULA,
        sampler_kwargs={},
        batch_size=1,
        X_init=None,
        lam_reg=None,
        project="clamp",
        noise=None,
        alpha=0.5,
        **kwargs
    ):
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
        self.f_add = L2(
            sigma / torch.sqrt(1 - self.alpha), physics
        )  # likelihood for Ax + N + e*calpha
        self.f_sub = L2(
            sigma / torch.sqrt(self.alpha), physics
        )  # likelihood for Ax + N + e/calpha

        self.calpha = torch.sqrt(self.alpha / (1 - self.alpha))
        self.y = y
        self.dimx = y.numel()

        if project == "clamp":
            proj = lambda t: torch.clamp(t, 0.0, 1.0)
        elif project == "refl":
            proj = lambda t: torch.abs(t)
        else:
            proj = lambda t: t

        if X_init is None:
            X_post = proj(torch.randn((batch_size,) + y.shape[1:])).to(device)
        else:
            X_post = proj(X_init.clone())
            if X_post.shape[0] != batch_size:
                X_post = X_post.repeat(
                    *[batch_size] + [1 for _ in range(X_post.dim() - 1)]
                )

        self.batch_size = batch_size

        self.y_sub, self.y_add = None, None

        self._add_noise(noise)  # generate y+, y-

        gradU = lambda t, y: self.f_sub.grad(t, y) + self.prior.grad(t, lam_reg)
        if sampler == GaussianDiag:  # if x follows a diagonal Gaussian prior
            self.factor = (
                lambda t: self.alpha
                * kwargs["sigmax"] ** 2
                * t
                / (self.f.sigma**2 + self.alpha * kwargs["sigmax"] ** 2)
            )
        else:
            self.factor = lambda t: t

        self.physics = physics
        self.sampler = sampler(gradU, gamma, X_post, proj=proj, **sampler_kwargs)

        if sampler == DiffPIR:
            print("Using DiffPIR")
            self.diff_flag = True
            self.sampler.p.noise_model.sigma = self.f.sigma / torch.sqrt(self.alpha)
        else:
            self.diff_flag = False
            lam_regp = kwargs.get("lam_regp", lam_reg)
            gradUP = lambda t, y: self.f_add.grad(t, y) + self.prior.grad(t, lam_regp)
            self.samplerp = sampler(gradUP, kwargs["gammap"], X_post, proj=proj, **sampler_kwargs)

    def _update_alpha(self, new_val):
        self.alpha = new_val
        self.calpha = torch.sqrt(self.alpha / (1 - self.alpha))
        if self.diff_flag:  # update DIFFPIR noise model
            self.sampler.p.noise_model.sigma.sigma = self.f.sigma / torch.sqrt(
                self.alpha
            )

    def _add_noise(self, noise=None):
        if noise is None:
            noise = torch.randn_like(self.y, device=device) * self.f.sigma
        self.y_sub, self.y_add = (
            self.y - noise / self.calpha,
            self.y + noise * self.calpha,
        )

    def _get_nit(self, nb_steps, burnin_ratio):
        it_burnin = int(burnin_ratio * nb_steps) if burnin_ratio < 1 else burnin_ratio
        n_rem = nb_steps - it_burnin if burnin_ratio < 1 else nb_steps
        return it_burnin, n_rem

    def compute_test(
        self,
        nb_steps,
        log_stats=False,
        burnin_ratio=0.25,
        thinning=1,
        normalize=False,
        logsum=True,
    ):
        """Compute log p(y+/y-) or E_x/y-(log p(y+/x)) using MC for a fixed iteration of noise."""
        it_burnin, n_rem = self._get_nit(nb_steps, burnin_ratio)

        n_rem = n_rem // self.batch_size
        lik_trace = torch.zeros((self.batch_size, n_rem), device=device)
        axis = tuple(range(1, self.sampler.X.dim()))

        if log_stats:
            post_hist = torch.zeros(n_rem, device=device)
            X_post_trace = torch.zeros([n_rem, self.dimx], device=device)

        with torch.no_grad():
            for n in tqdm.tqdm(range(it_burnin)):  # warmup stage
                self.sampler(self.y_sub)

        trange = tqdm.tqdm(range(n_rem), mininterval=1)
        with torch.no_grad():
            for n in trange:
                for _ in range(thinning):
                    self.sampler(self.factor(self.y_sub))
                lik1 = -self.f_add(self.sampler.X, self.y_add, dim=axis)

                lik_trace[:, n] = lik1
                if log_stats:
                    X_post_trace[n] = torch.flatten(self.sampler.X).detach()
                    post_hist[n] = lik1 + self.prior(self.sampler.X)

        n_rem = torch.tensor(n_rem, device=device)
        if logsum:
            lik1_mean = torch.logsumexp(lik_trace, (0, 1)) - torch.log(
                n_rem * self.batch_size
            )
        else:
            lik1_mean = torch.mean(lik_trace, (0, 1))

        if normalize:
            lik1_mean = (
                lik1_mean
                - 0.5 * self.dimx * torch.log(torch.tensor(2 * torch.pi, device=device))
                - self.dimx * torch.log(self.f_add.sigma)
            )
            lik_trace = (
                lik_trace
                - 0.5 * self.dimx * torch.log(torch.tensor(2 * torch.pi, device=device))
                - self.dimx * torch.log(self.f_add.sigma)
            )
        res = lik_trace.cpu().reshape(-1), lik1_mean.item()
        if log_stats:
            res = (X_post_trace.cpu(), post_hist.cpu()) + res

        return res

    def compute_test2(
        self,
        nb_steps,
        nb_noise,
        burnin_ratio=0.25,
        thinning=10,
        thinning_noise=0,
        normalize=False,
        log_post=False,
        logsum=True,
        noise_schedule=None,
        x=None,
        verbose=1,
    ):
        """Compute log E_eps(p(y+/y-)) or E_eps,x/y-(log p(y+/x)) using MC (average over noise and x/y-)."""
        it_burnin, n_rem = self._get_nit(nb_steps, burnin_ratio)

        nb_batches = nb_noise // self.batch_size
        if noise_schedule is None:
            noise_schedule = (
                torch.randn(
                    (nb_batches, self.batch_size) + self.y.shape[1:], device=device
                )
                * self.f.sigma
            )
        else:
            assert (noise_schedule.shape[0], noise_schedule.shape[1]) == (
                nb_batches,
                self.batch_size,
            ), "invalid shape {} for noise schedule".format(noise_schedule.shape)

        self._add_noise(noise_schedule[0])
        lik_trace = torch.zeros((nb_batches, self.batch_size, n_rem), device=device)

        post_mean = torch.zeros_like(self.y)
        if x is not None:
            psnr_trace = torch.zeros((nb_batches, n_rem), device=device)
            psnr = PSNR()

        if log_post:
            X_trace = torch.zeros((nb_batches, self.batch_size, n_rem, self.dimx)).cpu()
        axis = tuple(range(1, self.sampler.X.dim()))

        trange = tqdm.tqdm(range(it_burnin)) if verbose >= 1 else range(it_burnin)

        with torch.no_grad():
            for n in trange:  # warmup stage
                self.sampler(self.y_sub)

        trange = tqdm.tqdm(range(nb_batches)) if verbose >= 1 else range(nb_noise)
        trange2 = (
            tqdm.tqdm(range(n_rem), mininterval=1) if verbose >= 2 else range(n_rem)
        )
        with torch.no_grad():
            for t in trange:
                nit_global = t * self.batch_size * n_rem
                self._add_noise(noise_schedule[t])
                for _ in range(thinning_noise):
                    self.sampler(self.y_sub)

                for n in trange2:
                    for _ in range(thinning):
                        self.sampler(self.factor(self.y_sub))
                    lik1 = -self.f_add(self.sampler.X, self.y_add, dim=axis)
                    lik_trace[t, :, n] = lik1
                    post_mean_loc = torch.mean(self.sampler.X, axis=0) * self.batch_size
                    nit_loc = n * self.batch_size
                    post_mean = (post_mean * (nit_global + nit_loc) + post_mean_loc) / (
                        nit_global + nit_loc + self.batch_size
                    )
                    if log_post:
                        X_trace[t, :, n] = self.sampler.X.reshape(
                            [self.batch_size, -1]
                        ).cpu()

                    if x is not None:
                        psnr_trace[t, n] = psnr(x, post_mean)

        n_rem = torch.tensor(n_rem * nb_noise, device=device)
        if logsum:
            lik1_mean = torch.logsumexp(lik_trace, (0, 1, 2)) - torch.log(n_rem)
        else:
            lik1_mean = torch.mean(lik_trace, (0, 1, 2))
        if normalize:
            lik1_mean = (
                lik1_mean
                - 0.5 * self.dimx * torch.log(torch.tensor(2 * torch.pi, device=device))
                - self.dimx * torch.log(self.f_add.sigma)
            )
            lik_trace = (
                lik_trace
                - 0.5 * self.dimx * torch.log(torch.tensor(2 * torch.pi, device=device))
                - self.dimx * torch.log(self.f_add.sigma)
            )
        res = lik_trace.cpu().reshape((nb_noise, -1)), lik1_mean.item(), post_mean.cpu()
        if log_post:
            res = res + (X_trace.cpu(),)
        if x is not None:
            res = res + (psnr_trace.cpu(),)

        return res

    def compute_test3(
        self,
        nb_steps,
        nb_noise,
        burnin_ratio=0.25,
        thinning=10,
        thinning_noise=0,
        normalize=False,
        noise_schedule=None,
        x=None,
        verbose=1,
    ):
        """Compute log E_eps(p(y+/y-)) or E_eps,x/y-(log p(y+/x)) using MC (average over noise and x/y-)."""
        it_burnin, n_rem = self._get_nit(nb_steps, burnin_ratio)

        nb_batches = nb_noise // self.batch_size
        if noise_schedule is None:
            noise_schedule = (
                torch.randn(
                    (nb_batches, self.batch_size) + self.y.shape[1:], device=device
                )
                * self.f.sigma
            )
        else:
            assert (noise_schedule.shape[0], noise_schedule.shape[1]) == (
                nb_batches,
                self.batch_size,
            ), "invalid shape {} for noise schedule".format(noise_schedule.shape)

        self._add_noise(noise_schedule[0])
        lik_trace = torch.zeros((nb_batches, self.batch_size, n_rem), device=device)

        post_mean = torch.zeros_like(self.y)
        if x is not None:
            psnr_trace = torch.zeros((nb_batches, n_rem), device=device)
            psnr = PSNR()

        axis = tuple(range(1, self.sampler.X.dim()))

        trange = tqdm.tqdm(range(it_burnin)) if verbose >= 1 else range(it_burnin)

        with torch.no_grad():
            for n in trange:  # warmup stage
                self.sampler(self.y_sub)

        d, dims = np.prod(self.y.shape[1:]), self.y.shape[1:]
        # used for storing samples at each noise realization
        loc_trace = torch.zeros((self.batch_size, n_rem, d), device=device)  # (n, D)
        es_trace = torch.zeros(
            (nb_batches, self.batch_size, n_rem * (n_rem - 1) // 2), device=device
        )  # (nb_noise )
        es_trace2 = torch.zeros(
            (nb_batches, self.batch_size, n_rem * (n_rem - 1) // 2), device=device
        )  # (nb_noise )

        trange = tqdm.tqdm(range(nb_batches)) if verbose >= 1 else range(nb_noise)
        trange2 = (
            tqdm.tqdm(range(n_rem), mininterval=1) if verbose >= 2 else range(n_rem)
        )
        with torch.no_grad():
            for t in trange:
                nit_global = t * self.batch_size * n_rem

                self._add_noise(noise_schedule[t])
                for _ in range(thinning_noise):
                    self.sampler(self.y_sub)

                for n in trange2:
                    for _ in range(thinning):
                        self.sampler(self.factor(self.y_sub))
                    lik1 = -self.f_add(
                        self.sampler.X.to(torch.float32), self.y_add, dim=axis
                    )
                    lik_trace[t, :, n] = lik1  # - l2 error (normalized by sigma_add)

                    # update posterior mean
                    post_mean_loc = torch.mean(self.sampler.X, axis=0) * self.batch_size
                    nit_loc = n * self.batch_size
                    post_mean = (post_mean * (nit_global + nit_loc) + post_mean_loc) / (
                        nit_global + nit_loc + self.batch_size
                    )

                    # update local trace
                    loc_trace[:, n] = self.sampler.X.reshape(
                        self.batch_size, -1
                    ).clone()

                    if n > 0:  # compute diffs with respect to the previous samples
                        ind_diag = n * (n - 1) // 2  # number of diffs already computed
                        es_trace[t, :, ind_diag : ind_diag + n] = torch.cdist(
                            loc_trace[:, :n, :],  # bs, n, d
                            loc_trace[:, n].view(-1, 1, d),
                        ).view(-1, n)
                        es_trace2[t, :, ind_diag : ind_diag + n] = torch.cdist(
                            self.physics.A(
                                loc_trace[:, :n, :].reshape(
                                    (n * self.batch_size,) + dims
                                )
                            ).view(-1, n, d),
                            self.physics.A(loc_trace[:, n].view((-1,) + dims)).view(
                                -1, 1, d
                            ),
                        ).view(-1, n)

                    if x is not None:
                        psnr_trace[t, n] = psnr(x, post_mean)

        if normalize:
            lik_trace = (
                lik_trace
                - 0.5 * self.dimx * torch.log(torch.tensor(2 * torch.pi, device=device))
                - self.dimx * torch.log(self.f_add.sigma)
            )
            es_trace = (
                es_trace / self.f_add.sigma
                - 0.5 * self.dimx * torch.log(torch.tensor(2 * torch.pi, device=device))
                - self.dimx * torch.log(self.f_add.sigma)
            )
            es_trace2 = (
                es_trace2 / self.f_add.sigma
                - 0.5 * self.dimx * torch.log(torch.tensor(2 * torch.pi, device=device))
                - self.dimx * torch.log(self.f_add.sigma)
            )
        else:
            lik_trace = -self.f_add.sigma * lik_trace

        res = (
            lik_trace.cpu().reshape((nb_noise, -1)),
            es_trace.cpu().reshape((nb_noise, -1)),
            es_trace2.cpu().reshape((nb_noise, -1)),
            post_mean.cpu(),
            loc_trace.cpu(),
        )

        if x is not None:
            res = res + (psnr_trace.cpu(),)

        return res

    def save_samples(
        self,
        nb_steps,
        nb_noise,
        burnin_ratio=0.25,
        thinning=10,
        thinning_noise=0,
        normalize=False,
        noise_schedule=None,
        compute_xp=False,
        verbose=1,
    ):
        """Just save the posterior samples and the y+ at each iteration"""
        it_burnin, n_rem = self._get_nit(nb_steps, burnin_ratio)

        nb_batches = nb_noise // self.batch_size
        if noise_schedule is None:
            noise_schedule = (
                torch.randn(
                    (nb_batches, self.batch_size) + self.y.shape[1:], device=device
                )
                * self.f.sigma
            )
        else:
            assert (noise_schedule.shape[0], noise_schedule.shape[1]) == (
                nb_batches,
                self.batch_size,
            ), "invalid shape {} for noise schedule".format(noise_schedule.shape)

        self._add_noise(noise_schedule[0])
        samples_x = torch.zeros(
            (nb_batches, self.batch_size, n_rem) + self.y.shape[1:], device=device
        )
        if compute_xp:  # samples of x | y+
            samples_xp = torch.zeros(
                (nb_batches, self.batch_size) + self.y.shape[1:], device=device
            )
        yp_trace = torch.zeros(
            (nb_batches, self.batch_size) + self.y.shape[1:], device=device
        )
        ym_trace = torch.zeros(
            (nb_batches, self.batch_size) + self.y.shape[1:], device=device
        )

        post_mean = torch.zeros_like(self.y)

        trange = tqdm.tqdm(range(it_burnin)) if verbose >= 1 else range(it_burnin)

        with torch.no_grad():
            for n in trange:  # warmup stage
                self.sampler(self.y_sub)

        trange = tqdm.tqdm(range(nb_batches)) if verbose >= 1 else range(nb_batches)
        trange2 = (
            tqdm.tqdm(range(n_rem), mininterval=1) if verbose >= 2 else range(n_rem)
        )
        with torch.no_grad():
            for t in trange:
                self._add_noise(noise_schedule[t])
                for _ in range(thinning_noise):
                    self.sampler(self.y_sub)
                yp_trace[t] = self.y_add.clone()  # save y+
                ym_trace[t] = self.y_sub.clone()  # save y-

                for n in trange2:
                    for _ in range(thinning):
                        self.sampler(self.factor(self.y_sub))
                    samples_x[t, :, n] = self.sampler.X.view(
                        (self.batch_size,) + self.y.shape[1:]
                    ).clone()
                if compute_xp:
                    if self.diff_flag:  # update to y+ sigma for sampling
                        self.sampler.p.noise_model.sigma = self.f.sigma / torch.sqrt(
                            1 - self.alpha
                        )
                        self.sampler.model.sigma = self.f.sigma / torch.sqrt(
                            1 - self.alpha
                        )
                        samples_xp[t] = self.sampler.X.view(
                            (self.batch_size,) + self.y.shape[1:]
                        ).clone()

                        self.sampler.p.noise_model.sigma = self.f.sigma / torch.sqrt(
                            self.alpha
                        )
                        self.sampler.model.sigma = self.f.sigma / torch.sqrt(self.alpha)
                    else:
                        self.samplerp.X = self.sampler.X.clone()

                        for _ in range(thinning):
                            self.samplerp(self.factor(self.y_add))
                            
                        samples_xp[t] = self.samplerp.X.view(
                            (self.batch_size,) + self.y.shape[1:]
                        ).clone()


        samples_x = samples_x.reshape((-1, n_rem) + samples_x.shape[-3:])
        ym_trace = ym_trace.reshape((-1,) + self.y.shape[1:])
        yp_trace = yp_trace.reshape((-1,) + self.y.shape[1:])

        res = samples_x.cpu(), ym_trace.cpu(), yp_trace.cpu()
        if compute_xp:
            samples_xp = samples_xp.reshape((-1,) + samples_xp.shape[-3:])

            res = res + (samples_xp.cpu(),)

        return res

    def save_samples_alpha_diff(
        self,
        nb_steps,
        nb_noise,
        alpha_schedule,
        burnin_ratio=0.25,
        thinning=10,
        thinning_noise=0,
        normalize=False,
        noise_schedule=None,
        verbose=1,
    ):
        """Just save the posterior samples and the y+ at each iteration"""
        it_burnin, n_rem = self._get_nit(nb_steps, burnin_ratio)
        nb_alphas = len(alpha_schedule)

        nb_batches = nb_noise // self.batch_size
        if noise_schedule is None:
            noise_schedule = (
                torch.randn(
                    (nb_batches, self.batch_size) + self.y.shape[1:], device=device
                )
                * self.f.sigma
            )
        else:
            assert (
                noise_schedule.shape[0],
                noise_schedule.shape[1],
                noise_schedule.shape[2],
            ) == (
                nb_alphas,
                nb_batches,
                self.batch_size,
            ), "invalid shape {} for noise schedule".format(
                noise_schedule.shape
            )

        self._update_alpha(alpha_schedule[0])
        self._add_noise(noise_schedule[0, 0])
        samples_x = torch.zeros(
            (nb_alphas, nb_batches, self.batch_size, n_rem) + self.y.shape[1:],
            device=device,
        )
        yp_trace = torch.zeros(
            (nb_alphas, nb_batches, self.batch_size) + self.y.shape[1:], device=device
        )
        ym_trace = torch.zeros(
            (nb_alphas, nb_batches, self.batch_size) + self.y.shape[1:], device=device
        )

        post_mean = torch.zeros_like(self.y)

        trange = tqdm.tqdm(range(it_burnin)) if verbose >= 1 else range(it_burnin)

        with torch.no_grad():
            for n in trange:  # warmup stage
                self.sampler(self.y_sub)

        trangea = tqdm.tqdm(range(nb_alphas)) if verbose >= 1 else range(nb_alphas)
        trange = tqdm.tqdm(range(nb_batches)) if verbose >= 2 else range(nb_batches)
        trange2 = (
            tqdm.tqdm(range(n_rem), mininterval=1) if verbose >= 2 else range(n_rem)
        )
        with torch.no_grad():
            for a in trangea:
                self._update_alpha(alpha_schedule[a])
                for t in trange:
                    self._add_noise(noise_schedule[a, t])
                    for _ in range(thinning_noise):
                        self.sampler(self.y_sub)
                    yp_trace[a, t] = self.y_add.clone()  # save y+
                    ym_trace[a, t] = self.y_sub.clone()  # save y+

                    for n in trange2:
                        for _ in range(thinning):
                            self.sampler(self.factor(self.y_sub))

                        samples_x[a, t, :, n] = self.sampler.X.view(
                            (self.batch_size,) + self.y.shape[1:]
                        ).clone()

        return samples_x.cpu(), ym_trace.cpu(), yp_trace.cpu()
