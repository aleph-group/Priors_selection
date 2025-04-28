from priors import DiagonalWeightedTikhonovPrior
from experiments_utils import generate_measurements_gaussian_diag, compute_evidence_gaussian_diag, compute_test_gaussian_diag
import torch
import numpy as np
from utils import device
from prior_comparison import DegradedLikelihood
from sampling import SKROCK
fig_folder = 'figs_tests'
import sys
import os

test_case = int(sys.argv[1])

########################################################################

# I. Tests on synthetic problems : $y = x + n$, $x\sim \mathcal N(0,\sigma_x^2I_d)$, $n\sim \mathcal N(0, \sigma^2 I_d)$ or $y=Ax+n$, $x\sim \mathcal L(0, \sigma_x^2)$.

########

#1. Convergence speed 

if test_case == 0 or test_case == 1:
    # plot the error to the true value of $p(y^+/y^-)$ in function of the number of iterations for different alphas and dimensions
    # 0 : skrock for sampling post, 1: analytical post
    save_folder = os.path.join(fig_folder, "convergence_speed")
    file_name = "skrock_trace.npy" if test_case == 0 else "analytical_post.npy"
    torch.manual_seed(0)
    np.random.seed(0)
    sigmax, sigma = 1., 0.05
    nval, ndim = 5, 5  # test nval alphas, with ndim dimensions
    alphas = torch.linspace(0.1, 0.9, nval, device=device)
    dims = torch.logspace(1, 4, ndim, dtype=torch.int, device=device)
    hist = np.zeros([nval, ndim])
    hist_exact = np.zeros([nval, ndim])  # using exact p(x/y-)
    nb_steps, burnin_ratio = 300000, 0.001
    burnin_ratio = 0.001 if test_case == 0 else 0
    nb_steps_sampler = int(nb_steps/(1-burnin_ratio)) if test_case == 0 else nb_steps
    evidences2 = np.zeros([nval, ndim])  # using p(y+/y-)
    lik_traces = np.zeros((nval, ndim, nb_steps))  # list of lists (nval, ndim)    
    # generate measurements
    ys, xs, ps, noises = [], [], [], []
    for i in range(ndim):
        y, x, p = generate_measurements_gaussian_diag(dims[i], sigmax, sigma)
        ys.append(y.float().to(device)), xs.append(x.float().to(device)), ps.append(p)
        noises.append(torch.randn([1, 1, dims[i]], device=device)*sigma)

    for l in range(ndim):
        d = dims[l]
        print("d={}".format(d))
        y, x, p, noise = ys[l], xs[l], ps[l], noises[l]
        g = DiagonalWeightedTikhonovPrior(torch.tensor(1., device=device), torch.ones(d, device=device)/sigmax**2)

        evidences[l] = compute_evidence_gaussian_diag(d, sigmax, sigma, y.cpu().numpy(), mlog=True)
        for i in range(nval):
            alpha = alphas[i]
            L_f = 1 / (sigma**2 / (1-alpha) )  
            L_g = d / sigmax**2
            gamma = 0.98*1/(L_f + L_g)
            torch.manual_seed(1)
            if test_case == 0:
                dl = DegradedLikelihood(y, g, p, sigma, gamma, X_init=p.A_A_adjoint(y).to(device).clone(), noise=noise,
                                        sampler=SKROCK,sampler_kwargs={'s':15}, lam_reg=None, project=None, alpha=alpha)
            else:
                dl = DegradedLikelihood(y, g, p, sigma, gamma, X_init=p.A_A_adjoint(y).to(device).clone(),sampler=GaussianDiag, noise=noise,
                                sampler_kwargs={'d':torch.tensor(d).to(device), 'sigma':torch.tensor(sigma*sigmax / np.sqrt(sigma**2+alpha*sigmax**2)).to(device)},
                                lam_reg=None, project=None, alpha=alpha, sigmax=sigmax)
            # use a sampler 
            lik_trace, lik_mean = dl.compute_test(nb_steps_sampler, burnin_ratio=burnin_ratio, log_stats=False, thinning=1, normalize=True)        

            lik_traces[i, l] = lik_trace
            hist[i, l] = -lik_mean

            evidences2[i, l] = compute_test_gaussian_diag(d, sigmax, sigma, alpha.item(), y.cpu().numpy(), yp=dl.y_add.cpu().numpy(), ym=dl.y_sub.cpu().numpy(), mlog=True)

            print("p(y+/y-) sampler: {}\n exact: {}".format(hist[i, l], evidences2[i, l]))
            np.save(os.path.join(save_folder, file_name), lik_trace)
            np.save(os.path.join(save_folder, "test_exact.npy"), evidences2)
