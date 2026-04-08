from priors import DiagonalWeightedTikhonovPrior
from experiments_utils import generate_measurements_gaussian_diag, compute_test_gaussian_diag
import torch
import numpy as np
from utils import device
from prior_comparison import DegradedLikelihood
from sampling import SKROCK, GaussianDiag, ULA
fig_folder = 'figs_tests'
import sys
import os
from tqdm import tqdm
test_case = int(sys.argv[1])


########################################################################

# Tests on synthetic problems : $y = x + n$, $x\sim \mathcal N(0,\sigma_x^2I_d)$, $n\sim \mathcal N(0, \sigma^2 I_d)$ or $y=Ax+n$, $x\sim \mathcal L(0, \sigma_x^2)$.

########

# 1. Convergence speed 

if test_case == - 1:   # test that skrock outputs the correct posterior distribution
    torch.manual_seed(0)
    np.random.seed(0)
    batch_size = 2000
    nb_samples, nb_warmup = 10000, 10000
    sigmax, sigma = 1., 0.05
    alpha = torch.tensor(0.75)
    d = 1
    y, x, p = generate_measurements_gaussian_diag(d, sigmax, sigma)
    noise = torch.randn([1, 1, d], device=device)*sigma
    L_f = 1 / (sigma**2 / (alpha) )
    L_g = 1 / sigmax**2
    gamma = 0.98*1/(L_f + L_g)
    g = DiagonalWeightedTikhonovPrior(torch.tensor(1., device=device), torch.ones(d, device=device)/sigmax**2)
    dl = DegradedLikelihood(y, g, p, sigma, gamma, X_init=p.A_A_adjoint(y).to(device).clone(), noise=noise,
                            sampler=SKROCK, sampler_kwargs={'s':15}, lam_reg=None, project=None, alpha=alpha)
    dl_ula = DegradedLikelihood(y, g, p, sigma, gamma, X_init=p.A_A_adjoint(y).to(device).clone(), noise=noise,
                            sampler=ULA, lam_reg=None, project=None, alpha=alpha)
    dl_analytical = DegradedLikelihood(y, g, p, sigma, gamma, X_init=p.A_A_adjoint(y).to(device).clone(), noise=noise,
                            sampler=GaussianDiag, 
                            sampler_kwargs={'d':torch.tensor(d).to(device), 
                                            'sigma':torch.tensor(sigma*sigmax / 
                                                                 np.sqrt(sigma**2+alpha*sigmax**2)).to(device)},
                            lam_reg=None, project=None, alpha=alpha, sigmax=sigmax)
    trace_skrock, trace_analytical, trace_ula = [], [], []
    with torch.no_grad():
        for i in tqdm(range(nb_warmup)):
            dl.sampler(dl_analytical.y_sub)
            dl_ula.sampler(dl_analytical.y_sub)
        for i in tqdm(range(nb_samples)):
            trace_skrock.append(dl.sampler(dl_analytical.y_sub))
            trace_ula.append(dl_ula.sampler(dl_analytical.y_sub))

        trace_skrock = torch.concatenate(trace_skrock, dim=0)
        trace_ula = torch.concatenate(trace_ula, dim=0)
        for i in tqdm(range(nb_samples)):
            trace_analytical.append(dl_analytical.sampler(dl_analytical.factor(dl_analytical.y_sub)))
        trace_analytical = torch.concatenate(trace_analytical, dim=0)
        trace_skrock = trace_skrock.cpu().numpy()
        trace_analytical = trace_analytical.cpu().numpy()
        np.savez(os.path.join(fig_folder, "skrock_test.npz"),
                 trace_skrock=trace_skrock,
                 trace_analytical=trace_analytical,
                 trace_ula=trace_ula)


if test_case == 0 or test_case == 1:
    # plot the error to the true value of $p(y^+/y^-)$ in function of the number of iterations for different alphas and dimensions
    # 0 : skrock for sampling post, 1: analytical post
    save_folder = os.path.join(fig_folder, "convergence_speed")
    file_name = "convergence_skrock.npz" if test_case == 0 else "convergence_analytical.npz"
    torch.manual_seed(0)
    np.random.seed(0)
    sigmax, sigma = 1., 0.05
    nval = 3    # test nval alphas, with ndim dimensions
    ntry = 25
    alphas = torch.linspace(0.25, 0.75, nval, device=device)
    dims = torch.tensor([10, 100, 1000], device=device)
    ndim = len(dims)
    nb_steps = 250000
    burnin_ratio = 100 if test_case == 0 else 0
    batch_size = 50
    evidences2 = np.zeros([nval, ndim, ntry])  # using p(y+/y-)
    evidences3 = np.zeros([nval, ndim, ntry])  # using p(y+/y-)

    lik_traces = np.zeros((nval, ndim, ntry, nb_steps))  # list of lists (nval, ndim)  

    # generate measurements
    ys, xs, ps, noises = [], [], [], []
    for i in range(ndim):
        y, x, p = generate_measurements_gaussian_diag(dims[i], sigmax, sigma)
        ys.append(y.float().to(device)), xs.append(x.float().to(device)), ps.append(p)
        noises.append(torch.randn([1, 1, dims[i], ntry], device=device)*sigma)
    for l in range(ndim):
        d = dims[l]
        print("------ d={}".format(d))
        y, x, p = ys[l], xs[l], ps[l]
        g = DiagonalWeightedTikhonovPrior(torch.tensor(1., device=device), torch.ones(d, device=device)/sigmax**2)

        for i in range(nval):
            torch.manual_seed(1)

            alpha = alphas[i]
            L_f = 1 / (sigma**2 / alpha)  # noise level for y-  
            L_g = 1 / sigmax**2
            gamma = 0.98*1/(L_f + L_g)          
            for t in range(ntry):
                print("------- t={}/{}".format(t, ntry-1))
                noise = noises[l][:, :, :, t]
                
                if test_case == 0:
                    dl = DegradedLikelihood(y, g, p, sigma, gamma, X_init=p.A_adjoint(y).to(device).clone(), 
                                            noise=noise, batch_size=batch_size,
                                            sampler=SKROCK,sampler_kwargs={'s':15}, 
                                            lam_reg=None, project=None, alpha=alpha)
                else:
                    dl = DegradedLikelihood(y, g, p, sigma, gamma, X_init=p.A_adjoint(y).to(device).clone(),
                                            sampler=GaussianDiag, noise=noise,batch_size=batch_size,
                                            sampler_kwargs={'d':torch.tensor(d).to(device), 
                                                            'sigma':torch.tensor(sigma*sigmax / 
                                                                                 np.sqrt(sigma**2 + 
                                                                                         alpha.cpu()*sigmax**2)).to(device)},
                                            lam_reg=None, project=None, alpha=alpha, sigmax=sigmax)
                lik_trace, lik_mean = dl.compute_test(nb_steps, burnin_ratio=burnin_ratio, 
                                                      log_stats=False, thinning=1, normalize=True)        
                lik_traces[i, l, t] =  - (torch.logcumsumexp(lik_trace, 0).cpu().numpy() - torch.log(
                    torch.arange(1, nb_steps+1, device=device)).cpu().numpy())

                evidences2[i, l, t] = compute_test_gaussian_diag(d, sigmax, sigma, alpha.item(), y.cpu().numpy(), 
                                                                 yp=dl.y_add.cpu().numpy(), ym=dl.y_sub.cpu().numpy(), 
                                                                 mlog=True).cpu().numpy()
                
                print("p(y+/y-) sampler: {}\n exact: {}\n lik_mean: {}".format(lik_traces[i, l, t][-1], 
                                                                               evidences2[i, l, t], lik_mean))
        np.savez(os.path.join(save_folder, file_name), 
                 lik_traces=lik_traces,  test_exact=evidences2, alphas=alphas.cpu().numpy(), dims=dims.cpu().numpy())

if test_case == 2 or test_case == 3:
    # plot the relative error in function of dimension at a fixed alpha and number of iterations
    # 2 : skrock for sampling post, 3: analytical post
    save_folder = os.path.join(fig_folder, "convergence_speed")
    file_name = "convergence_dim_skrock.npz" if test_case == 2 else "convergence_dim_analytical.npz"
    torch.manual_seed(0)
    np.random.seed(0)
    sigmax, sigma = 1., 0.05
    alpha = torch.tensor(0.5, device=device)
    ntry = 25
    ndim = 7
    dims = torch.logspace(0, 4, ndim, device=device, dtype=torch.int)
    nb_steps = 50000
    burnin_ratio = 500 if test_case == 2 else 0
    batch_size = 5
    evidences2 = np.zeros([ndim, ntry])  # using p(y+/y-)
    lik_traces = np.zeros((ndim, ntry))  # list of lists (nval, ndim)

    # generate measurements
    ys, xs, ps, noises = [], [], [], []
    for i in range(ndim):
        y, x, p = generate_measurements_gaussian_diag(dims[i], sigmax, sigma)
        ys.append(y.float().to(device)), xs.append(x.float().to(device)), ps.append(p)
        noises.append(torch.randn([1, 1, dims[i], ntry], device=device)*sigma)
    for l in range(ndim):
        d = dims[l]
        print("d={}".format(d))
        g = DiagonalWeightedTikhonovPrior(torch.tensor(1., device=device), 
                                          torch.ones(d, device=device)/sigmax**2)

        L_f = 1 / (sigma**2 / alpha)  
        L_g = 1 / sigmax**2
        gamma = 0.98*1/(L_f + L_g)
        y, x, p = ys[l], xs[l], ps[l]
        for t in range(ntry):
            print("------- t={}/{}, d={}".format(t, ntry-1, d))
            noise = noises[l][:, :, :, t]

            if test_case == 2:
                dl = DegradedLikelihood(y, g, p, sigma, gamma, X_init=p.A_adjoint(y).to(device).clone(), 
                                        noise=noise, batch_size=batch_size,
                                        sampler=SKROCK,sampler_kwargs={'s':15}, 
                                        lam_reg=None, project=None, alpha=alpha)
            else:
                dl = DegradedLikelihood(y, g, p, sigma, gamma, X_init=p.A_adjoint(y).to(device).clone(),
                                        sampler=GaussianDiag, noise=noise, 
                                        sampler_kwargs={'d':torch.tensor(d).to(device), 
                                                        'sigma':torch.tensor(sigma*sigmax / 
                                                                             np.sqrt(sigma**2 + 
                                                                                     alpha*sigmax**2)).to(device)},
                                        lam_reg=None, project=None, alpha=alpha, sigmax=sigmax)
            # use a sampler 
            lik_trace, lik_mean = dl.compute_test(nb_steps, burnin_ratio=burnin_ratio, 
                                                  log_stats=False, thinning=1, normalize=True)        
        
            lik_traces[l, t] =  - lik_mean
            evidences2[l, t] = compute_test_gaussian_diag(dims[l], sigmax, sigma, alpha.item(), y.cpu().numpy(), 
                                                          yp=dl.y_add.cpu().numpy(), ym=dl.y_sub.cpu().numpy(),
                                                            mlog=True)

            print("p(y+/y-) sampler: {}\n exact: {}".format(-lik_mean, evidences2[l, t]))
            np.savez(os.path.join(save_folder, file_name), lik_traces=lik_traces,  
                     test_exact=evidences2, dims=dims.cpu().numpy(), alpha=alpha.cpu().numpy())

########

# 2. Accuracy

if test_case == 4:
    # Model misspecification : plot p(y+/y-, sigma)/p(y+/y-, sigma_true) in function of sigma for different alpha
    save_folder = os.path.join(fig_folder, "accuracy")
    
    np.random.seed(1)
    torch.manual_seed(1)
    nsigma = 250  # test nval dimensions, with ntry noise samples
    ntry = 250
    d = 1000
    alphas = torch.tensor([0.25, 0.5, 0.75]).to(device)
    nval = len(alphas)
    evidences2 = np.zeros([nval, nsigma, ntry])  # using p(y+/y-)
    sigmax_ex, sigma = 1., 0.05
    y, x, p = generate_measurements_gaussian_diag(d, sigmax_ex, sigma)
    noises = torch.randn([ntry, 1, 1, d], device=device)*sigma
    sigmaxs = torch.linspace(0.25, 2., nsigma)
    
    exact_vals = np.zeros([nval, ntry])

    with torch.no_grad():
        for i in range(nval):
            alpha = alphas[i]
            
            for k in range(ntry):
                print("------- k={}/{}".format(k, ntry-1))
                noise = noises[k]
                # start with exact sigma
                L_f = 1 / (sigma**2 / alpha )  
                L_g = 1 / sigmax_ex**2
                gamma = torch.tensor(0.98*1/(L_f + L_g), device=device, dtype=torch.float32)
                g = DiagonalWeightedTikhonovPrior(torch.tensor(1., device=device), 
                                                  torch.ones(d, device=device)/sigmax_ex**2)
                # use SKROCK and MC
                dl = DegradedLikelihood(y, g, p, sigma, gamma, X_init=p.A_A_adjoint(y).to(device).clone(), noise=noise,
                                        sampler=SKROCK,sampler_kwargs={'s':15}, lam_reg=None, project=None, alpha=alpha)   
            
                # true sigma, exact value
                exact_vals[i, k] = compute_test_gaussian_diag(d, sigmax_ex, sigma, alpha.cpu().numpy(), y.cpu().numpy(), 
                                                              yp=dl.y_add.cpu().numpy(), ym=dl.y_sub.cpu().numpy(),
                                                              mlog=True)

                for l in range(nsigma):
                   
                    new_sigmax = sigmaxs[l]
                    L_g = 1 / new_sigmax**2
                    gamma = 0.98*1/(L_f + L_g)

                    g = DiagonalWeightedTikhonovPrior(torch.tensor(1., device=device), 
                                                      torch.ones(d, device=device)/new_sigmax**2)
                
                    dl = DegradedLikelihood(y, g, p, sigma, gamma, X_init=p.A_A_adjoint(y).to(device).clone(),
                                            noise=noise, sampler=SKROCK,sampler_kwargs={'s':15}, 
                                            lam_reg=None, project=None, alpha=alpha)
                    # compute exact p(y+/y-)
                    evidences2[i, l, k] = compute_test_gaussian_diag(d, new_sigmax.cpu().numpy(), sigma, alpha.cpu().numpy(), 
                                                                     y.cpu().numpy(), yp=dl.y_add.cpu().numpy(), 
                                                                     ym=dl.y_sub.cpu().numpy(), mlog=True)
                    
    evidences2 = evidences2 - exact_vals[:, None, :]
    evidencesmean = np.mean(evidences2, axis=-1)
    np.savez(os.path.join(save_folder, "res_exact.npz"), exact_ratio_trace = evidences2,
             exact_ratio_gaussian=evidencesmean, sigmas=sigmaxs.cpu().numpy()) 
