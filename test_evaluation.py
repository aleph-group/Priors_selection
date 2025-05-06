from priors import DiagonalWeightedTikhonovPrior, L1Prior
from experiments_utils import generate_measurements_gaussian_diag, compute_evidence_gaussian_diag, compute_test_gaussian_diag, generate_measurements_laplace
import torch
import numpy as np
from utils import device
from prior_comparison import DegradedLikelihood
from sampling import SKROCK, GaussianDiag
fig_folder = 'figs_tests'
import sys
import os

test_case = int(sys.argv[1])

########################################################################

# I. Tests on synthetic problems : $y = x + n$, $x\sim \mathcal N(0,\sigma_x^2I_d)$, $n\sim \mathcal N(0, \sigma^2 I_d)$ or $y=Ax+n$, $x\sim \mathcal L(0, \sigma_x^2)$.

########

# 1. Convergence speed 

if test_case == 0 or test_case == 1:
    # plot the error to the true value of $p(y^+/y^-)$ in function of the number of iterations for different alphas and dimensions
    # 0 : skrock for sampling post, 1: analytical post
    save_folder = os.path.join(fig_folder, "convergence_speed")
    file_name = "skrock_trace.npy" if test_case == 0 else "analytical_post.npy"
    torch.manual_seed(0)
    np.random.seed(0)
    sigmax, sigma = 1., 0.05
    nval = 3    # test nval alphas, with ndim dimensions
    ntry = 50
    alphas = torch.linspace(0.25, 0.75, nval, device=device)
    dims = torch.tensor([10, 100, 1000], device=device)
    ndim = len(dims)
    nb_steps = 500000
    burnin_ratio = 0.002 if test_case == 0 else 0
    nb_steps_sampler = int(nb_steps/(1-burnin_ratio)) if test_case == 0 else nb_steps
    evidences2 = np.zeros([nval, ndim, ntry])  # using p(y+/y-)
    lik_traces = np.zeros((nval, ndim, ntry, nb_steps))  # list of lists (nval, ndim)    
    np.save(os.path.join(save_folder, "alphas.npy"), alphas.cpu().numpy())
    np.save(os.path.join(save_folder, "dims.npy"), dims.cpu().numpy())
    # generate measurements
    ys, xs, ps, noises = [], [], [], []
    for i in range(ndim):
        y, x, p = generate_measurements_gaussian_diag(dims[i], sigmax, sigma)
        ys.append(y.float().to(device)), xs.append(x.float().to(device)), ps.append(p)
        noises.append(torch.randn([1, 1, dims[i], ntry], device=device)*sigma)
    lik_traces = np.load(os.path.join(save_folder, file_name))
    for l in range(ndim):
        d = dims[l]
        print("------ d={}".format(d))
        y, x, p = ys[l], xs[l], ps[l]
        g = DiagonalWeightedTikhonovPrior(torch.tensor(1., device=device), torch.ones(d, device=device)/sigmax**2)

        for i in range(nval):
            torch.manual_seed(1)

            alpha = alphas[i]
            L_f = 1 / (sigma**2 / (1-alpha) )  
            L_g = d / sigmax**2
            gamma = 0.98*1/(L_f + L_g)          
            for t in range(ntry):
                print("------- t={}/{}".format(t, ntry-1))
                noise = noises[l][:, :, :, t]
                
                if test_case == 0:
                    dl = DegradedLikelihood(y, g, p, sigma, gamma, X_init=p.A_A_adjoint(y).to(device).clone(), noise=noise,
                                            sampler=SKROCK,sampler_kwargs={'s':15}, lam_reg=None, project=None, alpha=alpha)
                else:
                    dl = DegradedLikelihood(y, g, p, sigma, gamma, X_init=p.A_A_adjoint(y).to(device).clone(),sampler=GaussianDiag, noise=noise,
                                    sampler_kwargs={'d':torch.tensor(d).to(device), 'sigma':torch.tensor(sigma*sigmax / np.sqrt(sigma**2+alpha*sigmax**2)).to(device)},
                                    lam_reg=None, project=None, alpha=alpha, sigmax=sigmax)
                # use a sampler 
                lik_trace, lik_mean = dl.compute_test(nb_steps_sampler, burnin_ratio=burnin_ratio, log_stats=False, thinning=1, normalize=True)        
            
                lik_traces[i, l, t] =  (- torch.logcumsumexp(lik_trace, 0) + 0.5 * dl.dimx * torch.log(torch.tensor(2*torch.pi, device=device)) + dl.dimx * torch.log(dl.f_add.sigma) + torch.log(torch.arange(1, nb_steps+1, device=device))).cpu().numpy()
    
                #lik_traces[i, l, t] = lik_traces[i, l, t] + (d.item() * np.log(2*np.pi) + 2*d.item() * np.log(dl.f_add.sigma.item()) + 2*np.log(np.arange(1, nb_steps+1)))
                new_ev = compute_test_gaussian_diag(d, sigmax, sigma, alpha.item(), y.cpu().numpy(), yp=dl.y_add.cpu().numpy(), ym=dl.y_sub.cpu().numpy(), mlog=True).cpu().numpy()

                evidences2[i, l, t] = new_ev
                print("p(y+/y-) sampler: {}\n exact: {}".format(lik_traces[i, l, t][-1], new_ev), lik_traces[i, l, t][-1])

        np.save(os.path.join(save_folder, file_name), lik_traces)
        #np.save(os.path.join(save_folder, "test_exact.npy"), evidences2)

if test_case == 2 or test_case == 3:
    # plot the relative error in function of dimension at a fixed alpha and number of iterations
    # 2 : skrock for sampling post, 3: analytical post
    save_folder = os.path.join(fig_folder, "convergence_speed")
    file_name = "skrock_trace2.npy" if test_case == 2 else "analytical_post2.npy"
    torch.manual_seed(0)
    np.random.seed(0)
    sigmax, sigma = 1., 0.05
    alpha = torch.tensor(0.5, device=device)
    ntry = 50
    ndim = 7
    dims = torch.logspace(0, 4, ndim, device=device, dtype=torch.int)
    nb_steps = 50000
    burnin_ratio = 0.04 if test_case == 2 else 0
    nb_steps_sampler = int(nb_steps/(1-burnin_ratio)) if test_case == 2 else nb_steps
    evidences2 = np.zeros([ndim, ntry])  # using p(y+/y-)
    lik_traces = np.zeros((ndim, ntry))  # list of lists (nval, ndim)
    np.save(os.path.join(save_folder, "alpha2.npy"), alpha)
    np.save(os.path.join(save_folder, "dims2.npy"), dims.cpu().numpy())
    # generate measurements
    ys, xs, ps, noises = [], [], [], []
    for i in range(ndim):
        y, x, p = generate_measurements_gaussian_diag(dims[i], sigmax, sigma)
        ys.append(y.float().to(device)), xs.append(x.float().to(device)), ps.append(p)
        noises.append(torch.randn([1, 1, dims[i], ntry], device=device)*sigma)
    for l in range(ndim):
        d = dims[l]
        print("d={}".format(d))
        g = DiagonalWeightedTikhonovPrior(torch.tensor(1., device=device), torch.ones(d, device=device)/sigmax**2)

        L_f = 1 / (sigma**2 / (1-alpha) )  
        L_g = d / sigmax**2
        gamma = 0.98*1/(L_f + L_g)
        y, x, p = ys[l], xs[l], ps[l]
        for t in range(ntry):
            print("------- t={}/{}, d={}".format(t, ntry-1, d))
            noise = noises[l][:, :, :, t]

            if test_case == 2:
                dl = DegradedLikelihood(y, g, p, sigma, gamma, X_init=p.A_A_adjoint(y).to(device).clone(), noise=noise,
                                        sampler=SKROCK,sampler_kwargs={'s':15}, lam_reg=None, project=None, alpha=alpha)
            else:
                dl = DegradedLikelihood(y, g, p, sigma, gamma, X_init=p.A_A_adjoint(y).to(device).clone(),sampler=GaussianDiag, noise=noise,
                                sampler_kwargs={'d':torch.tensor(d).to(device), 'sigma':torch.tensor(sigma*sigmax / np.sqrt(sigma**2+alpha*sigmax**2)).to(device)},
                                lam_reg=None, project=None, alpha=alpha, sigmax=sigmax)
            # use a sampler 
            lik_trace, lik_mean = dl.compute_test(nb_steps_sampler, burnin_ratio=burnin_ratio, log_stats=False, thinning=1, normalize=True)        
        
            lik_traces[l, t] =  - lik_mean
            evidences2[l, t] = compute_test_gaussian_diag(dims[l], sigmax, sigma, alpha.item(), y.cpu().numpy(), yp=dl.y_add.cpu().numpy(), ym=dl.y_sub.cpu().numpy(), mlog=True)

            print("p(y+/y-) sampler: {}\n exact: {}".format(-lik_mean, evidences2[l, t]))
            np.save(os.path.join(save_folder, file_name), lik_traces)
            np.save(os.path.join(save_folder, "test_exact2.npy"), evidences2)

if test_case == 4: 
    # Model misspecification : plot p(y+/y-, sigma)/p(y+/y-, sigma_true) in function of sigma for different alpha, smooth over 50 tries
    save_folder = os.path.join(fig_folder, "accuracy")
    
    np.random.seed(0)
    torch.manual_seed(0)
    nsigma = 50  # test nval dimensions, with ntry noise samples
    ntry = 50
    d = 5000
    alphas = torch.tensor([0.25, 0.5, 0.75]).to(device)
    nval = len(alphas)
    evidences2 = np.zeros([nval, nsigma, ntry])  # using p(y+/y-)
    sigmax_ex, sigma = 1., 0.05
    y, x, p = generate_measurements_gaussian_diag(d, sigmax_ex, sigma)
    noises = torch.randn([ntry, 1, 1, d], device=device)*sigma
    sigmaxs = torch.linspace(0.25, 2., nsigma)

    nb_steps, burnin_ratio = 20000, 0.05
    nb_steps_sampler = int(nb_steps/(1-burnin_ratio))
    
    exact_vals = np.zeros([nval, ntry])
    approx_vals = np.zeros([nval, nsigma, ntry])
    approx_val_true = np.zeros([nval, ntry])
    with torch.no_grad():
        for i in range(nval):
            alpha = alphas[i]
            
            for k in range(ntry):
                print("------- k={}/{}".format(k, ntry-1))
                noise = noises[k]
                # start with exact sigma
                L_f = 1 / (sigma**2 / (1-alpha) )  
                L_g = d / sigmax_ex**2
                gamma = torch.tensor(0.98*1/(L_f + L_g), device=device, dtype=torch.float32)
                g = DiagonalWeightedTikhonovPrior(torch.tensor(1., device=device), torch.ones(d, device=device)/sigmax_ex**2)
                # use SKROCK and MC
                dl = DegradedLikelihood(y, g, p, sigma, gamma, X_init=p.A_A_adjoint(y).to(device).clone(), noise=noise,
                                        sampler=SKROCK,sampler_kwargs={'s':15}, lam_reg=None, project=None, alpha=alpha)   
                lik_trace, lik_mean = dl.compute_test(nb_steps_sampler, burnin_ratio=burnin_ratio, log_stats=False, thinning=1, normalize=True)
            
                approx_val_true[i, k] = - lik_mean

                # true sigma, exact value
                exact_vals[i, k] = compute_test_gaussian_diag(d, sigmax_ex, sigma, alpha.cpu(), y.cpu().numpy(), yp=dl.y_add.cpu().numpy(), ym=dl.y_sub.cpu().numpy(), mlog=True)

                for l in range(nsigma):
                   
                    new_sigmax = sigmaxs[l]
                    L_g = d / new_sigmax**2
                    gamma = 0.98*1/(L_f + L_g)

                    g = DiagonalWeightedTikhonovPrior(torch.tensor(1., device=device), torch.ones(d, device=device)/new_sigmax**2)
                
                    dl = DegradedLikelihood(y, g, p, sigma, gamma, X_init=p.A_A_adjoint(y).to(device).clone(), noise=noise,
                                        sampler=SKROCK,sampler_kwargs={'s':15}, lam_reg=None, project=None, alpha=alpha)
                    lik_trace, lik_mean = dl.compute_test(nb_steps_sampler, burnin_ratio=burnin_ratio, log_stats=False, thinning=1, normalize=True)
                    approx_vals[i, l, k] = - lik_mean
                    # compute exact p(y+/y-)
                    evidences2[i, l, k] = compute_test_gaussian_diag(d, new_sigmax, sigma, alpha.cpu(), y.cpu().numpy(), yp=dl.y_add.cpu().numpy(), ym=dl.y_sub.cpu().numpy(), mlog=True)
                    
    
    evidences2 = np.cumsum(evidences2, axis=-1)
    evidences2 /= np.arange(1, ntry+1)[None, None, :]
    exact_vals = np.cumsum(exact_vals, axis=-1)
    exact_vals /= np.arange(1, ntry+1)[None, :]

    approx_vals = np.cumsum(approx_vals, axis=-1)
    approx_vals /= np.arange(1, ntry+1)[None, None, :]
    approx_val_true = np.cumsum(approx_val_true, axis=-1)
    approx_val_true /= np.arange(1, ntry+1)[None, :]    

    approx_vals = approx_vals -  approx_val_true[:, None, :]  # log ratio
    evidences2 = evidences2 - exact_vals[:, None, :]

    np.save(os.path.join(save_folder,  "approx_ratio_gaussian.npy"), approx_vals)
    np.save(os.path.join(save_folder, "exact_ratio_gaussian.npy"), evidences2)
    np.save(os.path.join(save_folder, "sigmas.npy"), sigmaxs.cpu().numpy()) 
    np.save(os.path.join(save_folder, "alphas.npy"), alphas.cpu().numpy())
    np.save(os.path.join(save_folder, "exact_vals.npy"), exact_vals)
    np.save(os.path.join(save_folder, "approx_vals.npy"), approx_val_true)


if test_case == 5:
    # Same as 4 but only compute exact values (more plot points)
    save_folder = os.path.join(fig_folder, "accuracy")
    
    np.random.seed(0)
    torch.manual_seed(0)
    nsigma = 500  # test nval dimensions, with ntry noise samples
    ntry = 50
    d = 5000
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
                L_f = 1 / (sigma**2 / (1-alpha) )  
                L_g = d / sigmax_ex**2
                gamma = torch.tensor(0.98*1/(L_f + L_g), device=device, dtype=torch.float32)
                g = DiagonalWeightedTikhonovPrior(torch.tensor(1., device=device), torch.ones(d, device=device)/sigmax_ex**2)
                # use SKROCK and MC
                dl = DegradedLikelihood(y, g, p, sigma, gamma, X_init=p.A_A_adjoint(y).to(device).clone(), noise=noise,
                                        sampler=SKROCK,sampler_kwargs={'s':15}, lam_reg=None, project=None, alpha=alpha)   
            
                # true sigma, exact value
                exact_vals[i, k] = compute_test_gaussian_diag(d, sigmax_ex, sigma, alpha.cpu(), y.cpu().numpy(), yp=dl.y_add.cpu().numpy(), ym=dl.y_sub.cpu().numpy(), mlog=True)

                for l in range(nsigma):
                   
                    new_sigmax = sigmaxs[l]
                    L_g = d / new_sigmax**2
                    gamma = 0.98*1/(L_f + L_g)

                    g = DiagonalWeightedTikhonovPrior(torch.tensor(1., device=device), torch.ones(d, device=device)/new_sigmax**2)
                
                    dl = DegradedLikelihood(y, g, p, sigma, gamma, X_init=p.A_A_adjoint(y).to(device).clone(), noise=noise,
                                        sampler=SKROCK,sampler_kwargs={'s':15}, lam_reg=None, project=None, alpha=alpha)
                    # compute exact p(y+/y-)
                    evidences2[i, l, k] = compute_test_gaussian_diag(d, new_sigmax, sigma, alpha.cpu(), y.cpu().numpy(), yp=dl.y_add.cpu().numpy(), ym=dl.y_sub.cpu().numpy(), mlog=True)
                    
    evidences2 = evidences2 - exact_vals[:, None, :]
    evidences2 = np.mean(evidences2, axis=-1)
    np.save(os.path.join(save_folder, "new_exact_ratio_gaussian.npy"), evidences2)
    np.save(os.path.join(save_folder, "new_sigmas.npy"), sigmaxs.cpu().numpy()) 


if test_case == 6:
    # model misspecification : plot p(y+/y-, sigma)/p(y+/y-, sigma_true) in function of sigma for different alpha, laplace prior
    img_size = 128
    np.random.seed(5)
    torch.manual_seed(5)

    sigmax_ex, sigma = 0.25, 0.05
    y, x, p = generate_measurements_laplace(img_size, sigmax_ex, sigma)
    noise = torch.randn([1, 1, img_size, img_size], device=device)*sigma

    nsigmas = 50
    sigmaxs = torch.linspace(0.05, 30., nsigmas)

    alphas = torch.tensor([0.25, 0.5, 0.75]).to(device)
    nval = len(alphas)
    
    nb_steps, burnin_ratio = 50000, 0.04
    nb_steps_sampler = int(nb_steps/(1-burnin_ratio))

    approx_vals = np.zeros([nval, nsigmas])
    axpprox_vals_gt = np.zeros([nval])
    print(y.shape, x.shape)
    for i in range(nval):
        alpha = alphas[i]
        for l in range(nsigmas):
            new_sigmax = sigmaxs[l]
            L_f = 1 / (sigma**2 / (1-alpha) )  
            lam_reg = min(1/L_f, 2.)   
            L_g = 1/lam_reg
            gamma = 0.98*1/(L_f + L_g)

            g = L1Prior(torch.tensor(new_sigmax, device=device))
        
            dl = DegradedLikelihood(y, g, p, sigma, gamma, X_init=p.A_A_adjoint(y).to(device).clone(), noise=noise,
                                    sampler=SKROCK,sampler_kwargs={'s':15}, lam_reg=lam_reg, project=None, alpha=alpha)
            # compute exact p(y+/y-)
            _, lmean = dl.compute_test(nb_steps_sampler, burnin_ratio=burnin_ratio, log_stats=False, thinning=1, normalize=True)
            approx_vals[i, l] = - lmean
        g = L1Prior(torch.tensor(sigmax_ex, device=device))
        dl = DegradedLikelihood(y, g, p, sigma, gamma, X_init=p.A_A_adjoint(y).to(device).clone(), noise=noise,
                                sampler=SKROCK,sampler_kwargs={'s':15}, lam_reg=lam_reg, project=None, alpha=alpha)
        _, lmean = dl.compute_test(nb_steps_sampler, burnin_ratio=burnin_ratio, log_stats=False, thinning=1, normalize=True)
        axpprox_vals_gt[i] = - lmean

    np.save(os.path.join(save_folder, "approx_ratio_laplace.npy"), approx_vals)
    np.save(os.path.join(save_folder, "sigmas_laplace.npy"), sigmaxs.cpu().numpy())
    np.save(os.path.join(save_folder, "alphas_laplace.npy"), alphas.cpu().numpy())
    np.save(os.path.join(save_folder, "gt_vals_laplace.npy"), axpprox_vals_gt)