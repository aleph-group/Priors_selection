from priors import DiagonalWeightedTikhonovPrior, L1Prior, GSDPrior
from experiments_utils import generate_measurements_gaussian_diag, compute_evidence_gaussian_diag, compute_test_gaussian_diag, generate_measurements_laplace, generate_measurements_natural, generate_blur_operator
import torch
import numpy as np
from utils import device
from prior_comparison import DegradedLikelihood
from sampling import SKROCK, GaussianDiag, ULA
import deepinv as dinv
fig_folder = 'figs_tests'
import sys
import os
from tqdm import tqdm
test_case = int(sys.argv[1])


########################################################################

# I. Tests on synthetic problems : $y = x + n$, $x\sim \mathcal N(0,\sigma_x^2I_d)$, $n\sim \mathcal N(0, \sigma^2 I_d)$ or $y=Ax+n$, $x\sim \mathcal L(0, \sigma_x^2)$.

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
    L_g = d / sigmax**2
    gamma = 0.98*1/(L_f + L_g)
    g = DiagonalWeightedTikhonovPrior(torch.tensor(1., device=device), torch.ones(d, device=device)/sigmax**2)
    dl = DegradedLikelihood(y, g, p, sigma, gamma, X_init=p.A_A_adjoint(y).to(device).clone(), noise=noise,
                            sampler=SKROCK, sampler_kwargs={'s':15}, lam_reg=None, project=None, alpha=alpha)
    dl_ula = DegradedLikelihood(y, g, p, sigma, gamma, X_init=p.A_A_adjoint(y).to(device).clone(), noise=noise,
                            sampler=ULA, lam_reg=None, project=None, alpha=alpha)
    dl_analytical = DegradedLikelihood(y, g, p, sigma, gamma, X_init=p.A_A_adjoint(y).to(device).clone(), noise=noise,
                            sampler=GaussianDiag, sampler_kwargs={'d':torch.tensor(d).to(device), 'sigma':torch.tensor(sigma*sigmax / np.sqrt(sigma**2+alpha*sigmax**2)).to(device)},
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
        np.save(os.path.join(fig_folder, "trace_skrock_sampling.npy"), trace_skrock)
        np.save(os.path.join(fig_folder, "trace_ula_sampling.npy"), trace_ula)
        np.save(os.path.join(fig_folder, "trace_analytical_sampling.npy"), trace_analytical)
    

if test_case == 0 or test_case == 1:
    # plot the error to the true value of $p(y^+/y^-)$ in function of the number of iterations for different alphas and dimensions
    # 0 : skrock for sampling post, 1: analytical post
    save_folder = os.path.join(fig_folder, "convergence_speed")
    file_name = "skrock_trace.npy" if test_case == 0 else "analytical_post.npy"
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
    lik_traces = np.zeros((nval, ndim, ntry, nb_steps))  # list of lists (nval, ndim)  
    
    np.save(os.path.join(save_folder, "alphas.npy"), alphas.cpu().numpy())
    np.save(os.path.join(save_folder, "dims.npy"), dims.cpu().numpy())
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
            L_f = 1 / (sigma**2 / alpha)  
            L_g = d / sigmax**2
            gamma = 0.98*1/(L_f + L_g)          
            for t in range(ntry):
                print("------- t={}/{}".format(t, ntry-1))
                noise = noises[l][:, :, :, t]
                
                if test_case == 0:
                    dl = DegradedLikelihood(y, g, p, sigma, gamma, X_init=p.A_A_adjoint(y).to(device).clone(), noise=noise, batch_size=batch_size,
                                            sampler=SKROCK,sampler_kwargs={'s':15}, lam_reg=None, project=None, alpha=alpha)
                else:
                    dl = DegradedLikelihood(y, g, p, sigma, gamma, X_init=p.A_A_adjoint(y).to(device).clone(),sampler=GaussianDiag, noise=noise,batch_size=batch_size,
                                    sampler_kwargs={'d':torch.tensor(d).to(device), 'sigma':torch.tensor(sigma*sigmax / np.sqrt(sigma**2+alpha*sigmax**2)).to(device)},
                                    lam_reg=None, project=None, alpha=alpha, sigmax=sigmax)
                # use a sampler 
                lik_trace, lik_mean = dl.compute_test(nb_steps, burnin_ratio=burnin_ratio, log_stats=False, thinning=1, normalize=True)        
                lik_traces[i, l, t] =  (- torch.logcumsumexp(lik_trace, 0) + 0.5 * dl.dimx * torch.log(torch.tensor(2*torch.pi, device=device)) + dl.dimx * torch.log(dl.f_add.sigma) + torch.log(torch.arange(1, nb_steps+1, device=device))).cpu().numpy()
                new_ev = compute_test_gaussian_diag(d, sigmax, sigma, alpha.item(), y.cpu().numpy(), yp=dl.y_add.cpu().numpy(), ym=dl.y_sub.cpu().numpy(), mlog=True).cpu().numpy()

                evidences2[i, l, t] = new_ev
                print("p(y+/y-) sampler: {}\n exact: {}".format(lik_traces[i, l, t][-1], new_ev), lik_traces[i, l, t][-1])

        np.save(os.path.join(save_folder, file_name), lik_traces)
        np.save(os.path.join(save_folder, "test_exact.npy"), evidences2)

if test_case == 2 or test_case == 3:
    # plot the relative error in function of dimension at a fixed alpha and number of iterations
    # 2 : skrock for sampling post, 3: analytical post
    save_folder = os.path.join(fig_folder, "convergence_speed")
    file_name = "skrock_trace2.npy" if test_case == 2 else "analytical_post2.npy"
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

        L_f = 1 / (sigma**2 / alpha)  
        L_g = d / sigmax**2
        gamma = 0.98*1/(L_f + L_g)
        y, x, p = ys[l], xs[l], ps[l]
        for t in range(ntry):
            print("------- t={}/{}, d={}".format(t, ntry-1, d))
            noise = noises[l][:, :, :, t]

            if test_case == 2:
                dl = DegradedLikelihood(y, g, p, sigma, gamma, X_init=p.A_A_adjoint(y).to(device).clone(), noise=noise, batch_size=batch_size,
                                        sampler=SKROCK,sampler_kwargs={'s':15}, lam_reg=None, project=None, alpha=alpha)
            else:
                dl = DegradedLikelihood(y, g, p, sigma, gamma, X_init=p.A_A_adjoint(y).to(device).clone(),sampler=GaussianDiag, noise=noise, 
                                sampler_kwargs={'d':torch.tensor(d).to(device), 'sigma':torch.tensor(sigma*sigmax / np.sqrt(sigma**2+alpha*sigmax**2)).to(device)},
                                lam_reg=None, project=None, alpha=alpha, sigmax=sigmax)
            # use a sampler 
            lik_trace, lik_mean = dl.compute_test(nb_steps, burnin_ratio=burnin_ratio, log_stats=False, thinning=1, normalize=True)        
        
            lik_traces[l, t] =  - lik_mean
            evidences2[l, t] = compute_test_gaussian_diag(dims[l], sigmax, sigma, alpha.item(), y.cpu().numpy(), yp=dl.y_add.cpu().numpy(), ym=dl.y_sub.cpu().numpy(), mlog=True)

            print("p(y+/y-) sampler: {}\n exact: {}".format(-lik_mean, evidences2[l, t]))
            np.save(os.path.join(save_folder, file_name), lik_traces)
            np.save(os.path.join(save_folder, "test_exact2.npy"), evidences2)

########

# 2. Accuracy

if test_case == 4:
    # Model misspecification : plot p(y+/y-, sigma)/p(y+/y-, sigma_true) in function of sigma for different alpha, smooth over 50 tries
    save_folder = os.path.join(fig_folder, "accuracy")
    
    np.random.seed(0)
    torch.manual_seed(0)
    nsigma = 50  # test nval dimensions, with ntry noise samples
    ntry = 5
    d = 500
    alphas = torch.tensor([0.9]).to(device)
    nval = len(alphas)
    evidences2 = np.zeros([nval, nsigma, ntry])  # using p(y+/y-)
    sigmax_ex, sigma = 1., 0.05
    y, x, p = generate_measurements_gaussian_diag(d, sigmax_ex, sigma)
    noises = torch.randn([ntry, 1, 1, d], device=device)*sigma
    sigmaxs = torch.linspace(0.05, 10., nsigma)

    batch_size = 25
    nb_steps = 100000
    burnin_ratio = 500 if test_case == 4 else 0
    thinning = 1 if test_case == 4 else 1
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
                L_f = 1 / (sigma**2 / alpha)  
                L_g = d / sigmax_ex**2
                gamma = torch.tensor(0.98*1/(L_f + L_g), device=device, dtype=torch.float32)
                g = DiagonalWeightedTikhonovPrior(torch.tensor(1., device=device), torch.ones(d, device=device)/sigmax_ex**2)
                # use SKROCK and MC
                dl = DegradedLikelihood(y, g, p, sigma, gamma, X_init=p.A_A_adjoint(y).to(device).clone(), noise=noise,
                                        sampler=SKROCK,sampler_kwargs={'s':15}, lam_reg=None, project=None, alpha=alpha, batch_size=batch_size)   
                lik_trace, lik_mean = dl.compute_test(nb_steps, burnin_ratio=burnin_ratio, log_stats=False, 
                                                      thinning=thinning, normalize=True, log_wu=False)
                approx_val_true[i, k] = - lik_mean

                # true sigma, exact value
                exact_vals[i, k] = compute_test_gaussian_diag(d, sigmax_ex, sigma, alpha.cpu(), y.cpu().numpy(), yp=dl.y_add.cpu().numpy(), ym=dl.y_sub.cpu().numpy(), mlog=True)

                for l in range(nsigma):
                   
                    new_sigmax = sigmaxs[l]
                    L_g = d / new_sigmax**2
                    gamma = 0.98*1/(L_f + L_g)

                    g = DiagonalWeightedTikhonovPrior(torch.tensor(1., device=device), torch.ones(d, device=device)/new_sigmax**2)
                    dl = DegradedLikelihood(y, g, p, sigma, gamma, X_init=p.A_A_adjoint(y).to(device).clone(), noise=noise, batch_size=batch_size,
                                            sampler=SKROCK,sampler_kwargs={'s':15}, lam_reg=None, project=None, alpha=alpha)
                    
                    lik_trace, lik_mean = dl.compute_test(nb_steps, burnin_ratio=burnin_ratio, log_stats=False, thinning=thinning, normalize=True)
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

    np.save(os.path.join(save_folder, "approx_ratio_gaussian.npy"), approx_vals)
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

########################################################################

# II. Synthetic Laplace problem : $y = A x + n$, $x\sim \mathcal L(0,1/sigma_x)$, $n\sim \mathcal N(0, \sigma^2 I_d) where A applies a gaussian blur$ 

if test_case == 6:
    # model misspecification : plot p(y+/y-, sigma)/p(y+/y-, sigma_true) in function of sigma for different alpha, laplace prior
    img_size = 128
    np.random.seed(6)
    torch.manual_seed(6)
    save_folder = os.path.join(fig_folder, "accuracy")

    sigmax_ex, sigma = 0.25, 0.05
    y, x, p = generate_measurements_laplace(img_size, sigmax_ex, sigma)
    noise = torch.randn([1, 1, img_size, img_size], device=device)*sigma

    nsigmas = 50
    sigmaxs = torch.linspace(0.05, 20., nsigmas)

    alphas = torch.tensor([0.25, 0.5, 0.75]).to(device)
    nval = len(alphas)
    
    nb_steps, burnin_ratio = 20000, 1000
    batch_size = 5
    approx_vals = np.zeros([nval, nsigmas])
    approx_vals_gt = np.zeros([nval])
    for i in range(nval):

        alpha = alphas[i]
        for l in range(nsigmas):
            print("------- l={}/{}".format(l, nsigmas-1), "alpha={}".format(alpha))
            new_sigmax = sigmaxs[l]
            L_f = 1 / (sigma**2 / alpha)  
            lam_reg = min(1/L_f, 2.)   
            L_g = 1/lam_reg
            gamma = 0.98*1/(L_f + L_g)

            g = L1Prior(torch.tensor(new_sigmax, device=device))
        
            dl = DegradedLikelihood(y, g, p, sigma, gamma, X_init=p.A_A_adjoint(y).to(device).clone(), noise=noise, batch_size=batch_size,
                                    sampler=SKROCK,sampler_kwargs={'s':15}, lam_reg=lam_reg, project=None, alpha=alpha)
            # compute exact p(y+/y-)
            _, lmean = dl.compute_test(nb_steps, burnin_ratio=burnin_ratio, log_stats=False, thinning=1, normalize=True, log_wu=False)
            approx_vals[i, l] = - lmean
        g = L1Prior(torch.tensor(1/sigmax_ex, device=device))
        dl = DegradedLikelihood(y, g, p, sigma, gamma, X_init=p.A_A_adjoint(y).to(device).clone(), noise=noise, batch_size=batch_size,
                                sampler=SKROCK,sampler_kwargs={'s':15}, lam_reg=lam_reg, project=None, alpha=alpha)
        _, lmean = dl.compute_test(nb_steps, burnin_ratio=burnin_ratio, log_stats=False, thinning=1, normalize=True)
        approx_vals_gt[i] = - lmean

    np.save(os.path.join(save_folder, "approx_ratio_laplace2.npy"), approx_vals)
    np.save(os.path.join(save_folder, "sigmas_laplace2.npy"), sigmaxs.cpu().numpy())
    np.save(os.path.join(save_folder, "alphas_laplace2.npy"), alphas.cpu().numpy())
    np.save(os.path.join(save_folder, "gt_vals_laplace2.npy"), approx_vals_gt)


if test_case == 7:
    # model misspecification : plot p(y+/y-, kernel)/p(y+/y-, kernel_true) as a function of the kernel's sigma for different alpha, using the gradient step denoiser associated prior
    img_size = 128
    np.random.seed(test_case)
    torch.manual_seed(test_case)
    save_folder = os.path.join(fig_folder, "accuracy")
    sigma =  0.1
    sigmax_ex = 1
    y, x, p = generate_measurements_natural(img_size, sigma)
    noise = torch.randn([1, 1, img_size, img_size], device=device)*sigma

    nsigmas = 7
    sigmaxs = np.array([0.01, 0.25, .5, 0.75, 1.25, 1.5, 1.75])#torch.linspace(0.01, 10., nsigmas)

    alphas = torch.tensor([0.25, 0.5, 0.75]).to(device)
    nval = len(alphas)
    
    nb_steps, burnin_ratio = 20000, 5000
    batch_size = 2
    
    approx_vals = np.zeros([nval, nsigmas])
    approx_vals_gt = np.zeros([nval])
    path_ckpt = "GSDRUNet_grayscale_torch.ckpt" 
    denoiser = dinv.models.GSDRUNet(pretrained=path_ckpt, in_channels=1, out_channels=1, device=device)
    g = GSDPrior(torch.tensor(110.), denoiser)
    trace_exact = np.zeros([nval, nb_steps])
    thinning = 1
    for i in range(nval):

        alpha = alphas[i]
        for l in range(nsigmas):
            print("------- l={}/{}".format(l, nsigmas-1), "alpha={}".format(alpha))
            new_sigmax = sigmaxs[l]
            L_f = 1 / (sigma**2 / alpha)  
            L_g = 2

            gamma = 0.98*1/(L_f + L_g)
            p = generate_blur_operator(img_size, sigma=sigma, sigma_blur=new_sigmax)
            dl = DegradedLikelihood(y, g, p, sigma, gamma, X_init=p.A_A_adjoint(y).to(device).clone(), noise=noise,
                                    sampler=SKROCK,sampler_kwargs={'s':15}, project=None, alpha=alpha, batch_size=batch_size)
            _, lmean = dl.compute_test(nb_steps, burnin_ratio=burnin_ratio, log_stats=False, thinning=thinning, normalize=True, log_wu=False)
            approx_vals[i, l] = - lmean
        p = generate_blur_operator(img_size, sigma=sigma, sigma_blur=sigmax_ex)
        dl = DegradedLikelihood(y, g, p, sigma, gamma, X_init=p.A_A_adjoint(y).to(device).clone(), noise=noise,
                                sampler=SKROCK,sampler_kwargs={'s':15}, project=None, alpha=alpha, batch_size=batch_size)
        trace, lmean = dl.compute_test(nb_steps, burnin_ratio=burnin_ratio, log_stats=False, thinning=thinning, normalize=True)
        approx_vals_gt[i] = - lmean
        trace_exact[i] = trace.cpu()

    np.save(os.path.join(save_folder, "approx_nat2.npy"), approx_vals)
    np.save(os.path.join(save_folder, "trace_exact_nat2.npy"), trace_exact)
    np.save(os.path.join(save_folder, "sigmas_nat2.npy"), sigmaxs)
    np.save(os.path.join(save_folder, "alphas_nat2.npy"), alphas.cpu().numpy())
    np.save(os.path.join(save_folder, "gt_vals_nat2.npy"), approx_vals_gt)