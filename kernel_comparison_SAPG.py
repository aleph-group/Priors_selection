"""Script to run SAPG to tune the regularization parameter for each measurement and model."""

import deepinv as dinv
import torch
import matplotlib.pyplot as plt
from deepinv.utils.demo import load_dataset
from torchvision import transforms
import numpy as np
from sapg import SAPG
from experiments_utils import generate_blur_operator
from priors import WaveletPrior, GSDPrior, L2, CombinedPrior, L1Prior, CRRPrior, CRRPriorStaticMu
from utils import plot_MC_corr, device, laplace, moffat, uniform
import sys


ind_gt, ind_kernel = int(sys.argv[1]), int(sys.argv[2]) 

# Set the global random seed from pytorch to ensure reproducibility of the example.
torch.manual_seed(0)
img_size = 256


kernels = [dinv.physics.blur.gaussian_blur(sigma=(2, 2)),
           moffat((0.5, 1), size=7),
           laplace(0.4, size=10), uniform(3), 
           dinv.physics.blur.gaussian_blur(sigma=(2.5, 2.5)),]

nkernels = len(kernels)

measurements = torch.zeros(nkernels, 3, 1, img_size, img_size)
for i in range(5):
    for j in range(3):
        measurements[i, j] = torch.tensor(np.load("results/kernel_comparison/trace_{}_0_{}.npz".format(i, j))["y"])
best_params = np.zeros(3)

sigma = 0.1  # Gaussian Noise standard deviation for the degradation


# Lipschitz constant of nabla f
L_g = 5*250.

denoiser = dinv.models.GSDRUNet(pretrained="models/GSDRUNet_grayscale_torch.ckpt", 
                                in_channels=1, out_channels=1, device=device)
theta0 = torch.tensor(110., device=device)

g = GSDPrior(theta0, denoiser)
              
   
theta0 = torch.tensor(110., device=device)

physics = generate_blur_operator(img_size, kernels[ind_kernel], sigma=sigma)

L_f = physics.compute_norm(x0=torch.randn(1, 1, img_size, img_size).to(device), 
                           tol=1e-5) / sigma**2
# regularization parameter of proximity operator (lambda)
lam_reg = min(1/L_f, 2.)   
f = L2(sigma, physics)
for k in range(3):
    print("Processing image ", k)
    with torch.no_grad():
        y = measurements[ind_gt, k].to(device)
        L =  L_f + L_g
    
        gamma = 0.98*1/L
        d0 = 5000. / img_size**2 / theta0
        delta = lambda k: d0 / (k ** 0.6)
        sapg = SAPG(y, g, f, gamma, 0.98*gamma, lam_reg, X_init=physics.A_adjoint(y).to(device).detach().clone(), 
                    project=None, sampler='SKROCK', sampler_kwargs=dict(s=15)) 
        nwarm = 100
        sapg.warm_up(nwarm, log_stats=False, burnin_ratio=0.95, warm_up_prior=False)
    
        param, param_hist, mean_hist, post_hist, prior_hist = sapg.run(delta, 150, (50., 1e6), init_param=theta0, reuse_post=True,  burnin_ratio=0.5,
                                                                       thinning_post=15, thinning_prior=25, alpha=None, thinning_global=1, tol=1e-4, verbose=False)
        best_params[k] = mean_hist[-1]
        fig, ax = plt.subplots(1, 3, figsize=(10, 5))
        ax[0].plot(-post_hist.numpy(), label=r'$g(X)$')
        ax[0].plot(-prior_hist.numpy(), '--', label=r'$g(\bar X)$')
        ax[1].plot(param_hist.numpy(), label=r'$\theta$')
        ax[2].plot(mean_hist.numpy(), label=r'$\bar{\theta}$')
        ax[0].legend(), ax[1].legend(), ax[2].legend();
        plt.tight_layout()
        plt.savefig("figs/GSD_SAPG/{}_{}_{}.pdf".format(ind_gt, ind_kernel, k))
        plt.show()
        
np.save("results/kernel_comparison/sapg_res_{}_{}.npy".format(ind_gt, ind_kernel), best_params)