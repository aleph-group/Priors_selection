import torch
import numpy as np
import os
from prior_comparison import DegradedLikelihood
from utils import device, json_to_dict
from sampling import SKROCK
from priors import TVPrior, L2

import sys
from torchvision.datasets import ImageFolder
from torchvision import transforms
from experiments_utils import generate_blur_operator
from deepinv.physics.blur import gaussian_blur


if len(sys.argv[1:]) == 3:
    out_folder = sys.argv[1]
    ind_start, ind_end = int(sys.argv[2]), int(sys.argv[3])
else:
    print("Usage: python3 compute_samples_natural.py path ind_start ind_end")
    exit(1)
    
config_file = json_to_dict(os.path.join(out_folder, "config.json"))
in_folder = config_file["in_folder"]

sigma = config_file["sigma"]  # measurement noise
img_size = 256

nb_steps, nb_noise = 20, config_file["nb_noise"]  #config_file["nb_steps"]
batch_size = 5

noise_schedule_path = config_file["noise_path"]  
noise_schedule = torch.tensor(np.load(noise_schedule_path), device=device).float()

alpha = torch.tensor(config_file["alpha"], device=device)
sigma_blur = config_file["sigma_blur"]  # gaussian blur standard deviation

lam = 20.
g = TVPrior(torch.tensor(lam), n_it_max=100)
s=15
ls = (s - 0.5)**2*(2 - 4/3*0.05) - 1.5


val_transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
ds = ImageFolder(in_folder, val_transform)  # create a dataloader instance
print("sigma {} ".format(sigma_blur))
for i in range(ind_start, ind_end + 1):
    print("Computing for image " + str(i))
    
    physics = generate_blur_operator(img_size, 
                                     filter_torch=gaussian_blur(sigma=(sigma_blur, sigma_blur)).to(device), 
                                     sigma=sigma)
    if i == ind_start:
        L_f = physics.compute_norm(x0=torch.randn_like(ds[i][0].to(device)), tol=1e-5) / (sigma**2 / alpha)
        L_fp = physics.compute_norm(x0=torch.randn_like(ds[i][0].to(device)), tol=1e-5) / (sigma**2 / (1-alpha))

        lam_reg = min(1/L_f, 2.)   
        lam_regp = min(1/L_fp, 2.)   

        gamma = 0.9*ls/(L_f/ (sigma**2 / alpha)  + 1/lam_reg)
        gammap = 0.9*ls/(L_f/ (sigma**2 / (1-alpha))  + 1/lam_regp)
        print("gamma ", gamma, gammap)

    physics.noise_model.rng_manual_seed(i)  # for reproducibility
    
    img_path = ds.samples[i]
    x, cat = ds[i]
    x = x.unsqueeze(0).to(device)
    y = physics(x)  # apply blur and noise

    dl = DegradedLikelihood(y=y, prior=g, physics=physics, sigma=sigma, gamma=gamma, 
                            X_init=physics.A_adjoint(y).to(device).clone(),
                            sampler=SKROCK, sampler_kwargs={'s':s}, 
                            project=None, alpha=alpha, batch_size=batch_size, gammap=gammap,
                           lam_reg=lam_reg, lam_regp=lam_regp)


    
    samples_x, samples_ym, samples_yp, samples_xp = dl.save_samples(nb_steps, burnin_ratio=50, nb_noise=nb_noise, 
                                                                    thinning=5, thinning_noise=20, 
                                                                    noise_schedule=noise_schedule, compute_xp=True)

   
    np.savez(os.path.join(out_folder, "trace_{}.npz".format(i)), samples_x=samples_x.numpy(), 
             samples_ym=samples_ym.numpy(), samples_yp=samples_yp.numpy(), samples_xp=samples_xp.numpy(),
             img_path=img_path, y=y.cpu().numpy(), x=x.cpu().numpy(), cat=cat)

