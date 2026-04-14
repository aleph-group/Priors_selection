import deepinv as dinv
import torch
import numpy as np
import os
from prior_comparison import DegradedLikelihood
from utils import device, dict_to_json, json_to_dict
from PIL.Image import fromarray
from sampling import SKROCK
import sys
from torchvision.datasets import ImageFolder
from torchvision.transforms import Resize
from torchvision import transforms
from experiments_utils import generate_blur_operator
import tqdm
from priors import GSDPrior, L2
from deepinv.models import GSDRUNet


# parameters for the experiments
sigma = 0.05  # noise added to the image after blurring
img_size = 256

batch_size = 4

if len(sys.argv[1:]) == 4:
    out_folder = sys.argv[1]
    save_folder = sys.argv[4]
    ind_start, ind_end = int(sys.argv[2]), int(sys.argv[3])
else:
    ind_start, ind_end = 0, 74
config_file = json_to_dict(os.path.join(out_folder, "config.json"))

model_path = "models/GSDRUNet_torch.ckpt"
denoiser = GSDRUNet(pretrained=model_path, in_channels=3, out_channels=3, device=device)
sigma_blur = config_file["sigma_blur"]

lam = 700.
g = GSDPrior(torch.tensor(lam), denoiser)
L_g = lam*5
s=15
ls = (s - 0.5)**2*(2 - 4/3*0.05) - 1.5
gradU = lambda t, y: f.grad(t, y) + g.grad(t, lam)

nb_samples = 40
burnin = 50
thinning = 5

for i in tqdm.tqdm(range(ind_end - ind_start + 1)):
    curr_ind = i + ind_start

    
    
    print("Sampling for image {}".format(curr_ind))

    physics = generate_blur_operator(img_size, 
                                 filter_torch=dinv.physics.blur.gaussian_blur(sigma=(sigma_blur, sigma_blur)).to(device), 
                                 sigma=sigma)
    res_dict = np.load(os.path.join(out_folder, "trace_{}.npz".format(curr_ind)))
    y = torch.tensor(res_dict["y"].astype(np.float32), device=device)
    
    if i == 0:
        f = L2(sigma, physics)  # likelihood for Ax + N + e*calpha

        L_f = physics.compute_norm(x0=torch.randn_like(y.to(device)), tol=1e-5)  
        gamma = 0.98*ls/(L_f/ sigma**2  + L_g)
         
    sam = SKROCK(gradU, gamma, physics.A_adjoint(y).to(device).repeat(*[batch_size] + [1 for _ in range(3)]), proj=lambda x: x, s=15)
    for _ in range(burnin):
        with torch.no_grad():
            sam(y)
    res = torch.zeros((nb_samples, 3, 256, 256), device=device)
    for i in range(nb_samples//batch_size):
        with torch.no_grad():
            for _ in range(thinning):
                sam(y)
            res[batch_size*i:batch_size*(i+1)] = sam(y)
    
    np.save(os.path.join(save_folder, "samples_{}.npy".format(curr_ind)), res.cpu().numpy())
    #np.save(os.path.join(save_folder, "mean_{}.npy".format(curr_ind)), torch.mean(res, axis=0).cpu().numpy())


