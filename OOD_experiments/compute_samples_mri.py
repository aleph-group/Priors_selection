import torch
import numpy as np
import os
from prior_comparison import DegradedLikelihood
from utils import device, json_to_dict, normalize
from sampling import DiffPIR
import sys
from deepinv.physics import MRI, GaussianNoise, PoissonNoise
from deepinv.physics.generator import GaussianMaskGenerator


if len(sys.argv[1:]) == 3:
    out_folder = sys.argv[1]
    ind_start, ind_end = int(sys.argv[2]), int(sys.argv[3])
else:
    print("Usage: python3 compute_samples_natural.py path ind_start ind_end")
    exit(1)

    
config_file = json_to_dict(os.path.join(out_folder, "config.json"))
in_folder = config_file["in_folder"]

denoiser = torch.load(config_file["model_path"], weights_only=False).to(device)


class InputWrapper(torch.nn.Module):  # wrapper class to use deepinv MRI in DiffPIR
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model.to(device)

    def forward(self, x, scalar):
        extra = 2*normalize(x[:, :1]) -1 # extract the real part, map to [-1, 1]
        respart = normalize(self.base_model(extra, 2*scalar))  # noise is x2 after mapping, remap to [0, 1]
        res = torch.zeros_like(x, device=extra.device)
        res[:, 0] = respart[:, 0]
        return res

        
denoiser = InputWrapper(denoiser)

sigma = config_file["sigma"]  # measurement noise
img_size = 320

nb_steps, nb_noise = config_file["nb_steps"], config_file["nb_noise"] 
niter_diffpir = 500
batch_size = 2

noise_schedule_path = config_file["noise_path"]  
noise_schedule = torch.tensor(np.load(noise_schedule_path), device=device).float()

alpha = torch.tensor(config_file["alpha"], device=device)
acceleration = config_file["acceleration"]  # acceleration factor for MRI measurements


for i in range(ind_start, ind_end + 1):

    print("Computing for image " + str(i))
    img_path = os.path.join(in_folder, "img_{}.npy".format(i))

    x = torch.zeros([1, 2, img_size, img_size], device=device) 
    x[0, 0] = normalize(torch.tensor(np.load(img_path).reshape(1, 1, img_size, img_size), device=device).float())

    mask_gen = GaussianMaskGenerator(acceleration=acceleration, img_size=(img_size,img_size), seed=i)
    mask = mask_gen.step()["mask"].to(device)

    noise_model = PoissonNoise()#GaussianNoise(sigma, rng=torch.Generator(device=device))
    noise_model.rng_manual_seed(i)  # for reproducibility
    
    physics = MRI(mask=mask, img_size=(img_size, img_size), device=device, noise_model=noise_model)
    y = physics(x)  # generate measurements

    dl = DegradedLikelihood(y=y, prior=denoiser, physics=physics, sigma=sigma, gamma=0, 
                            X_init=physics.A_adjoint(y).to(device).clone(),
                            sampler=DiffPIR,sampler_kwargs={'batch_size':1, 'physics':physics, 
                                                            'denoiser':denoiser, 'max_iter': niter_diffpir}, 
                            project=None, alpha=alpha, batch_size=batch_size)


    
    samples_x, samples_ym, samples_yp, samples_xp = dl.save_samples(nb_steps, burnin_ratio=0, nb_noise=nb_noise, 
                                                                    thinning=1, thinning_noise=0, 
                                                                    noise_schedule=noise_schedule, compute_xp=True)

    samples_x, samples_xp = samples_x[:, :, :1], samples_xp[:, :1]
    x = x[:, :1]
    np.savez(os.path.join(out_folder, "trace_{}.npz".format(i)), samples_x=samples_x.numpy(), 
             samples_ym=samples_ym.numpy(), samples_yp=samples_yp.numpy(), samples_xp=samples_xp.numpy(),
             img_path=img_path, y=y.cpu().numpy(), x=x.cpu().numpy(), mask=mask.cpu().numpy())

