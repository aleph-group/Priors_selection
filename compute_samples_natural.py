import torch
import numpy as np
import os
from prior_comparison import DegradedLikelihood
from utils import device, json_to_dict
from sampling import DiffPIR
import sys
from torchvision.datasets import ImageFolder
from torchvision import transforms
from experiments_utils import generate_blur_operator
from deepinv.physics.blur import gaussian_blur
from deepinv.models import DiffUNet
from deepinv.physics import PoissonNoise, BlurFFT

if len(sys.argv[1:]) == 3:
    out_folder = sys.argv[1]
    ind_start, ind_end = int(sys.argv[2]), int(sys.argv[3])
else:
    print("Usage: python3 compute_samples_natural.py path ind_start ind_end")
    exit(1)
    
config_file = json_to_dict(os.path.join(out_folder, "config.json"))
in_folder = config_file["in_folder"]

model_path = config_file["model_path"]
denoiser = DiffUNet(pretrained=model_path, in_channels=3, out_channels=3, large_model=False).to(device)

sigma =  config_file["sigma"]  # measurement noise
img_size = 256

nb_steps, nb_noise = config_file["nb_steps"], config_file["nb_noise"] 
niter_diffpir = 50#300
batch_size = 1

#noise_schedule_path = config_file["noise_path"]  
#noise_schedule = torch.tensor(np.load(noise_schedule_path), device=device).float()

alpha = torch.tensor(config_file["alpha"], device=device)
sigma_blur = config_file["sigma_blur"]  # gaussian blur standard deviation

val_transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
ds = ImageFolder(in_folder, val_transform)  # create a dataloader instance

for i in range(ind_start, ind_end + 1):
    print("Computing for image " + str(i))
    
    noise_model = PoissonNoise(sigma, rng=torch.Generator(device=device), clip_positive=True)

    filter_torch = gaussian_blur(sigma=(0.5, 0.5)).to(device)

    physics =  BlurFFT(img_size=(1, 256, 256), filter=filter_torch, device=device, padding="circular",
    noise_model=noise_model)
    
    add = 0 if in_folder.endswith("ffhqsub") else 75
    physics.noise_model.rng_manual_seed(i+add)  # for reproducibility

    img_path = ds.samples[i]
    x, cat = ds[i]
    x = x.unsqueeze(0).to(device)
    y = physics(x)  # apply blur and noise
    
    dl = DegradedLikelihood(y=y, prior=denoiser, physics=physics, sigma=sigma, gamma=0, 
                            X_init=physics.A_adjoint(y).to(device).clone(),
                            sampler=DiffPIR, sampler_kwargs={'batch_size':1, 'physics':physics,  
                                                            'denoiser':denoiser, 'max_iter': niter_diffpir}, 
                            project=None, alpha=alpha, batch_size=batch_size)


    
    samples_x, samples_ym, samples_yp, samples_xp = dl.save_samples(nb_steps, burnin_ratio=0, nb_noise=nb_noise, 
                                                                    thinning=1, thinning_noise=0, 
                                                                    noise_schedule=None, compute_xp=True)

   
    np.savez(os.path.join(out_folder, "trace_{}.npz".format(i)), samples_x=samples_x.numpy(), 
             samples_ym=samples_ym.numpy(), samples_yp=samples_yp.numpy(), samples_xp=samples_xp.numpy(),
             img_path=img_path, y=y.cpu().numpy(), x=x.cpu().numpy(), cat=cat)

