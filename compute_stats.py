import deepinv as dinv
import torch
import numpy as np
import os
from prior_comparison import DegradedLikelihood
from utils import device, dict_to_json, json_to_dict
from PIL.Image import fromarray
from sampling import DiffPIR
import sys
from torchvision.datasets import ImageFolder
from torchvision.transforms import Resize
from torchvision import transforms
from experiments_utils import generate_blur_operator

folder = "datasets/afhq"
save_folder = "results/diffusion/stats_exp/afhq_dogs_afhq"#"results/diffusion/stats_exp/stats_afhq"

model_path = "models/diffusion/afhqdog_p2.pt"#"models/diffusion/ffhq_p2.pt"
denoiser = dinv.models.DiffUNet(pretrained=model_path, in_channels=3, out_channels=3, large_model=False).to(device)

# parameters for the experiments
noise_level_img = 0.05  # noise added to the image after blurring
img_size = 256

nb_steps, nb_noise = 1, 150
niter_diffpir = 300
batch_size = 2

noise_schedule_path = "figs_diff/ds_ffhqp2/noise_schedule2.npy"  # path for E_eps noise schedule (regenerated otherwise)
regenerate = False
if regenerate:
    noise_schedule = torch.randn((nb_noise//batch_size, batch_size,) + (3, img_size, img_size), device=device)*noise_level_img
    np.save(noise_schedule_path, noise_schedule.cpu().numpy())
else:
    noise_schedule = torch.tensor(np.load(noise_schedule_path).astype(np.float32)).to(device)

sigma_blur = 5.
blur_op = generate_blur_operator(img_size, 
                                 filter_torch=dinv.physics.blur.gaussian_blur(sigma=(sigma_blur, sigma_blur)).to(device), 
                                 sigma=noise_level_img)

alpha = 0.75

if len(sys.argv[1:]) == 2:
    ind_start, ind_end = int(sys.argv[1]), int(sys.argv[2])
else:
    ind_start, ind_end = 0, 99
    
meta_file_name = "params.json"
val_transform = transforms.Compose([Resize((256, 256)), transforms.ToTensor()])
ds = ImageFolder(folder, val_transform)  # create a dataloader instance

param_dict = dict(alpha=alpha, nb_noise=nb_noise, nb_steps=nb_steps, 
                  niter_diffpir=niter_diffpir, sigma_blur=sigma_blur, batch_size=batch_size, 
                  noise_schedule=noise_schedule.cpu().numpy())
dict_to_json(param_dict, os.path.join(save_folder, meta_file_name))
alpha = torch.tensor(alpha)
for i in range(ind_end - ind_start + 1):
    curr_ind = i + ind_start
    blur_op = generate_blur_operator(img_size, 
                                     filter_torch=dinv.physics.blur.gaussian_blur(sigma=(sigma_blur, sigma_blur)).to(device), 
                                     sigma=noise_level_img)
    print("Computing for image " + str(curr_ind))
    img_path = ds.samples[curr_ind]
    x, cat = ds[curr_ind]
    x = x.unsqueeze(0).to(device)
    y = blur_op(x)  # apply blur and noise

    dl = DegradedLikelihood(y, denoiser, blur_op, noise_level_img, 0, X_init=blur_op.A_A_adjoint(y).to(device).clone(),
                            sampler=DiffPIR,sampler_kwargs={'batch_size':1, 'physics':blur_op, 
                                                            'denoiser':denoiser, 'max_iter': niter_diffpir}, 
                            project=None, alpha=alpha, batch_size=batch_size)

    trace, lmean, post_mean, psnr_trace = dl.compute_test2(nb_steps, burnin_ratio=0, nb_noise=nb_noise,
                                                           thinning=1, thinning_noise=0, normalize=True, 
                                                           noise_schedule=noise_schedule, x=x)

    local_res_dict = dict(trace=trace.numpy(), post_mean=post_mean.numpy(), psnr_trace=psnr_trace.numpy(), 
                          img_path=img_path, cat=cat, y=y.cpu().numpy())
    dict_to_json(local_res_dict, os.path.join(save_folder, "trace_{}.json".format(curr_ind)))

