from deepinv.loss.metric import PSNR
from deepinv.physics import Decolorize
from deepinv.models import GSDRUNet
from deepinv.physics.blur import gaussian_blur
import torch
import numpy as np
import os
from priors import GSDPrior, L2
from prior_comparison import DegradedLikelihood
from utils import device, dict_to_json, laplace, moffat, uniform
from experiments_utils import generate_blur_operator
from sampling import SKROCK
import sys
from torchvision.datasets import ImageFolder
from torchvision.transforms import Resize
from torchvision import transforms

mode = "compute_psnr"

folder = "datasets/set3c/"
save_folder = "results/kernel_comparison"

torch.manual_seed(42)

if len(sys.argv[1:]) == 3:
    ind_gt, ind_start, ind_end = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
else:
    print("Usage: python kernel_comparison.py ind_gt ind_start ind_end")
    exit(1)


sigma =  0.1
kernels = [gaussian_blur(sigma=(2, 2)),
           moffat((0.5, 1), size=7),
           laplace(0.4, size=10), uniform(3), 
           gaussian_blur(sigma=(2.5, 2.5)),]
nkernels = len(kernels)

img_size = 256
val_transform = transforms.Compose([Resize((img_size, img_size)), transforms.ToTensor()])
ds_parent = ImageFolder(folder, val_transform)  # create a dataloader instance

ds = ds_parent 

alpha = 0.75
nb_noise = 10
nb_steps, burnin_ratio = 100, 50
thinning_noise = 20
nb_samples = 100  # nb of samples for computing the posterior mean
   
batch_size = 1

noise_schedule_path = "results/kernel_comparison/noise_schedule.npy"  # path for E_eps noise schedule (regenerated otherwise)
regenerate = False
if regenerate:
    noise_schedule = torch.randn((nb_noise//batch_size, batch_size,) + (1, img_size, img_size), device=device)*sigma
    np.save(noise_schedule_path, noise_schedule.cpu().numpy())
else:
    noise_schedule = torch.tensor(np.load(noise_schedule_path).astype(np.float32)).to(device)
    
meta_file_name = "params.json"

param_dict = dict(alpha=alpha, nb_noise=nb_noise, nb_steps=nb_steps, 
                  batch_size=batch_size, 
                  noise_schedule=noise_schedule.cpu().numpy())
dict_to_json(param_dict, os.path.join(save_folder, meta_file_name))

denoiser = GSDRUNet(pretrained="models/GSDRUNet_grayscale_torch.ckpt", in_channels=1, out_channels=1, device=device)
g = GSDPrior(torch.tensor(110.), denoiser)

alpha = torch.tensor(alpha).to(device)
s = 15
ls = (s - 0.5)**2*(2 - 4/3*0.05) - 1.5
deco = Decolorize(device=device)

for curr_ind in range(ind_start, ind_end + 1):
    blur_op_gt = generate_blur_operator(img_size, filter_torch=kernels[ind_gt], sigma=sigma)
    blur_op_gt.noise_model.rng_manual_seed(curr_ind)  # for reproducibility

    # generate blurred measurements using the ground truth kernel
    x, cat = ds[curr_ind]
    x = x.unsqueeze(0).to(device)

    x = deco(x)

    y = blur_op_gt(x)  # apply blur and noise
    img_path = ds_parent.samples[curr_ind][0]#ds_parent.samples[ds.indices[curr_ind]][0]

    for l in range(nkernels):
        blur_op = generate_blur_operator(img_size, filter_torch=kernels[l], sigma=sigma)
        print("Computing for image " + str(curr_ind))

        L_f = blur_op.compute_norm(x0=torch.randn_like(x), tol=1e-5) / (sigma**2 / alpha)  
        L_g = 5*110.
        gamma = 0.98*ls/(L_f + L_g)

        if mode == "compute_estimator":
            dl = DegradedLikelihood(y.clone(), g, blur_op, sigma, gamma=gamma, X_init=blur_op.A_adjoint(y).to(device).clone(),
                                    sampler=SKROCK, sampler_kwargs={'s':s},
                                    project=None, alpha=alpha, batch_size=batch_size)
            samples_x, samples_ym, samples_yp = dl.save_samples(nb_steps, burnin_ratio=burnin_ratio, nb_noise=nb_noise, 
                                                                    thinning=1, thinning_noise=thinning_noise, 
                                                                    noise_schedule=noise_schedule, compute_xp=False)
            np.savez(os.path.join(save_folder, "trace_{}_{}_{}.npz".format(ind_gt, l, curr_ind)), samples_x=samples_x.numpy(), 
                     samples_ym=samples_ym.numpy(), samples_yp=samples_yp.numpy(),
                     img_path=img_path, y=y.cpu().numpy(), x=x.cpu().numpy())
            
        elif mode == "compute_psnr":
            psnr = PSNR()

            res = np.zeros([nb_samples, 256**2], dtype=np.float32)

            f = L2(sigma, blur_op)

            gradU = lambda t, y: f.grad(t, y) + g.grad(t, None)
        
            sampler = SKROCK(gradU, gamma, blur_op.A_A_adjoint(y).to(device).clone(), lambda t: t, s=15)
        
            for i in range(100):  # warm up
                sampler(y)
        
            for i in range(nb_steps):
                sampler(y)
                res[i] = sampler.X.cpu().numpy().reshape(-1)
            np.save(os.path.join(save_folder, "samples_y_{}_{}_{}.npy".format(ind_gt, l, curr_ind)), res)
