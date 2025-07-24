import deepinv as dinv
import torch
import numpy as np
import os
from priors import GSDPrior, L2
from prior_comparison import DegradedLikelihood
from utils import device, dict_to_json, json_to_dict, laplace, moffat, uniform
from experiments_utils import generate_blur_operator
from PIL.Image import fromarray
from sampling import SKROCK
import sys
from torchvision.datasets import ImageFolder
from torchvision.transforms import Resize
from torchvision import transforms
from torch.utils.data import Subset

mode = "compute_psnr"

folder = "datasets/set3c/"#"datasets/test_ds/"  #
save_folder = "results/kernel_comparison"

torch.manual_seed(42)

if len(sys.argv[1:]) == 3:
    ind_gt, ind_start, ind_end = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
else:
    ind_gt, ind_start, ind_end = 0, 0, 2


sigma =  0.1
kernels = [dinv.physics.blur.gaussian_blur(sigma=(2, 2)),
            moffat((0.5, 1), size=7),
            laplace(0.4, size=10), uniform(3), 
            dinv.physics.blur.gaussian_blur(sigma=(2.5, 2.5)),]
nkernels = len(kernels)

img_size = 256
val_transform = transforms.Compose([Resize((img_size, img_size)), transforms.ToTensor()])
ds_parent = ImageFolder(folder, val_transform)  # create a dataloader instance
#lsun_idx =  ds_parent.class_to_idx['met']
#lsun_ind = [i for i, (img, label) in enumerate(ds_parent) if label == lsun_idx]
ds = ds_parent #Subset(ds_parent, lsun_ind)

alpha = 0.75
nb_noise = 10#100
nb_steps, burnin_ratio = 100, 50#10, 50
thinning_noise = 20
nb_samples = 100  # nb of samples for computing the posterior mean
   
batch_size = 1

noise_schedule_path = "results/kernel_comparison/noise_schedule.npy"  # path for E_eps noise schedule (regenerated otherwise)
regenerate = True
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

denoiser = dinv.models.GSDRUNet(pretrained="models/GSDRUNet_grayscale_torch.ckpt", in_channels=1, out_channels=1, device=device)
g = GSDPrior(torch.tensor(110.), denoiser)

alpha = torch.tensor(alpha).to(device)
s = 15
ls = (s - 0.5)**2*(2 - 4/3*0.05) - 1.5
deco = dinv.physics.Decolorize(device=device)

for curr_ind in range(ind_start, ind_end + 1):
    blur_op_gt = generate_blur_operator(img_size, filter_torch=kernels[ind_gt], sigma=sigma)

    # generate blurred measurements using the ground truth kernel
    x, cat = ds[curr_ind]
    x = x.unsqueeze(0).to(device)

    x = deco(x)

    y = blur_op_gt(x)  # apply blur and noise
    img_path = ds_parent.samples[curr_ind][0]#ds_parent.samples[ds.indices[curr_ind]][0]

    for l in range(nkernels):
        #torch.manual_seed(curr_ind+1)

        blur_op = generate_blur_operator(img_size, filter_torch=kernels[l], sigma=sigma)
        print("Computing for image " + str(curr_ind))

        L_f = blur_op.compute_norm(x0=torch.randn_like(x), tol=1e-5) / (sigma**2 / alpha)  
        L_g = 5*110.
        gamma = 0.98*ls/(L_f + L_g)

        if mode == "compute_estimator":
            dl = DegradedLikelihood(y.clone(), g, blur_op, sigma, gamma=gamma, X_init=blur_op.A_A_adjoint(y).to(device).clone(),
                                    sampler=SKROCK, sampler_kwargs={'s':s},
                                    project=None, alpha=alpha, batch_size=batch_size)
        
            trace, lmean, post_mean, psnr_trace = dl.compute_test2(nb_steps, burnin_ratio=burnin_ratio, nb_noise=nb_noise,
                                                                   thinning=1, thinning_noise=thinning_noise, normalize=True, 
                                                                   noise_schedule=noise_schedule.clone(), x=x.clone())
        
            local_res_dict = dict(trace=trace.numpy(), post_mean=post_mean.numpy(), psnr_trace=psnr_trace.numpy(), 
                                  img_path=img_path, cat=cat, y=y)
            dict_to_json(local_res_dict, os.path.join(save_folder, "trace_{}_{}_{}.json".format(ind_gt, l, curr_ind)))
            
        elif mode == "compute_psnr":
            psnr = dinv.loss.metric.PSNR()

            res = np.zeros([nb_samples, 256**2], dtype=np.float32)

            f = L2(sigma, blur_op)

            gradU = lambda t, y: f.grad(t, y) + g.grad(t, None)
        
            sampler = SKROCK(gradU, gamma, blur_op.A_A_adjoint(y).to(device).clone(), lambda t: t, s=15)
        
            for i in range(100):  # warm up
                sampler(y)
        
            for i in range(nb_steps):
                sampler(y)
                res[i] = sampler.X.cpu().numpy().reshape(-1)
            post_mean = np.mean(res, axis=0)
            psnr_val = psnr(torch.tensor(post_mean.reshape([1, 1, 256, 256])), x.cpu()).item()
            local_res_dict = dict(x=x.cpu().numpy(), y=y.cpu().numpy(), post_mean=post_mean, psnr=psnr_val)
            dict_to_json(local_res_dict, os.path.join(save_folder, "psnr_{}_{}_{}.json".format(ind_gt, l, curr_ind)))
            

