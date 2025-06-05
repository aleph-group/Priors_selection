import deepinv as dinv
import torch
import numpy as np
import os
from prior_comparison import DegradedLikelihood
from utils import device, dict_to_json, json_to_dict
from PIL.Image import fromarray
from sampling import DiffPIR


folder = "figs_diff/ds_ffhqp2/"
save_folder = "results/diffusion/stats_exp"
meta_file_name = "res.json"

model_path = "models/diffusion/ffhq_p2.pt"
denoiser = dinv.models.DiffUNet(pretrained=model_path, in_channels=3, out_channels=3, large_model=False).to(device)

# parameters for the experiments
noise_level_img = 0.05  # noise added to the image after blurring
img_size = 256

nb_steps, nb_noise = 10, 50  # 10 steps for 50 noise iterations
niter_diffpir = 300
batch_size = 2

noise_schedule_path = "figs_diff/ds_ffhqp2/noise_schedule.npy"  # path for E_eps noise schedule (regenerated otherwise)
regenerate = True
if regenerate:
    noise_schedule = torch.randn((nb_noise,) + (3, img_size, img_size), device=device)*noise_level_img
    np.save(noise_schedule_path, noise_schedule.cpu().numpy())
else:
    noise_schedule = torch.tensor(np.load(noise_schedule_path).astype(np.float32)).to(device)

sigma_blur = 2.
blur_op = dinv.physics.BlurFFT(img_size=(1, img_size, img_size), 
                            filter=dinv.physics.blur.gaussian_blur(sigma=(sigma_blur, sigma_blur)).to(device), 
                            device=device, padding="circular",
                            noise_model=dinv.physics.GaussianNoise(sigma=noise_level_img))
alpha = 0.75
    
ind_start, ind_end = 0, 99

test_vals_log = np.zeros(ind_end - ind_start + 1)
test_vals = np.zeros(ind_end - ind_start + 1)
res_dict = dict(ind_bounds=(ind_start, ind_end), vals_log=test_vals_log, vals=test_vals,
                alpha=alpha, nb_noise=nb_noise, nb_steps=nb_steps, niter_diffpir=niter_diffpir,
                sigma_blur=sigma_blur)
alpha = torch.tensor(alpha)
for i in range(ind_start, ind_end):
    img_path = os.path.join(folder, "sample_ffhqc_{}.png".format(i))
    x = dinv.utils.load_image(img_path, device=device)
    y = blur_op(x)  # apply blur and noise

    dl = DegradedLikelihood(y, denoiser, blur_op, noise_level_img, 0, X_init=blur_op.A_A_adjoint(y).to(device).clone(),
                            sampler=DiffPIR,sampler_kwargs={'batch_size':batch_size, 'physics':blur_op, 
                                                            'denoiser':denoiser, 'max_iter': niter_diffpir}, 
                            project=None, alpha=alpha, batch_size=batch_size)

    trace, lmean = dl.compute_test2(nb_steps, burnin_ratio=0, nb_noise=nb_noise,
                                    thinning=1, thinning_noise=0, normalize=True, noise_schedule=noise_schedule)
    test_vals_log[i] = lmean
    test_vals[i] = np.mean(trace)
    np.save(os.path.join(save_folder, "trace_{}.npy".format(i)), trace)

dict_to_json(res_dict, os.path.join(save_folder, meta_file_name))