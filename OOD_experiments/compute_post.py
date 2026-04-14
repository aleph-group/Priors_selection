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
import tqdm

# parameters for the experiments
noise_level_img = 0.05  # noise added to the image after blurring
img_size = 256

niter_diffpir = 300
batch_size = 4



if len(sys.argv[1:]) == 4:
    out_folder = sys.argv[1]
    ind_start, ind_end = int(sys.argv[2]), int(sys.argv[3])
    save_folder = sys.argv[4]
else:
    ind_start, ind_end = 0, 74
config_file = json_to_dict(os.path.join(out_folder, "config.json"))

sigma_blur = config_file["sigma_blur"]
model_path = config_file["model_path"]
denoiser = dinv.models.DiffUNet(pretrained=model_path, in_channels=3, out_channels=3, large_model=False).to(device)

nb_samples = 40


for i in range(ind_end - ind_start + 1):
    curr_ind = i + ind_start
    print("Sampling for image {}".format(curr_ind))

    blur_op = generate_blur_operator(img_size, 
                                 filter_torch=dinv.physics.blur.gaussian_blur(sigma=(sigma_blur, sigma_blur)).to(device), 
                                 sigma=noise_level_img)
    res_dict = np.load(os.path.join(out_folder, "trace_{}.npz".format(curr_ind)))
    y = torch.tensor(res_dict["y"].astype(np.float32), device=device)
    

    sam = DiffPIR(lambda x:x, 0, blur_op.A_adjoint(y).to(device), batch_size=batch_size, physics=blur_op,
                  denoiser=denoiser, max_iter=niter_diffpir, proj=lambda x: x, lambda_=7, verbose=True)
    
    res = torch.zeros((nb_samples, 3, 256, 256))
    for i in range(nb_samples//batch_size):
        with torch.no_grad():
            res[batch_size*i:batch_size*(i+1)] = sam(y)
    
    np.save(os.path.join(save_folder, "samples_{}.npy".format(curr_ind)), res.cpu().numpy())
    #np.save(os.path.join(save_folder, "mean_{}.npy".format(curr_ind)), torch.mean(res, axis=0).cpu().numpy())


